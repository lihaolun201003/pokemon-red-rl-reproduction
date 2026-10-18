"""PyBoy 2.4 wrapper used by the DeepSeek Pokémon Red agent.

Design rules (see ``ARCHITECTURE.md`` §5):

* Only this module imports PyBoy.  The decision loop never touches emulator
  internals and **never reads game RAM** — this class only exposes the
  framebuffer and the legal input surface.
* PyBoy is **not thread-safe**: every call must come from the single agent-loop
  thread.  No threads, locks or async code live here.
* ``frame_count`` is monotonic for the lifetime of the instance.  PyBoy's own
  counter is tracked through a base offset so a ``load_state`` (which may reset
  or preserve it depending on version) can never make the agent's frame counter
  move backwards.
* Only validated Game Boy buttons can be pressed; invalid input raises
  :class:`~llm_agent.contracts.ToolValidationError` before touching the emulator.
* All other emulator failures raise :class:`~llm_agent.contracts.EmulatorError`
  so the runtime can decide whether to retry, restore a checkpoint or stop.

Each button action advances exactly ``EmulatorConfig.action_frames`` game frames
(the RL baseline used ``press_frames=8`` held out of ``action_frames=24``),
reproducing ``neroRL``'s validated ``run_action_on_emulator`` timing::

    press button
    tick(press_frames, render=True)          # button held
    release button
    tick(action_frames - press_frames - 1)   # button released
    tick(1, render=True)                     # final rendered frame for capture

so a press costs ``press_frames + (action_frames - press_frames - 1) + 1``
= ``action_frames`` frames in total.
"""

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from llm_agent.config import EmulatorConfig
from llm_agent.contracts import (
    BUTTONS,
    MAX_ACTIONS_PER_CALL,
    MAX_WAIT_FRAMES,
    SIGNIFICANT_EVENTS,
    ActionOutcome,
    ActionStep,
    EmulatorError,
    Frame,
    ScreenEvent,
    ToolValidationError,
)
from llm_agent.observation import analyze, frame_from_rgba, save_png  # noqa: F401  (save_png re-exported for tools)

#: minimum frame budget for one press: the release event must be consumed
MIN_PRESS_FRAMES: int = 2
#: hard upper bound for a single observation wait (matches contracts)
MAX_TICK_FRAMES: int = MAX_WAIT_FRAMES


def _validate_button(button: Any) -> str:
    """Return the canonical lowercase button name or raise ``ToolValidationError``."""
    if not isinstance(button, str):
        raise ToolValidationError(f"button must be a string, got {type(button).__name__}")
    normalized = button.strip().lower()
    if normalized not in BUTTONS:
        raise ToolValidationError(
            f"invalid button {button!r}; allowed buttons: {', '.join(BUTTONS)}"
        )
    return normalized


def _validate_frames(frames: Any, *, minimum: int, maximum: int, name: str = "frames") -> int:
    """Validate an integer frame count (booleans are rejected on purpose)."""
    if isinstance(frames, bool) or not isinstance(frames, int):
        raise ToolValidationError(f"{name} must be an integer, got {type(frames).__name__}")
    if not minimum <= frames <= maximum:
        raise ToolValidationError(f"{name} must be in {minimum}..{maximum}, got {frames}")
    return frames


def _event_transition(baseline: ScreenEvent, current: ScreenEvent) -> bool:
    """True when the screen event *changed* into/out of a significant event.

    A persistent significant event (``DIALOG`` -> ``DIALOG`` while reading text
    pages) is **not** a transition and must not interrupt a batch.
    """
    if current == baseline:
        return False
    return current in SIGNIFICANT_EVENTS or baseline in SIGNIFICANT_EVENTS


def _atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically (unique temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            pass
    return path


class GameBoyEmulator:
    """Thin, validated wrapper around one PyBoy instance.

    Typical use::

        emu = GameBoyEmulator(cfg.emulator).start()
        try:
            frame = emu.observe()
            outcome = emu.run_batch(["up", "up", "a"], stop_on_event=True)
        finally:
            emu.close()
    """

    def __init__(self, cfg: EmulatorConfig) -> None:
        if cfg is None:
            raise EmulatorError("EmulatorConfig is required")
        self.cfg = cfg
        self._pyboy: Any = None
        self._frame_count: int = 0
        self._pyboy_marker: int = 0
        self._loaded_state: str | None = None
        self._last_frame: Frame | None = None
        self._rendered: bool = False
        #: optional real-time presentation hook (viewer) + pacing state
        self._frame_hook: Callable[[np.ndarray], None] | None = None
        self._presentation_fps: float = 0.0
        self._next_frame_at: float = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self, state_path: str | Path | None = None) -> "GameBoyEmulator":
        """Boot PyBoy (null window, fastest speed) and load a save state.

        ``state_path`` overrides ``cfg.state`` for this boot.  When neither is
        available the ROM boots to its title screen (used only by tooling).
        Returns ``self`` for chaining.
        """
        if self._pyboy is not None:
            if state_path is not None:
                self.load_state(state_path)
            return self

        rom = Path(self.cfg.rom)
        if not rom.is_file():
            raise EmulatorError(f"ROM not found: {rom}")
        window = str(self.cfg.window or "null")
        pyboy_class = self._load_pyboy_class()
        try:
            pyboy = pyboy_class(
                str(rom),
                window=window,
                sound_emulated=False,
                log_level="ERROR",
            )
        except Exception as exc:  # noqa: BLE001 - normalize PyBoy failures
            raise EmulatorError(f"failed to boot PyBoy with ROM {rom}: {exc}") from exc

        self._pyboy = pyboy
        self._frame_count = 0
        self._pyboy_marker = int(getattr(pyboy, "frame_count", 0) or 0)
        self._rendered = False
        self._last_frame = None
        self._loaded_state = None
        try:
            pyboy.set_emulation_speed(int(self.cfg.emulation_speed))
        except Exception as exc:  # noqa: BLE001 - not fatal for correctness
            self.close()
            raise EmulatorError(f"failed to set emulation speed: {exc}") from exc

        initial_state = state_path if state_path is not None else (self.cfg.state or None)
        if initial_state is not None:
            path = Path(initial_state)
            if path.is_file():
                self.load_state(path)
            elif state_path is not None:
                self.close()
                raise EmulatorError(f"initial save state not found: {path}")
            # else: configured state missing -> boot from the ROM title screen
        if not self._rendered:
            settle = max(1, int(getattr(self.cfg, "settle_frames", 1) or 1))
            self._tick(settle, render=True)
        return self

    def close(self) -> None:
        """Stop the emulator and discard cartridge-RAM changes (idempotent)."""
        pyboy = self._pyboy
        self._pyboy = None
        self._last_frame = None
        self._rendered = False
        if pyboy is None:
            return
        try:
            pyboy.stop(save=False)
        except Exception:  # noqa: BLE001 - shutdown must never crash the agent
            pass

    @property
    def started(self) -> bool:
        """True while a PyBoy instance is running."""
        return self._pyboy is not None

    def __enter__(self) -> "GameBoyEmulator":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # State save / load
    # ------------------------------------------------------------------ #
    def load_state(self, path: str | Path | bytes) -> None:
        """Load a PyBoy save state from a file path or raw state bytes.

        The agent's ``frame_count`` stays monotonic; stale queued inputs are
        discarded and the freshly loaded state is settled (``settle_frames``)
        so the following observation is rendered.
        """
        self._require_started("load_state")
        payload, label = self._read_state_payload(path)
        if not payload:
            raise EmulatorError(f"save state is empty: {label}")
        try:
            self._pyboy.load_state(io.BytesIO(payload))
        except Exception as exc:  # noqa: BLE001 - PyBoy raises generic exceptions
            raise EmulatorError(f"failed to load save state {label}: {exc}") from exc
        self._discard_pending_input()
        self._pyboy_marker = int(getattr(self._pyboy, "frame_count", 0) or 0)
        self._last_frame = None
        self._rendered = False
        self._loaded_state = label
        settle = int(getattr(self.cfg, "settle_frames", 0) or 0)
        if settle > 0:
            self._tick(settle, render=True)

    def save_state_bytes(self) -> bytes:
        """Serialize the complete emulator state to bytes (``BytesIO`` roundtrip)."""
        self._require_started("save_state_bytes")
        buffer = io.BytesIO()
        try:
            self._pyboy.save_state(buffer)
        except Exception as exc:  # noqa: BLE001
            raise EmulatorError(f"failed to save emulator state: {exc}") from exc
        payload = buffer.getvalue()
        if not payload:
            raise EmulatorError("failed to save emulator state: empty payload")
        return payload

    def save_state_file(self, path: str | Path) -> Path:
        """Atomically write the emulator state to ``path`` (size verified)."""
        payload = self.save_state_bytes()
        target = Path(path)
        _atomic_write_bytes(target, payload)
        if not target.is_file() or target.stat().st_size <= 0:
            raise EmulatorError(f"save state file was not written correctly: {target}")
        return target

    @property
    def loaded_state(self) -> str | None:
        """Label of the state loaded most recently (path or ``<memory>``)."""
        return self._loaded_state

    @staticmethod
    def _read_state_payload(path: str | Path | bytes) -> tuple[bytes, str]:
        if isinstance(path, (bytes, bytearray, memoryview)):
            payload = bytes(path)
            return payload, "<memory>"
        candidate = Path(path)
        if not candidate.is_file():
            raise EmulatorError(f"save state not found: {candidate}")
        try:
            payload = candidate.read_bytes()
        except OSError as exc:
            raise EmulatorError(f"failed to read save state {candidate}: {exc}") from exc
        return payload, str(candidate.resolve())

    def _discard_pending_input(self) -> None:
        """Drop queued/held inputs so they cannot leak into a restored state.

        ``queued_input`` holds delayed button releases and ``events`` holds
        events for the next tick; both are plain public PyBoy attributes.
        """
        pyboy = self._pyboy
        if pyboy is None:
            return
        for attribute in ("queued_input", "events"):
            try:
                container = getattr(pyboy, attribute, None)
                if isinstance(container, list):
                    container.clear()
            except Exception:  # noqa: BLE001 - attribute layouts are PyBoy internals
                continue

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def observe(self) -> Frame:
        """Capture the current framebuffer as a :class:`~llm_agent.contracts.Frame`.

        If nothing has been rendered since the last state load, one frame is
        ticked with rendering enabled so the buffer is guaranteed to be current.
        """
        self._require_started("observe")
        return self._capture_frame()

    @property
    def frame_count(self) -> int:
        """Monotonic count of emulated frames since ``start()``."""
        return self._frame_count

    @property
    def last_frame(self) -> Frame | None:
        """Most recently captured frame (or ``None``)."""
        return self._last_frame

    # ------------------------------------------------------------------ #
    # Real-time presentation (viewer only; harmless when unset)
    # ------------------------------------------------------------------ #
    @property
    def streaming(self) -> bool:
        """True when a frame hook is installed (real-time viewer mode)."""
        return self._frame_hook is not None

    def set_frame_hook(self, hook: Callable[[np.ndarray], None] | None, *, fps: float = 60.0) -> None:
        """Install a per-frame presentation callback for the live viewer.

        While installed, every emulated frame is rendered and passed to
        ``hook(rgba)`` and the emulator paces itself at ``fps`` so the viewer
        shows real-time movement instead of instantaneous jumps.  Headless runs
        never install a hook and keep the fast non-pacing path.
        """
        if hook is not None and not callable(hook):
            raise ToolValidationError("frame hook must be callable or None")
        self._frame_hook = hook
        self._presentation_fps = float(fps) if hook is not None else 0.0
        self._next_frame_at = 0.0

    def clear_frame_hook(self) -> None:
        self._frame_hook = None
        self._presentation_fps = 0.0

    def _present_frame(self) -> None:
        """Call the presentation hook with the current framebuffer (best effort)."""
        hook = self._frame_hook
        if hook is None or self._pyboy is None:
            return
        try:
            rgba = np.array(self._pyboy.screen.ndarray, dtype=np.uint8, copy=True, order="C")
            hook(rgba)
        except Exception:  # noqa: BLE001 - presentation must never break gameplay
            return
        fps = self._presentation_fps
        if fps <= 0:
            return
        now = time.perf_counter()
        if self._next_frame_at <= 0:
            self._next_frame_at = now
        delay = self._next_frame_at - now
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_frame_at = now
        self._next_frame_at += 1.0 / fps

    def _tick_streamed(self) -> None:
        """Advance exactly one rendered frame and present it to the viewer."""
        self._tick(1, render=True)
        self._present_frame()

    # ------------------------------------------------------------------ #
    # Input
    # ------------------------------------------------------------------ #
    def press(self, button: str, *, frames: int | None = None) -> Frame:
        """Press one validated button and return the resulting frame.

        ``frames`` optionally overrides ``cfg.action_frames`` for this action
        (2..600).  The sequence reproduces the validated RL baseline: press,
        hold ``press_frames``, release, then advance to the end of the
        ``frames`` budget and render the final frame for capture.  Exactly
        ``frames`` emulated frames pass in total.
        """
        self._require_started("press")
        canonical = _validate_button(button)
        action_frames = int(self.cfg.action_frames)
        if frames is None:
            budget = max(MIN_PRESS_FRAMES, action_frames)
        else:
            budget = _validate_frames(
                frames, minimum=MIN_PRESS_FRAMES, maximum=MAX_TICK_FRAMES, name="frames"
            )
        press_frames = int(getattr(self.cfg, "press_frames", 8) or 8)
        hold = max(1, min(press_frames, budget - 1))
        streamed = self._frame_hook is not None
        try:
            self._pyboy.button_press(canonical)
        except Exception as exc:  # noqa: BLE001 - PyBoy raises generic exceptions
            raise EmulatorError(f"failed to press button {canonical!r}: {exc}") from exc
        if streamed:
            for _ in range(hold):
                self._tick_streamed()
        else:
            self._tick(hold, render=True)
        try:
            self._pyboy.button_release(canonical)
        except Exception as exc:  # noqa: BLE001
            raise EmulatorError(f"failed to release button {canonical!r}: {exc}") from exc
        remaining = budget - hold - 1
        if streamed:
            for _ in range(max(0, remaining) + 1):
                self._tick_streamed()
        else:
            if remaining > 0:
                self._tick(remaining, render=True)
            self._tick(1, render=True)
        return self._capture_frame()

    def wait(self, frames: int) -> Frame:
        """Advance ``frames`` frames without input and return the new frame."""
        self._require_started("wait")
        budget = _validate_frames(
            frames, minimum=1, maximum=MAX_TICK_FRAMES, name="frames"
        )
        if self._frame_hook is not None:
            for _ in range(budget):
                self._tick_streamed()
        else:
            self._tick(budget, render=True)
        return self._capture_frame()

    def run_batch(
        self,
        buttons: Sequence[str],
        *,
        stop_on_event: bool = True,
        step_hook: Callable[[ActionStep], None] | None = None,
    ) -> ActionOutcome:
        """Press ``buttons`` in order, capturing one :class:`ActionStep` per press.

        Stops early when ``stop_on_event`` and an action *changes* the screen
        event relative to the event observed before the batch started (for
        example a new dialogue appearing, or an open dialogue closing).  A
        significant event that merely persists (``DIALOG`` -> ``DIALOG`` while
        pressing A through several text pages) does **not** stop the batch, so a
        multi-press dialogue can be read with one tool call.  A black screen
        appearing mid-batch also interrupts it.  ``stopped_reason`` is ``"event"``
        for such an interruption and ``"completed"`` when the full batch ran.
        Blocked movement is never hidden: the caller sees pixel-identical steps.

        The whole batch is validated before any frame is emulated.
        """
        self._require_started("run_batch")
        if isinstance(buttons, (str, bytes, bytearray)) or not isinstance(buttons, Sequence):
            raise ToolValidationError("buttons must be a sequence of button names")
        normalized = [_validate_button(button) for button in buttons]
        if not normalized:
            raise ToolValidationError("buttons must not be empty")
        if len(normalized) > MAX_ACTIONS_PER_CALL:
            raise ToolValidationError(
                f"at most {MAX_ACTIONS_PER_CALL} button presses per call, got {len(normalized)}"
            )
        if step_hook is not None and not callable(step_hook):
            raise ToolValidationError("step_hook must be callable or None")

        start_frame = self._frame_count
        previous = self._capture_frame()
        baseline_event = analyze(previous, None).event
        baseline_black = bool(previous.is_black)
        steps: list[ActionStep] = []
        events: list[ScreenEvent] = []
        stopped_reason = "completed"
        for button in normalized:
            frame = self.press(button)
            analysis = analyze(frame, previous)
            step = ActionStep(
                button=button,
                frame=frame,
                screen_changed=analysis.changed,
                changed_ratio=analysis.changed_ratio,
                event=analysis.event,
            )
            steps.append(step)
            if analysis.event != ScreenEvent.NONE:
                events.append(analysis.event)
            if step_hook is not None:
                step_hook(step)
            previous = frame
            if not stop_on_event:
                continue
            # Stop on an event *transition* only: a persistent DIALOG/MENU/BATTLE
            # must not interrupt a multi-press batch (reading dialogue pages).
            if _event_transition(baseline_event, analysis.event) or (
                analysis.is_black and not baseline_black
            ):
                stopped_reason = "event"
                break

        outcome = ActionOutcome(
            buttons=list(normalized),
            steps=steps,
            start_frame=start_frame,
            end_frame=self._frame_count,
            stopped_reason=stopped_reason,
            events=events,
            summary=self._batch_summary(
                normalized, steps, events, stopped_reason, baseline_event
            ),
        )
        return outcome

    @staticmethod
    def _batch_summary(
        requested: Sequence[str],
        steps: Sequence[ActionStep],
        events: Sequence[ScreenEvent],
        stopped_reason: str,
        baseline_event: ScreenEvent,
    ) -> str:
        changed = sum(1 for step in steps if step.screen_changed)
        final_event = steps[-1].event.value if steps else ScreenEvent.NONE.value
        return (
            f"{len(steps)}/{len(requested)} actions executed, {changed} changed screen, "
            f"start_event={baseline_event.value}, final_event={final_event}, "
            f"events={[event.value for event in events]}, stopped={stopped_reason}"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_pyboy_class() -> Any:
        """Import PyBoy lazily so pure tooling/tests can import this module."""
        try:
            from pyboy import PyBoy
        except Exception as exc:  # noqa: BLE001 - missing/broken PyBoy install
            raise EmulatorError(
                f"PyBoy is not importable in this environment: {type(exc).__name__}: {exc}"
            ) from exc
        return PyBoy

    def _require_started(self, operation: str) -> None:
        if self._pyboy is None:
            raise EmulatorError(f"emulator is not started (call start() before {operation}())")

    def _tick(self, frames: int, *, render: bool) -> None:
        """Tick PyBoy ``frames`` times and keep the monotonic counter in sync."""
        try:
            still_running = self._pyboy.tick(count=frames, render=render)
        except Exception as exc:  # noqa: BLE001
            raise EmulatorError(f"PyBoy tick({frames}) failed: {exc}") from exc
        self._sync_frame_counter()
        self._rendered = bool(render)
        if still_running is False:
            raise EmulatorError("PyBoy emulation ended unexpectedly (tick returned False)")

    def _sync_frame_counter(self) -> None:
        """Advance the monotonic frame counter from PyBoy's own counter."""
        pyboy_frames = int(getattr(self._pyboy, "frame_count", 0) or 0)
        delta = pyboy_frames - self._pyboy_marker
        if delta > 0:
            self._frame_count += delta
        # A state load in some PyBoy versions can reset the internal counter;
        # re-anchor without ever moving ``frame_count`` backwards.
        self._pyboy_marker = pyboy_frames

    def _capture_frame(self) -> Frame:
        """Copy the rendered screen buffer into an immutable ``Frame``."""
        if not self._rendered:
            self._tick(1, render=True)
        try:
            rgba = np.array(self._pyboy.screen.ndarray, dtype=np.uint8, copy=True, order="C")
        except Exception as exc:  # noqa: BLE001
            raise EmulatorError(f"failed to read the PyBoy screen buffer: {exc}") from exc
        frame = frame_from_rgba(self._frame_count, rgba)
        self._last_frame = frame
        return frame

    # ------------------------------------------------------------------ #
    # Convenience helpers (used by tooling/CLI diagnostics, optional)
    # ------------------------------------------------------------------ #
    def snapshot_png(self, path: str | Path) -> Path:
        """Save the current frame as a PNG (diagnostics only)."""
        return save_png(self.observe(), path)
