"""Real-time Tk viewer for the DeepSeek Pokémon Red agent.

The viewer runs on the **main thread** (Tk requirement).  The agent runtime runs
on a worker thread and communicates only through a thread-safe queue and
``threading.Event`` flags, so the emulator is never touched from the GUI thread.

Layout:
    +--------------------------------------+-----------------------------+
    |              game screen             |  status / goal / last act   |
    |              (nearest upscale)       |  decision summary / stats   |
    +--------------------------------------+-----------------------------+
    [Pause / Resume] [Save progress] [Quit safely]

Keyboard: Space = pause/resume, S = save, Q/Esc = safe quit.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm_agent.config import ViewerConfig

_QUEUE_SIZE = 8


@dataclass
class ViewerStatus:
    """Status fields rendered in the side panel (all observation/API derived)."""

    status: str = "starting"
    goal: str = ""
    last_action: str = ""
    decision: str = ""
    frame: int = 0
    turns: int = 0
    buttons: int = 0
    places: int = 0
    discoveries: int = 0
    map_cells: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    tokens: int = 0
    budget_left_usd: float = 0.0
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_lines(self) -> list[tuple[str, str]]:
        status_text = {
            "starting": "启动中…",
            "running": "运行中（AI 已决策）",
            "waiting_api": "等待 AI 决策…（约 5-6 秒）",
            "acting": "执行动作 / 回放画面…",
            "paused": "已暂停（可点继续）",
            "saving": "正在保存…",
            "blocked": "已阻塞（需人工处理）",
            "paused_budget": "预算暂停（已安全保存）",
            "finished": "已完成",
            "stopped": "已停止（安全退出）",
        }.get(self.status, self.status)
        lines = [
            ("状态 Status", status_text),
            ("目标 Goal", _clip(self.goal, 160) or "-"),
            ("最近动作 Last action", _clip(self.last_action, 160) or "-"),
            ("决策摘要 Decision", _clip(self.decision, 220) or "-"),
            ("步数 Frame / Turns", f"{self.frame} / {self.turns}"),
            ("按键数 Buttons", str(self.buttons)),
            ("已发现地点 Places", f"{self.places}（地图 {self.map_cells} 格）"),
            ("发现记录 Discoveries", str(self.discoveries)),
            ("API 调用 Calls", str(self.calls)),
            ("累计费用 Cost", f"${self.cost_usd:.4f}"),
            ("Token 总数", f"{self.tokens:,}"),
            ("预算余额 Budget left", f"${self.budget_left_usd:.4f}"),
            ("模型 Model", self.model or "-"),
        ]
        for key, value in self.extra.items():
            lines.append((key, str(value)))
        return lines


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class AgentViewer:
    """Thread-safe Tk viewer; instantiate on the main thread and call start()."""

    def __init__(self, cfg: ViewerConfig) -> None:
        self.cfg = cfg
        self.pause_event = threading.Event()
        self.save_event = threading.Event()
        self.stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_SIZE)
        self._status_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_SIZE)
        self._lock = threading.Lock()
        self._playback_frames: list[np.ndarray] = []
        self._playback_index = 0
        self._playback_fps = 2.5
        self._playback_next_at = 0.0
        self._latest_frame: np.ndarray | None = None
        self._shutdown = threading.Event()
        self._closed = threading.Event()
        self._started_at = time.time()
        self._last_status = ViewerStatus()
        self._last_rgba: np.ndarray | None = None
        self._spin_phase = 0
        self._spin_last = 0.0
        # Tk objects (main thread only)
        self._root: Any = None
        self._image_label: Any = None
        self._photo: Any = None
        self._status_vars: dict[str, Any] = {}
        self._pause_button: Any = None
        self._error: str | None = None

    # ------------------------------------------------------------------ #
    # Worker-thread API
    # ------------------------------------------------------------------ #
    def post(self, rgba: np.ndarray, status: ViewerStatus) -> None:
        """Publish the latest frame + status (non-blocking, drops old frames)."""
        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                self._status_queue.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            self._latest_frame = np.ascontiguousarray(rgba)
        try:
            self._frame_queue.put_nowait(np.ascontiguousarray(rgba))
        except queue.Full:
            pass
        try:
            self._status_queue.put_nowait(status)
        except queue.Full:
            pass

    def post_frame(self, rgba: np.ndarray) -> None:
        """Publish a single live frame (60 fps streaming; cheap, drops old frames)."""
        frame = np.ascontiguousarray(rgba)
        with self._lock:
            self._latest_frame = frame
        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def post_playback(self, frames: list[np.ndarray], *, fps: float | None = None) -> None:
        """Queue a smooth real-time replay of the action frames while the AI thinks.

        Each action frame represents ``action_frames`` emulated frames (e.g. 24 =
        0.4 s of game time), so replaying them at ``60/action_frames`` fps shows
        the movement at the game's normal walking speed instead of jumping.
        """
        if not frames:
            return
        fps = float(fps or self._playback_fps) or self._playback_fps
        prepared = [np.ascontiguousarray(frame) for frame in frames]
        with self._lock:
            self._playback_frames = prepared
            self._playback_index = 0
            self._playback_fps = max(0.5, min(30.0, fps))
            self._playback_next_at = time.time()

    def request_shutdown(self) -> None:
        self._shutdown.set()

    def is_closed(self) -> bool:
        return self._closed.is_set()

    def wait_closed(self, timeout: float = 5.0) -> bool:
        return self._closed.wait(timeout)

    # ------------------------------------------------------------------ #
    # Main-thread API
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Create the window and block in Tk's main loop until closed."""
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except Exception as exc:  # pragma: no cover - GUI environment problems
            self._error = f"Tk viewer unavailable: {exc}"
            self._closed.set()
            raise RuntimeError(self._error) from exc

        self._tk = tk
        self._pil_image = Image
        self._pil_imagetk = ImageTk
        root = tk.Tk()
        self._root = root
        root.title(self.cfg.title)
        root.resizable(False, False)  # fixed layout: the game canvas must never be clipped
        root.protocol("WM_DELETE_WINDOW", self._on_quit)
        root.bind("<space>", lambda _e: self._toggle_pause())
        root.bind("<KeyPress-s>", lambda _e: self._request_save())
        root.bind("<KeyPress-S>", lambda _e: self._request_save())
        root.bind("<KeyPress-q>", lambda _e: self._on_quit())
        root.bind("<Escape>", lambda _e: self._on_quit())
        root.configure(bg="#101418")

        outer = tk.Frame(root, bg="#101418")
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._image_label = tk.Label(outer, bg="#000000", bd=2, relief=tk.SUNKEN)
        self._image_label.grid(row=0, column=0, sticky="nw")

        panel = tk.Frame(outer, bg="#101418", padx=10, pady=4)
        panel.grid(row=0, column=1, sticky="nsew")
        outer.columnconfigure(1, weight=1)

        title = tk.Label(
            panel,
            text="DeepSeek · Pokémon Red",
            fg="#9fd3ff",
            bg="#101418",
            font=("Segoe UI", 13, "bold"),
            anchor="w",
        )
        title.pack(fill=tk.X, pady=(0, 6))

        for label in (
            "状态 Status",
            "目标 Goal",
            "最近动作 Last action",
            "决策摘要 Decision",
            "步数 Frame / Turns",
            "按键数 Buttons",
            "已发现地点 Places",
            "发现记录 Discoveries",
            "API 调用 Calls",
            "累计费用 Cost",
            "Token 总数",
            "预算余额 Budget left",
            "模型 Model",
        ):
            row = tk.Frame(panel, bg="#101418")
            row.pack(fill=tk.X, pady=1)
            tk.Label(
                row,
                text=label,
                fg="#7f8c99",
                bg="#101418",
                font=("Segoe UI", 9),
                width=18,
                anchor="w",
            ).pack(side=tk.LEFT)
            value = tk.StringVar(value="-")
            self._status_vars[label] = value
            tk.Label(
                row,
                textvariable=value,
                fg="#e6edf3",
                bg="#101418",
                font=("Segoe UI", 10),
                anchor="w",
                justify=tk.LEFT,
                wraplength=380,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        controls = tk.Frame(panel, bg="#101418", pady=8)
        controls.pack(fill=tk.X, side=tk.BOTTOM)
        self._pause_button = tk.Button(
            controls,
            text="暂停 Pause",
            command=self._toggle_pause,
            bg="#1f6feb",
            fg="white",
            activebackground="#388bfd",
            activeforeground="white",
            width=14,
        )
        self._pause_button.pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            controls,
            text="保存进度 Save",
            command=self._request_save,
            bg="#238636",
            fg="white",
            activebackground="#2ea043",
            activeforeground="white",
            width=14,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            controls,
            text="安全退出 Quit",
            command=self._on_quit,
            bg="#8b2e2e",
            fg="white",
            activebackground="#b33a3a",
            activeforeground="white",
            width=14,
        ).pack(side=tk.LEFT)

        hint = tk.Label(
            panel,
            text="Space 暂停/继续 · S 保存 · Q/Esc 安全退出\n"
            "暂停与安全退出只记录请求，运行时会在下一边界响应。",
            fg="#58636d",
            bg="#101418",
            font=("Segoe UI", 8),
            anchor="w",
            justify=tk.LEFT,
        )
        hint.pack(fill=tk.X, side=tk.BOTTOM, pady=(2, 0))

        root.after(self.cfg.refresh_ms, self._poll)
        try:
            root.mainloop()
        finally:
            self._closed.set()
            try:
                root.destroy()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Callbacks / polling
    # ------------------------------------------------------------------ #
    def _toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
        else:
            self.pause_event.set()
        self._sync_pause_button()

    def _request_save(self) -> None:
        self.save_event.set()

    def _on_quit(self) -> None:
        self.stop_event.set()
        self._shutdown.set()

    def _sync_pause_button(self) -> None:
        if self._pause_button is None:
            return
        try:
            self._pause_button.configure(text="继续 Resume" if self.pause_event.is_set() else "暂停 Pause")
        except Exception:
            pass

    def _poll(self) -> None:
        if self._root is None:
            return
        self._render_playback()
        try:
            rgba = None
            while True:
                rgba = self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        if rgba is not None:
            self._last_rgba = rgba
            # Do not jump ahead of an ongoing real-time playback of the actions.
            with self._lock:
                playing = self._playback_index < len(self._playback_frames)
            if not playing:
                self._render_frame(rgba)
        status = None
        try:
            while True:
                status = self._status_queue.get_nowait()
        except queue.Empty:
            pass
        if status is not None:
            self._last_status = status
            self._render_status(status)
        self._animate_waiting()
        self._sync_pause_button()
        if self._shutdown.is_set():
            self._closed.set()
            try:
                self._root.destroy()
            except Exception:
                pass
            return
        try:
            self._root.after(self.cfg.refresh_ms, self._poll)
        except Exception:
            self._closed.set()

    def _render_playback(self) -> None:
        """Advance the real-time action playback (Tk main thread only)."""
        with self._lock:
            frames = self._playback_frames
            index = self._playback_index
            if not frames or index >= len(frames):
                return
            now = time.time()
            if now < self._playback_next_at:
                return
            frame = frames[index]
            self._playback_index += 1
            self._playback_next_at = now + (1.0 / self._playback_fps)
            if self._playback_index >= len(frames):
                self._playback_frames = []
                self._playback_index = 0
                latest = self._latest_frame
            else:
                latest = None
        self._render_frame(frame)
        if latest is not None:
            self._render_frame(latest)

    def _render_frame(self, rgba: np.ndarray) -> None:
        try:
            image = self._pil_image.fromarray(rgba[..., :3], mode="RGB")
            scale = max(1, int(self.cfg.scale))
            if scale != 1:
                image = image.resize(
                    (image.width * scale, image.height * scale),
                    self._pil_image.NEAREST,
                )
            self._photo = self._pil_imagetk.PhotoImage(image)
            self._image_label.configure(image=self._photo)
        except Exception:
            pass

    def _animate_waiting(self) -> None:
        """Animate a small spinner while the AI request is in flight."""
        status_var = self._status_vars.get("状态 Status")
        if status_var is None:
            return
        if self._last_status.status not in ("waiting_api", "acting"):
            return
        now = time.time()
        if now - self._spin_last < 0.25:
            return
        self._spin_last = now
        self._spin_phase = (self._spin_phase + 1) % 4
        dots = "·" * self._spin_phase + " " * (3 - self._spin_phase)
        if self._last_status.status == "waiting_api":
            status_var.set(f"等待 AI 决策…（约 5-6 秒，画面实时运行）{dots}")
        else:
            status_var.set(f"执行动作 / 回放画面…{dots}")

    def _render_status(self, status: ViewerStatus) -> None:
        try:
            for label, value in status.to_lines():
                variable = self._status_vars.get(label)
                if variable is not None:
                    variable.set(value)
            if status.status in ("waiting_api", "acting"):
                self._spin_last = 0.0  # refresh the spinner text immediately
        except Exception:
            pass
