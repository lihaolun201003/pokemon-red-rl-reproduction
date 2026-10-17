import argparse
import hashlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def emulator_probe(window="null"):
    """Original tiny test cartridge, no Nintendo ROM or game assets."""
    from pyboy import PyBoy
    with tempfile.TemporaryDirectory(prefix="phase1_pyboy_") as directory:
        rom = bytearray(32768)
        # Entry and endless JR -2. Header has a non-Nintendo title, ROM-only mapper.
        rom[0x100:0x104] = bytes([0x00, 0xC3, 0x50, 0x01])
        rom[0x134:0x13C] = b"RLPROBE\0"
        rom[0x150:0x152] = bytes([0x18, 0xFE])
        checksum = 0
        for byte in rom[0x134:0x14D]:
            checksum = (checksum - byte - 1) & 255
        rom[0x14D] = checksum
        path = Path(directory) / "probe.gb"
        path.write_bytes(rom)
        emulator = PyBoy(str(path), window=window, sound_emulated=False)
        try:
            emulator.set_emulation_speed(0)
            emulator.tick(120, True)
            assert emulator.screen.ndarray.shape == (144, 160, 4)
            state = io.BytesIO()
            emulator.save_state(state)
            emulator.tick(2, True)
            state.seek(0)
            emulator.load_state(state)
            emulator.tick(2, True)
        finally:
            emulator.stop(save=False)
    return {"cartridge": "original synthetic diagnostic; NOT Pokemon", "window": window,
            "tick_save_load_close": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/doctor.json")
    args = parser.parse_args()
    from reproduction.runtime import ROOT, metadata, write_json
    report = metadata()
    report["imports"] = {}
    for name in ["torch", "pyboy", "gymnasium", "neroRL", "cv2", "tensorboard"]:
        importlib.import_module(name)
        try:
            version = importlib.metadata.version(name if name != "cv2" else "opencv-python")
        except importlib.metadata.PackageNotFoundError:
            version = "source import"
        report["imports"][name] = version
    import torch
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        x = torch.randn(16, 16, device=device, requires_grad=True)
        (x @ x.T).square().mean().backward()
        assert torch.isfinite(x.grad).all()
        report[f"{device}_matmul_backward"] = "passed"
    report["pyboy_probe"] = emulator_probe()
    from reproduction.runtime import ROM_SHA1
    candidates = sorted({*ROOT.glob("local/*.gb"), *ROOT.glob("local/*.gbc")})
    report["rom_candidates"] = [{"filename": p.name, "bytes": p.stat().st_size,
        "sha1": hashlib.sha1(p.read_bytes()).hexdigest()} for p in candidates]
    report["rom_present"] = bool(candidates)
    report["canonical_rom_present"] = any(p["sha1"] == ROM_SHA1 for p in report["rom_candidates"])
    report["state_present"] = (ROOT / "local/has_pokedex_nballs.state").is_file()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
