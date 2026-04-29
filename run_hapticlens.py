from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def add_dll_path(path: Path) -> None:
    if path.exists():
        os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))


def configure_local_runtime() -> None:
    nvidia = ROOT / ".venv" / "Lib" / "site-packages" / "nvidia"
    for rel in (
        "cublas/bin",
        "cufft/bin",
        "cuda_runtime/bin",
        "cuda_nvrtc/bin",
    ):
        add_dll_path(nvidia / rel)

    ffmpeg_bin = ROOT / "tools" / "ffmpeg" / "ffmpeg-8.1-essentials_build" / "bin"
    os.environ.setdefault("FFMPEG_PATH", str(ffmpeg_bin / "ffmpeg.exe"))
    os.environ.setdefault("FFPROBE_PATH", str(ffmpeg_bin / "ffprobe.exe"))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_hapticlens.py <script.py> [args...]")

    configure_local_runtime()
    src_dir = str(ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    script = Path(sys.argv[1])
    if not script.is_absolute():
        script = ROOT / script

    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
