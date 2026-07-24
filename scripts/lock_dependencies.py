from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

UV_VERSION = "0.11.32"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "requirements"

LOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("production.lock", ()),
    ("collection.lock", ("--extra", "collection")),
    ("training.lock", ("--extra", "training")),
    (
        "validation.lock",
        ("--extra", "collection", "--extra", "ranking", "--extra", "dev"),
    ),
    (
        "development.lock",
        ("--extra", "research", "--extra", "dev"),
    ),
)


def _uv_executable() -> Path:
    discovered = shutil.which("uv")
    candidates = (
        Path(discovered) if discovered else None,
        ROOT / ".venv" / "Scripts" / "uv.exe",
        ROOT / ".venv" / "bin" / "uv",
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise RuntimeError(
        f"uv {UV_VERSION} is required; install it with "
        f"`python -m pip install uv=={UV_VERSION}`"
    )


def main() -> None:
    uv = _uv_executable()
    resolved_version = subprocess.run(
        [str(uv), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not resolved_version.startswith(f"uv {UV_VERSION} "):
        raise RuntimeError(
            f"Expected uv {UV_VERSION}, found {resolved_version!r}"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    common = (
        str(uv),
        "pip",
        "compile",
        str(ROOT / "pyproject.toml"),
        "--universal",
        "--python-version",
        "3.11",
        "--generate-hashes",
        "--no-progress",
        "--quiet",
        "--custom-compile-command",
        "python scripts/lock_dependencies.py",
    )
    for output_name, extra_arguments in LOCKS:
        subprocess.run(
            [
                *common,
                *extra_arguments,
                "--output-file",
                str(OUTPUT_DIRECTORY / output_name),
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
