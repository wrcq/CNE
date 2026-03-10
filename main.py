"""Unified project entrypoint.

Runs all demo scripts in sequence.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(name: str) -> None:
    script = ROOT / "scripts" / name
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    run_script("demo_grid.py")
    run_script("demo_random.py")
    run_script("compare_k.py")
    print("\nOK: all demos completed. See output/ for generated images.")


if __name__ == "__main__":
    main()
