"""Forward to the shared synthetic-conversations attribute filter."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parents[2]
        / "synthetic_conversations"
        / "scripts"
        / "filter_by_user_attributes.py"
    )
    runpy.run_path(str(target), run_name="__main__")
