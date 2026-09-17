"""Offline pointer for the Study 1 primary analysis module.

Does not make API calls. Task 003 writes artifacts through
src.study1_analysis.write_primary_analysis_bundle.
"""

from __future__ import annotations

from src.study1_primary import RETURN_DIR


def main() -> None:
    raise SystemExit(
        "Do not use this script to launch paid calls. "
        "Task 003 execution is `python -m src.cli study1-run --phase primary --yes --max-calls 2800`. "
        f"Analysis helper lives in src/study1_analysis.py; outputs go to {RETURN_DIR}."
    )


if __name__ == "__main__":
    main()
