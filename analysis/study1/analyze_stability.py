"""Offline pointer for Task 004 stability analysis. Makes no paid API calls."""

from __future__ import annotations

from src.study1_repeats import (
    RETURN_DIR,
    audit_task003_parse_repairs,
    reconstruct_repeats_payload,
    write_parse_repair_audit,
)
from src.study1_stability import write_stability_analysis_bundle


def main() -> None:
    audit = audit_task003_parse_repairs()
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    write_parse_repair_audit(audit, RETURN_DIR / "preflight_parse_repair_audit.md")
    payload = reconstruct_repeats_payload()
    write_stability_analysis_bundle(payload, RETURN_DIR)
    print(
        {
            "audit_proceed": audit["proceed"],
            "paid_calls": 0,
            "rewrote": str(RETURN_DIR),
        }
    )


if __name__ == "__main__":
    main()
