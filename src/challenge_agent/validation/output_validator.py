from pathlib import Path
from typing import Iterable


def validate_output_file(output_path: Path, total_transactions: int | None = None) -> None:
    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")

    with open(output_path, "r", encoding="ascii") as f:
        lines = [line.rstrip("\n") for line in f]

    if any(not line.strip() for line in lines):
        raise ValueError("Output contains empty lines")

    if len(lines) != len(set(lines)):
        raise ValueError("Output contains duplicate transaction IDs")

    if len(lines) == 0:
        raise ValueError("Invalid submission: no transactions are reported")

    if total_transactions is not None and len(lines) >= total_transactions:
        raise ValueError("Invalid submission: all transactions are reported")


def validate_ids_exist(predicted_ids: Iterable[str], valid_ids: set[str]) -> None:
    bad = [x for x in predicted_ids if x not in valid_ids]
    if bad:
        raise ValueError(f"Output contains unknown transaction IDs, sample={bad[:5]}")
