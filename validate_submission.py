import argparse
from pathlib import Path

from src.challenge_agent.io.loader import load_challenge_data
from src.challenge_agent.io.schema import infer_columns
from src.challenge_agent.validation.output_validator import (
    validate_ids_exist,
    validate_output_file,
)


def _resolve_dataset_dir(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None

    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
        return path.parent
    return path


def _load_output_ids(path: Path) -> list[str]:
    with open(path, "r", encoding="ascii") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one or more submission files.")
    parser.add_argument("--output", help="Optional output file to validate.")
    parser.add_argument(
        "--dataset-dir",
        "--dataset",
        dest="dataset_dir",
        help="Optional dataset directory used to validate line count and transaction IDs.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    outputs = [Path(args.output).expanduser().resolve()] if args.output else sorted((root / "outputs").rglob("*.txt"))

    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    total_transactions = None
    valid_ids = None
    if dataset_dir is not None:
        data = load_challenge_data(dataset_dir)
        cols = infer_columns(data.transactions)
        tx_id_col = cols["tx_id"]
        total_transactions = len(data.transactions)
        valid_ids = set(data.transactions[tx_id_col].astype(str))

    checked = 0
    for txt in outputs:
        validate_output_file(txt, total_transactions=total_transactions)
        if valid_ids is not None:
            validate_ids_exist(_load_output_ids(txt), valid_ids)
        print(f"[OK] {txt}")
        checked += 1

    if checked == 0:
        print("[WARN] no output files found")


if __name__ == "__main__":
    main()
