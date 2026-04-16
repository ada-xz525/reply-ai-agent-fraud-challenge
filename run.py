import argparse
from pathlib import Path

from src.challenge_agent.observability.session import (
    generate_session_id,
    save_session_record,
)
from src.challenge_agent.observability.tracing import langfuse_client
from src.challenge_agent.pipeline import run_pipeline
from src.challenge_agent.settings import load_settings
from src.challenge_agent.validation.output_validator import (
    validate_ids_exist,
    validate_output_file,
)


def _resolve_dataset_dir(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
        return path.parent
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Reply Mirror fraud detector on a challenge dataset.",
    )
    parser.add_argument(
        "--dataset-dir",
        "--dataset",
        dest="dataset_dir",
        required=True,
        help="Dataset directory containing transactions.csv/Transactions.csv or a direct path to the transactions file.",
    )
    parser.add_argument("--mode", default="training", choices=["training", "evaluation"])
    args = parser.parse_args()

    settings = load_settings()
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    session_id = generate_session_id()

    output_dir = settings.project_root / "outputs" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_dir.name}_output.txt"

    result = run_pipeline(
        dataset_dir=dataset_dir,
        output_path=output_path,
        session_id=session_id,
        settings=settings,
    )

    validate_output_file(output_path, total_transactions=result["total_transactions"])
    validate_ids_exist(result["final_ids"], result["valid_tx_ids"])

    save_session_record(
        settings.project_root / "sessions" / "session_registry.json",
        {
            "session_id": session_id,
            "dataset_dir": str(dataset_dir),
            "output_path": str(output_path),
            "mode": args.mode,
            "total_transactions": result["total_transactions"],
            "predicted_count": result["predicted_count"],
            "candidate_count": result["candidate_count"],
        },
    )

    langfuse_client.flush()

    print(f"[OK] output written to: {output_path}")
    print(f"[OK] session_id: {session_id}")
    print(f"[OK] total transactions: {result['total_transactions']}")
    print(f"[OK] predicted suspicious: {result['predicted_count']}")


if __name__ == "__main__":
    main()
