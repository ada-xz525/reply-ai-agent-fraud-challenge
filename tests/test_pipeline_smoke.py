from pathlib import Path

import pandas as pd

from src.challenge_agent.pipeline import run_pipeline
from src.challenge_agent.settings import load_settings
from src.challenge_agent.validation.output_validator import (
    validate_ids_exist,
    validate_output_file,
)


def _make_dataset_copy(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    source = Path(__file__).resolve().parents[1] / "data" / "raw" / "The Truman Show - train" / "transactions.csv"
    dataset_dir = tmp_path / "sample_level"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source).head(40)
    df.to_csv(dataset_dir / "transactions.csv", index=False)
    return dataset_dir, df


def test_pipeline_runs_end_to_end_without_optional_llm_dependencies(tmp_path: Path):
    dataset_dir, df = _make_dataset_copy(tmp_path)
    output_path = tmp_path / "submission.txt"

    result = run_pipeline(
        dataset_dir=dataset_dir,
        output_path=output_path,
        session_id="test-session",
        settings=load_settings(),
    )

    assert output_path.exists()
    assert 0 < result["predicted_count"] < len(df)

    validate_output_file(output_path, total_transactions=len(df))
    validate_ids_exist(
        output_path.read_text(encoding="ascii").splitlines(),
        set(df["transaction_id"].astype(str)),
    )
