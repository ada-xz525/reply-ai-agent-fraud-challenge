from pathlib import Path

import pytest

from src.challenge_agent.validation.output_validator import validate_output_file


def test_validate_output_file_accepts_valid_output(tmp_path: Path):
    output_path = tmp_path / "submission.txt"
    output_path.write_text("tx-1\ntx-2\n", encoding="ascii")

    validate_output_file(output_path, total_transactions=10)


def test_validate_output_file_rejects_duplicates(tmp_path: Path):
    output_path = tmp_path / "submission.txt"
    output_path.write_text("tx-1\ntx-1\n", encoding="ascii")

    with pytest.raises(ValueError, match="duplicate"):
        validate_output_file(output_path, total_transactions=10)


def test_validate_output_file_rejects_all_rows(tmp_path: Path):
    output_path = tmp_path / "submission.txt"
    output_path.write_text("tx-1\ntx-2\n", encoding="ascii")

    with pytest.raises(ValueError, match="all transactions"):
        validate_output_file(output_path, total_transactions=2)
