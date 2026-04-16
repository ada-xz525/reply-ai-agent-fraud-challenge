from pathlib import Path

from src.challenge_agent.io.output_writer import write_output


def test_write_output_deduplicates_and_preserves_ascii(tmp_path: Path):
    output_path = tmp_path / "submission.txt"
    write_output(["tx-1", "tx-2", "tx-1", "", "  "], str(output_path))

    assert output_path.read_text(encoding="ascii") == "tx-1\ntx-2\n"
