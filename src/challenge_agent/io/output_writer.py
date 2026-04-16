from pathlib import Path
from typing import Iterable


def _stable_unique_tx_ids(tx_ids: Iterable[str]) -> list[str]:
    seen = set()
    out = []

    for x in tx_ids:
        x = str(x).strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out


def validate_ascii_lines(lines: list[str]) -> None:
    for i, line in enumerate(lines, start=1):
        try:
            line.encode("ascii")
        except UnicodeEncodeError as e:
            raise ValueError(f"Line {i} is not ASCII: {line}") from e


def write_output(tx_ids: Iterable[str], output_path: str) -> None:
    """
    Competition format:
    - ASCII text file
    - one suspected fraudulent transaction_id per line
    """
    lines = _stable_unique_tx_ids(tx_ids)
    validate_ascii_lines(lines)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="ascii", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")