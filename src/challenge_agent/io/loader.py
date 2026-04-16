from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ChallengeData:
    transactions: pd.DataFrame
    locations: Optional[pd.DataFrame] = None
    users: Optional[pd.DataFrame] = None
    conversations: Optional[pd.DataFrame] = None
    messages: Optional[pd.DataFrame] = None
    audio_events: Optional[pd.DataFrame] = None


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_json(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return pd.json_normalize(payload)

    return pd.json_normalize([payload])


def _find_existing_file(dataset_dir: Path, candidates: list[str]) -> Optional[Path]:
    if not dataset_dir.exists():
        return None

    files_by_name = {
        path.name.lower(): path
        for path in dataset_dir.iterdir()
        if path.is_file()
    }

    for candidate in candidates:
        match = files_by_name.get(candidate.lower())
        if match is not None:
            return match

    return None


def _read_optional_table(dataset_dir: Path, candidates: list[str]) -> Optional[pd.DataFrame]:
    path = _find_existing_file(dataset_dir, candidates)
    if path is None:
        return None

    if path.suffix.lower() == ".json":
        return _read_json(path)

    return _read_csv(path)


def _read_audio_events(dataset_dir: Path) -> Optional[pd.DataFrame]:
    audio_dir = dataset_dir / "audio"
    if not audio_dir.exists() or not audio_dir.is_dir():
        return None

    rows: list[dict] = []
    for path in sorted(audio_dir.glob("*.mp3")):
        stem = path.stem
        if "-" not in stem:
            continue

        timestamp_token, contact_token = stem.split("-", 1)
        event_ts = pd.to_datetime(timestamp_token, format="%Y%m%d_%H%M%S", errors="coerce")
        if pd.isna(event_ts):
            continue

        rows.append(
            {
                "audio_path": str(path),
                "contact_name": contact_token.replace("_", " "),
                "event_ts": event_ts,
            }
        )

    if not rows:
        return None

    return pd.DataFrame(rows)


def load_challenge_data(dataset_dir: Path) -> ChallengeData:
    tx_path = _find_existing_file(dataset_dir, ["Transactions.csv", "transactions.csv"])
    if tx_path is None:
        raise FileNotFoundError(f"transactions.csv not found in {dataset_dir}")

    transactions = _read_csv(tx_path)
    locations = _read_optional_table(dataset_dir, ["Locations.csv", "locations.csv", "locations.json"])
    users = _read_optional_table(dataset_dir, ["Users.csv", "users.csv", "users.json"])
    conversations = _read_optional_table(
        dataset_dir,
        ["Conversations.csv", "conversations.csv", "sms.json"],
    )
    messages = _read_optional_table(
        dataset_dir,
        ["Messages.csv", "messages.csv", "mails.json", "emails.json"],
    )
    audio_events = _read_audio_events(dataset_dir)

    return ChallengeData(
        transactions=transactions,
        locations=locations,
        users=users,
        conversations=conversations,
        messages=messages,
        audio_events=audio_events,
    )
