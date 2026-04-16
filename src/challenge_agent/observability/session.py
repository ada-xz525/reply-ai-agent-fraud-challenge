import json
import os
from pathlib import Path
import uuid

try:
    import ulid
except ImportError:
    ulid = None


def generate_session_id() -> str:
    team = os.getenv("TEAM_NAME", "team").replace(" ", "-")
    token = ulid.new().str if ulid is not None else uuid.uuid4().hex.upper()
    return f"{team}-{token}"


def save_session_record(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                payload = []
    else:
        payload = []

    if not isinstance(payload, list):
        payload = [payload]

    payload.append(record)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
