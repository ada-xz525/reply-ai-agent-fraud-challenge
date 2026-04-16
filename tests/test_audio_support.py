from pathlib import Path

import pandas as pd

from src.challenge_agent.io.loader import load_challenge_data
from src.challenge_agent.tools.text_signal_tool import build_text_risk_features


def test_loader_reads_audio_event_metadata(tmp_path: Path):
    dataset_dir = tmp_path / "level"
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "transaction_id": "tx-1",
                "sender_id": "USR-1",
                "sender_iban": "IBAN-1",
                "recipient_id": "R-1",
                "amount": 10.0,
                "timestamp": "2087-01-17T03:00:00",
            }
        ]
    ).to_csv(dataset_dir / "transactions.csv", index=False)
    pd.DataFrame(
        [
            {
                "first_name": "Guido",
                "last_name": "Döhn",
                "iban": "IBAN-1",
            }
        ]
    ).to_json(dataset_dir / "users.json", orient="records")
    (audio_dir / "20870117_010505-guido_döhn.mp3").write_bytes(b"fake-mp3")

    data = load_challenge_data(dataset_dir)

    assert data.audio_events is not None
    assert len(data.audio_events) == 1
    assert data.audio_events.iloc[0]["contact_name"] == "guido döhn"


def test_audio_events_raise_text_risk_for_nearby_transactions():
    transactions = pd.DataFrame(
        [
            {
                "transaction_id": "tx-1",
                "sender_id": "USR-1",
                "sender_iban": "IBAN-1",
                "recipient_id": "R-1",
                "amount": 55.0,
                "timestamp": "2087-01-17T08:00:00",
            },
            {
                "transaction_id": "tx-2",
                "sender_id": "USR-1",
                "sender_iban": "IBAN-1",
                "recipient_id": "R-2",
                "amount": 25.0,
                "timestamp": "2087-01-25T08:00:00",
            },
        ]
    )
    users = pd.DataFrame(
        [
            {
                "first_name": "Guido",
                "last_name": "Döhn",
                "iban": "IBAN-1",
            }
        ]
    )
    audio_events = pd.DataFrame(
        [
            {"contact_name": "guido döhn", "event_ts": "2087-01-17T01:05:05"},
        ]
    )
    cols = {
        "tx_id": "transaction_id",
        "sender_id": "sender_id",
        "sender_iban": "sender_iban",
        "timestamp": "timestamp",
    }

    text_risk = build_text_risk_features(
        transactions=transactions,
        cols=cols,
        users=users,
        conversations=None,
        messages=None,
        audio_events=audio_events,
    ).set_index("transaction_id")["_text_risk"]

    assert text_risk["tx-1"] > 0.0
    assert text_risk["tx-2"] == 0.0
