import pandas as pd

from src.challenge_agent.io.schema import infer_columns
from src.challenge_agent.scoring.risk_score import build_risk_score


def test_build_risk_score_favors_behavioral_digital_anomalies():
    df = pd.DataFrame(
        [
            {
                "transaction_id": "tx-1",
                "sender_id": "USER-A",
                "recipient_id": "SHOP-NEW",
                "transaction_type": "e-commerce",
                "amount": 120.0,
                "payment_method": "PayPal",
                "sender_iban": "IT00A",
                "recipient_iban": "FR00B",
                "balance_after": 40.0,
                "description": "",
                "timestamp": "2087-04-16T01:15:00",
                "_behavior_change": 0.9,
                "_recipient_rare": 1.0,
                "_counterparty_instability": 1.0,
                "_text_risk": 3.0,
            },
            {
                "transaction_id": "tx-2",
                "sender_id": "EMP-A",
                "recipient_id": "USER-A",
                "transaction_type": "transfer",
                "amount": 2800.0,
                "payment_method": "",
                "sender_iban": "IT99Z",
                "recipient_iban": "IT00A",
                "balance_after": 9000.0,
                "description": "Salary payment Apr",
                "timestamp": "2087-04-16T10:00:00",
                "_behavior_change": 0.0,
                "_recipient_rare": 0.0,
                "_counterparty_instability": 0.0,
                "_text_risk": 0.0,
            },
        ]
    )

    cols = infer_columns(df)
    scored = build_risk_score(df, cols, {})

    assert "_risk_score" in scored.columns
    assert scored.loc[0, "_risk_score"] > scored.loc[1, "_risk_score"]
