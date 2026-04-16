import pandas as pd

from src.challenge_agent.io.schema import infer_columns


def test_infer_columns_with_challenge_headers():
    df = pd.DataFrame(
        columns=[
            "Transaction ID",
            "Sender ID",
            "Recipient ID",
            "Transaction Type",
            "Amount",
            "Location",
            "Payment Method",
            "Sender IBAN",
            "Recipient IBAN",
            "Balance",
            "Timestamp",
        ]
    )

    cols = infer_columns(df)

    assert cols["tx_id"] == "Transaction ID"
    assert cols["sender_id"] == "Sender ID"
    assert cols["recipient_id"] == "Recipient ID"
    assert cols["transaction_type"] == "Transaction Type"
    assert cols["payment_method"] == "Payment Method"
    assert cols["sender_iban"] == "Sender IBAN"
    assert cols["recipient_iban"] == "Recipient IBAN"


def test_infer_columns_with_lowercase_headers():
    df = pd.DataFrame(
        columns=[
            "transaction_id",
            "sender_id",
            "recipient_id",
            "transaction_type",
            "amount",
            "location",
            "payment_method",
            "sender_iban",
            "recipient_iban",
            "balance_after",
            "description",
            "timestamp",
        ]
    )

    cols = infer_columns(df)

    assert cols["tx_id"] == "transaction_id"
    assert cols["sender_id"] == "sender_id"
    assert cols["recipient_id"] == "recipient_id"
    assert cols["balance"] == "balance_after"
    assert cols["description"] == "description"
