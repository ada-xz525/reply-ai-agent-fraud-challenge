import re
from typing import Dict, Optional

import pandas as pd


COLUMN_CANDIDATES = {
    "tx_id": ["transaction_id", "transaction id", "tx_id", "id", "payment_id", "event_id"],
    "sender_id": ["sender_id", "sender id", "account_id", "user_id", "customer_id", "client_id"],
    "recipient_id": ["recipient_id", "recipient id", "merchant_id", "merchant id", "beneficiary_id"],
    "amount": ["amount", "transaction_amount", "amt", "value"],
    "timestamp": ["timestamp", "datetime", "event_time", "created_at", "time", "date"],
    "transaction_type": ["transaction_type", "transaction type", "type"],
    "location": ["location", "merchant_location", "transaction_location"],
    "payment_method": ["payment_method", "payment method", "method"],
    "sender_iban": ["sender_iban", "sender iban", "from_iban"],
    "recipient_iban": ["recipient_iban", "recipient iban", "to_iban"],
    "balance": ["balance", "balance_after", "balance after", "remaining_balance"],
    "description": ["description", "memo", "details", "note"],
    "account_id": ["account_id", "user_id", "customer_id", "client_id"],
    "card_id": ["card_id", "pan_id", "payment_card_id"],
    "merchant_id": ["merchant_id", "merchant", "shop_id", "store_id"],
    "device_id": ["device_id", "device", "fingerprint", "browser_id"],
    "ip": ["ip", "ip_address", "client_ip", "source_ip"],
    "country": ["country", "country_code", "txn_country"],
    "status": ["status", "result", "auth_result", "transaction_status"],
}


def normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    normalized = {c: normalize_col_name(c) for c in df.columns}
    reverse_map = {}
    for raw, norm in normalized.items():
        reverse_map.setdefault(norm, []).append(raw)

    result = {}
    for logical_name, candidates in COLUMN_CANDIDATES.items():
        found = None
        for candidate in candidates:
            candidate_norm = normalize_col_name(candidate)
            if candidate_norm in reverse_map:
                found = reverse_map[candidate_norm][0]
                break
        result[logical_name] = found

    if result["tx_id"] is None:
        for raw in df.columns:
            if "id" in normalize_col_name(raw):
                result["tx_id"] = raw
                break

    if result["sender_id"] is None and result["account_id"] is not None:
        result["sender_id"] = result["account_id"]
    if result["account_id"] is None and result["sender_id"] is not None:
        result["account_id"] = result["sender_id"]

    return result
