import re
import unicodedata

import numpy as np
import pandas as pd


RISK_PATTERNS = [
    (r"paypa1|amaz0n|micr0soft|ub3r|netfl1x|r1d3share|citydriv3", 2.0),
    (r"bit\.ly|tinyurl|shorturl", 1.5),
    (r"http://", 1.0),
    (r"\burgent\b|\bimmediate action\b|\bverify\b|\baccount lock\b|\bsuspicious\b", 1.0),
    (r"\bsecurity\b|\blogin\b|\bsuspension\b|\brestore access\b", 0.8),
]


def _find_column(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None

    normalized = {
        re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_"): column
        for column in df.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "_", candidate.strip().lower()).strip("_")
        if key in normalized:
            return normalized[key]
    return None


def _score_text(text: str) -> float:
    if not isinstance(text, str):
        return 0.0

    lowered = text.lower()
    return float(sum(weight for pattern, weight in RISK_PATTERNS if re.search(pattern, lowered)))


def _normalize_text(text) -> str:
    if not isinstance(text, str):
        return ""

    normalized = unicodedata.normalize("NFKD", text)
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    lowered = without_marks.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _extract_timestamp(text: str):
    if not isinstance(text, str):
        return pd.NaT

    match = re.search(r"Date:\s*([^\n]+)", text)
    if not match:
        return pd.NaT

    timestamp = pd.to_datetime(match.group(1).strip(), errors="coerce", utc=True)
    if pd.isna(timestamp):
        timestamp = pd.to_datetime(match.group(1).strip(), errors="coerce")

    if pd.isna(timestamp):
        return pd.NaT

    if getattr(timestamp, "tzinfo", None) is not None:
        return timestamp.tz_convert(None)

    return timestamp


def _build_user_sender_map(transactions: pd.DataFrame, cols: dict, users: pd.DataFrame | None) -> dict[str, str]:
    if users is None:
        return {}

    sender_id_col = cols.get("sender_id") or cols.get("account_id")
    sender_iban_col = cols.get("sender_iban")
    if not sender_id_col or not sender_iban_col:
        return {}

    if sender_id_col not in transactions.columns or sender_iban_col not in transactions.columns:
        return {}

    iban_to_sender = (
        transactions[[sender_id_col, sender_iban_col]]
        .dropna()
        .drop_duplicates(subset=[sender_iban_col])
        .set_index(sender_iban_col)[sender_id_col]
        .to_dict()
    )

    first_name_col = _find_column(users, ["first_name", "first name"])
    last_name_col = _find_column(users, ["last_name", "last name"])
    iban_col = _find_column(users, ["iban"])
    if not first_name_col or not last_name_col or not iban_col:
        return {}

    mapping = {}
    for _, row in users.iterrows():
        sender_id = iban_to_sender.get(row[iban_col])
        if not sender_id:
            continue
        first_name = _normalize_text(str(row[first_name_col]))
        last_name = _normalize_text(str(row[last_name_col]))
        full_name = _normalize_text(f"{row[first_name_col]} {row[last_name_col]}")

        for token in [first_name, last_name, full_name]:
            if token:
                mapping[token] = sender_id

    return mapping


def _collect_suspicious_events(
    df: pd.DataFrame | None,
    text_column: str | None,
    user_lookup: dict[str, str],
) -> pd.DataFrame:
    if df is None or text_column is None or not user_lookup:
        return pd.DataFrame(columns=["sender_id", "event_ts", "event_score"])

    rows = []
    for _, row in df.iterrows():
        text = row[text_column]
        score = _score_text(text)
        if score <= 0:
            continue

        lowered = _normalize_text(str(text))
        sender_id = None
        for token, mapped_sender in user_lookup.items():
            if token and token in lowered:
                sender_id = mapped_sender
                break

        if not sender_id:
            continue

        event_ts = _extract_timestamp(text)
        if pd.isna(event_ts):
            continue

        rows.append(
            {
                "sender_id": sender_id,
                "event_ts": event_ts,
                "event_score": score,
            }
        )

    return pd.DataFrame(rows)


def _collect_audio_events(
    df: pd.DataFrame | None,
    user_lookup: dict[str, str],
) -> pd.DataFrame:
    if df is None or df.empty or not user_lookup:
        return pd.DataFrame(columns=["sender_id", "event_ts", "event_score"])

    rows = []
    for _, row in df.iterrows():
        contact_name = _normalize_text(str(row.get("contact_name", "")))
        sender_id = user_lookup.get(contact_name)
        event_ts = pd.to_datetime(row.get("event_ts"), errors="coerce")
        if not sender_id or pd.isna(event_ts):
            continue

        hour = int(event_ts.hour)
        is_odd_hour = hour <= 6 or hour >= 22
        rows.append(
            {
                "sender_id": sender_id,
                "event_ts": event_ts,
                "event_score": 0.30 + (0.35 if is_odd_hour else 0.0),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["sender_id", "event_ts", "event_score"])

    events = pd.DataFrame(rows).sort_values(["sender_id", "event_ts"]).reset_index(drop=True)
    repeated_contact = (
        events.groupby("sender_id")["event_ts"].diff().dt.total_seconds().div(86400.0).le(21).fillna(False)
    )
    events["event_score"] = events["event_score"] + repeated_contact.astype(float) * 0.35
    return events


def build_text_risk_features(
    transactions: pd.DataFrame,
    cols: dict,
    users: pd.DataFrame | None,
    conversations: pd.DataFrame | None,
    messages: pd.DataFrame | None,
    audio_events: pd.DataFrame | None = None,
    window_hours: int = 72,
) -> pd.DataFrame:
    tx_id_col = cols.get("tx_id")
    sender_id_col = cols.get("sender_id") or cols.get("account_id")
    timestamp_col = cols.get("timestamp")
    if not tx_id_col or not sender_id_col or not timestamp_col:
        return pd.DataFrame(columns=[tx_id_col or "transaction_id", "_text_risk"])

    if tx_id_col not in transactions.columns or sender_id_col not in transactions.columns or timestamp_col not in transactions.columns:
        return pd.DataFrame(columns=[tx_id_col, "_text_risk"])

    user_lookup = _build_user_sender_map(transactions, cols, users)
    sms_col = _find_column(conversations, ["sms", "message"])
    mail_col = _find_column(messages, ["mail", "email", "message"])

    events = pd.concat(
        [
            _collect_suspicious_events(conversations, sms_col, user_lookup),
            _collect_suspicious_events(messages, mail_col, user_lookup),
            _collect_audio_events(audio_events, user_lookup),
        ],
        ignore_index=True,
    )
    if events.empty:
        return pd.DataFrame(columns=[tx_id_col, "_text_risk"])

    tx = transactions[[tx_id_col, sender_id_col, timestamp_col]].copy()
    tx[timestamp_col] = pd.to_datetime(tx[timestamp_col], errors="coerce")
    tx["__row_idx"] = range(len(tx))
    tx["_text_risk"] = 0.0

    for sender_id, sender_events in events.groupby("sender_id"):
        sender_tx = tx[tx[sender_id_col] == sender_id].copy()
        if sender_tx.empty:
            continue

        for _, event in sender_events.iterrows():
            delta_hours = (
                sender_tx[timestamp_col] - event["event_ts"]
            ).dt.total_seconds() / 3600.0
            mask = delta_hours.between(0, window_hours, inclusive="both")
            if not mask.any():
                continue

            decay = 1.0 - (delta_hours[mask] / window_hours)
            candidate_scores = event["event_score"] * decay.clip(lower=0.0)
            target_rows = sender_tx.loc[mask, "__row_idx"].to_numpy()
            current_scores = tx.loc[tx["__row_idx"].isin(target_rows), "_text_risk"].to_numpy()
            tx.loc[tx["__row_idx"].isin(target_rows), "_text_risk"] = np.maximum(
                current_scores,
                candidate_scores.to_numpy(),
            )

    return tx[[tx_id_col, "_text_risk"]]
