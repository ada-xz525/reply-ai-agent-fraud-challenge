import pandas as pd


def _minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo = numeric.min()
    hi = numeric.max()
    if hi <= lo:
        return pd.Series(0.0, index=numeric.index)
    return (numeric - lo) / (hi - lo)


def _extract_country_code(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.extract(r"^([A-Z]{2})", expand=False)
        .fillna("")
    )


def build_velocity_features(transactions: pd.DataFrame, cols: dict, repeat_window_seconds: int = 120) -> pd.DataFrame:
    out = transactions.copy()
    out["__row_idx"] = range(len(out))

    sender_col = cols.get("sender_id") or cols.get("account_id")
    recipient_col = cols.get("recipient_id")
    timestamp_col = cols.get("timestamp")
    tx_type_col = cols.get("transaction_type")
    sender_iban_col = cols.get("sender_iban")
    recipient_iban_col = cols.get("recipient_iban")
    description_col = cols.get("description")

    out["_rapid_repeat"] = 0.0
    out["_device_accounts_norm"] = 0.0
    out["_ip_accounts_norm"] = 0.0
    out["_country_switch"] = 0.0
    out["_status_failed"] = 0.0
    out["_is_night"] = 0.0
    out["_behavior_change"] = 0.0
    out["_recipient_rare"] = 0.0
    out["_counterparty_instability"] = 0.0

    if timestamp_col and timestamp_col in out.columns:
        out["_event_ts"] = pd.to_datetime(out[timestamp_col], errors="coerce")
        out["_is_night"] = out["_event_ts"].dt.hour.isin([0, 1, 2, 3, 4, 5, 23]).astype(float)
    else:
        out["_event_ts"] = pd.NaT

    if sender_col and sender_col in out.columns and recipient_col and recipient_col in out.columns:
        sender_fanout = out.groupby(sender_col)[recipient_col].transform("nunique")
        recipient_fanin = out.groupby(recipient_col)[sender_col].transform("nunique")
        recipient_global_count = out.groupby(recipient_col)[recipient_col].transform("count")

        out["_device_accounts_norm"] = _minmax_scale(sender_fanout)
        out["_ip_accounts_norm"] = _minmax_scale(recipient_fanin)
        out["_recipient_rare"] = 1.0 - _minmax_scale(recipient_global_count)

        pair_count = out.groupby([sender_col, recipient_col])[recipient_col].transform("count")
        pair_novelty = 1.0 - _minmax_scale(pair_count)

        if tx_type_col and tx_type_col in out.columns:
            ordered = out.sort_values([sender_col, "_event_ts", "__row_idx"]).copy()
            ordered["_new_type_for_sender"] = ordered.groupby(sender_col)[tx_type_col].transform(
                lambda series: (~series.duplicated()).astype(float)
            )
            out.loc[ordered.index, "_new_type_for_sender"] = ordered["_new_type_for_sender"]
        else:
            out["_new_type_for_sender"] = 0.0

        ordered = out.sort_values([sender_col, "_event_ts", "__row_idx"]).copy()
        ordered["_new_counterparty"] = ordered.groupby(sender_col)[recipient_col].transform(
            lambda series: (~series.duplicated()).astype(float)
        )
        out.loc[ordered.index, "_new_counterparty"] = ordered["_new_counterparty"]

        out["_behavior_change"] = (
            0.20 * out["_new_counterparty"]
            + 0.15 * out["_new_type_for_sender"]
            + 0.35 * pair_novelty
            + 0.30 * out["_recipient_rare"]
        )

        if recipient_iban_col and recipient_iban_col in out.columns:
            recipient_iban_variants = out.groupby(recipient_col)[recipient_iban_col].transform("nunique")
            out["_counterparty_instability"] = _minmax_scale(recipient_iban_variants)

    if sender_col and sender_col in out.columns and timestamp_col and timestamp_col in out.columns:
        ordered = out.sort_values([sender_col, "_event_ts", "__row_idx"]).copy()
        time_delta = ordered.groupby(sender_col)["_event_ts"].diff().dt.total_seconds()
        rapid_repeat = time_delta.le(repeat_window_seconds).fillna(False)
        if recipient_col and recipient_col in ordered.columns:
            same_recipient = ordered.groupby(sender_col)[recipient_col].transform(
                lambda series: series.eq(series.shift()).fillna(False)
            )
            rapid_repeat &= same_recipient
        out.loc[ordered.index, "_rapid_repeat"] = rapid_repeat.astype(float)

    sender_country = None
    recipient_country = None
    if sender_iban_col and sender_iban_col in out.columns:
        sender_country = _extract_country_code(out[sender_iban_col])
    elif sender_col and sender_col in out.columns:
        sender_country = _extract_country_code(out[sender_col])

    if recipient_iban_col and recipient_iban_col in out.columns:
        recipient_country = _extract_country_code(out[recipient_iban_col])
    elif recipient_col and recipient_col in out.columns:
        recipient_country = _extract_country_code(out[recipient_col])

    if sender_country is not None and recipient_country is not None:
        out["_country_switch"] = (
            sender_country.ne("")
            & recipient_country.ne("")
            & sender_country.ne(recipient_country)
        ).astype(float)

    if description_col and description_col in out.columns:
        suspicious_status = out[description_col].fillna("").astype(str).str.lower().str.contains(
            r"\b(?:fail|failed|declined|retry|reversed|rejected|chargeback)\b",
            regex=True,
        )
        out["_status_failed"] = suspicious_status.astype(float)

    return out.drop(columns=["__row_idx"], errors="ignore")
