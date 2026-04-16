import re

import numpy as np
import pandas as pd


def robust_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    median = x.median()
    mad = (x - median).abs().median()

    if pd.isna(mad) or mad == 0:
        std = x.std()
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return ((x - x.mean()) / std).fillna(0.0)

    return (0.6745 * (x - median) / mad).fillna(0.0)


def minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = numeric.min(), numeric.max()
    if hi <= lo:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - lo) / (hi - lo)


def _ensure_col(df: pd.DataFrame, col: str, default=0.0) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df


def build_risk_score(df: pd.DataFrame, cols: dict, weights: dict) -> pd.DataFrame:
    out = df.copy()

    amount_col = cols.get("amount")
    if amount_col and amount_col in out.columns:
        out["_amount"] = pd.to_numeric(out[amount_col], errors="coerce").fillna(0.0)
    else:
        out["_amount"] = 0.0

    balance_col = cols.get("balance")
    if balance_col and balance_col in out.columns:
        balance_after = pd.to_numeric(out[balance_col], errors="coerce").fillna(0.0)
        pre_tx_balance = (out["_amount"] + balance_after).replace(0, np.nan)
        out["_balance_drain"] = (out["_amount"] / pre_tx_balance).fillna(0.0).clip(0.0, 1.0)
    else:
        out["_balance_drain"] = 0.0

    tx_type_col = cols.get("transaction_type")
    if tx_type_col and tx_type_col in out.columns:
        normalized_type = (
            out[tx_type_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
        )
    else:
        normalized_type = pd.Series("", index=out.index)

    type_risk_map = {
        "e_commerce": 1.0,
        "direct_debit": 0.80,
        "in_person_payment": 0.20,
        "withdrawal": 0.60,
        "transfer": 0.20,
    }
    out["_type_risk"] = normalized_type.map(type_risk_map).fillna(0.15)

    payment_method_col = cols.get("payment_method")
    if payment_method_col and payment_method_col in out.columns:
        normalized_method = (
            out[payment_method_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
        )
    else:
        normalized_method = pd.Series("", index=out.index)

    method_risk_map = {
        "paypal": 1.0,
        "smartwatch": 0.30,
        "mobile_device": 0.55,
        "googlepay": 0.65,
        "debit_card": 0.35,
    }
    out["_method_risk"] = normalized_method.map(method_risk_map).fillna(0.0)

    description_col = cols.get("description")
    if description_col and description_col in out.columns:
        desc_text = out[description_col].fillna("").astype(str).str.lower()
    else:
        desc_text = pd.Series("", index=out.index)

    out["_description_missing"] = (
        desc_text.str.strip().eq("")
        & normalized_type.isin(["e_commerce", "direct_debit"])
    ).astype(float)

    desc_patterns = [
        r"\burgent\b",
        r"\bcrypto\b",
        r"\bgift ?card\b",
        r"\bwallet\b",
        r"\brefund\b",
        r"\bverification\b",
        r"\bsecurity\b",
        r"\bprize\b",
        r"\bfee\b",
    ]
    out["_description_risk"] = desc_text.apply(
        lambda text: float(sum(bool(re.search(pattern, text)) for pattern in desc_patterns))
    )
    out["_description_risk"] = minmax_scale(out["_description_risk"])

    out["_amount_z"] = robust_zscore(out["_amount"])
    out["_amount_norm"] = minmax_scale(out["_amount_z"].clip(lower=0))
    out["_high_amount_bonus"] = minmax_scale(np.log1p(out["_amount"]))

    timestamp_col = cols.get("timestamp")
    if timestamp_col and timestamp_col in out.columns:
        timestamp = pd.to_datetime(out[timestamp_col], errors="coerce")
        out["_is_night"] = timestamp.dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(float)
    else:
        out["_is_night"] = 0.0

    for col in [
        "_rapid_repeat",
        "_device_accounts_norm",
        "_ip_accounts_norm",
        "_country_switch",
        "_status_failed",
        "_text_risk",
        "_geo_risk",
        "_behavior_change",
        "_recipient_rare",
        "_counterparty_instability",
    ]:
        out = _ensure_col(out, col, 0.0)

    digital_factor = normalized_type.isin(["e_commerce", "direct_debit", "in_person_payment"]).astype(float)
    out["_text_risk_norm"] = minmax_scale(out["_text_risk"]) * (0.2 + 0.8 * digital_factor)
    out["_geo_risk_norm"] = minmax_scale(out["_geo_risk"])

    out["_risk_score"] = (
        weights.get("amount_z", 0.35) * out["_amount_norm"]
        + weights.get("rapid_repeat", 0.60) * out["_rapid_repeat"]
        + weights.get("device_accounts", 0.80) * out["_device_accounts_norm"]
        + weights.get("ip_accounts", 0.70) * out["_ip_accounts_norm"]
        + weights.get("country_switch", 0.40) * out["_country_switch"]
        + weights.get("status_failed", 0.50) * out["_status_failed"]
        + weights.get("night", 0.35) * out["_is_night"]
        + weights.get("text_risk", 2.40) * out["_text_risk_norm"]
        + weights.get("geo_risk", 0.70) * out["_geo_risk_norm"]
        + weights.get("high_amount_bonus", 0.20) * out["_high_amount_bonus"]
        + weights.get("balance_drain", 0.60) * out["_balance_drain"]
        + weights.get("behavior_change", 2.20) * out["_behavior_change"]
        + weights.get("recipient_rare", 1.80) * out["_recipient_rare"]
        + weights.get("counterparty_instability", 2.60) * out["_counterparty_instability"]
        + weights.get("description_missing", 1.00) * out["_description_missing"]
        + weights.get("method_risk", 1.10) * out["_method_risk"]
        + weights.get("type_risk", 1.30) * out["_type_risk"]
    )

    return out
