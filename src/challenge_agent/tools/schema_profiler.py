import pandas as pd


def profile_dataset(df: pd.DataFrame, cols: dict) -> dict:
    summary = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
        "inferred_columns": cols,
        "missing_ratio": {
            str(col): float(df[col].isna().mean()) for col in df.columns
        },
    }

    amount_col = cols.get("amount")
    if amount_col and amount_col in df.columns:
        amount = pd.to_numeric(df[amount_col], errors="coerce")
        summary["amount_stats"] = {
            "min": float(amount.min()) if len(amount) else 0.0,
            "median": float(amount.median()) if len(amount) else 0.0,
            "p95": float(amount.quantile(0.95)) if len(amount) else 0.0,
            "p99": float(amount.quantile(0.99)) if len(amount) else 0.0,
            "max": float(amount.max()) if len(amount) else 0.0,
        }

    timestamp_col = cols.get("timestamp")
    if timestamp_col and timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
        summary["time_min"] = str(ts.min()) if ts.notna().any() else None
        summary["time_max"] = str(ts.max()) if ts.notna().any() else None

    return summary