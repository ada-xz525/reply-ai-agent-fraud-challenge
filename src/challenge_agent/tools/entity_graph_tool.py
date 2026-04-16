import pandas as pd


def add_behavioral_features(tx: pd.DataFrame, cols: dict) -> pd.DataFrame:
    out = tx.copy()

    sender_col = cols["sender_id"]
    recipient_col = cols["recipient_id"]
    method_col = cols["payment_method"]
    type_col = cols["transaction_type"]
    time_col = cols["timestamp"]

    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    out = out.sort_values([sender_col, time_col]).copy()

    # 每个 sender 的交易数
    out["_sender_tx_count"] = out.groupby(sender_col)[sender_col].transform("count")

    # sender 对应多少个不同 recipient
    out["_sender_unique_recipients"] = out.groupby(sender_col)[recipient_col].transform("nunique")

    # sender 对应多少种支付方式 / 交易类型
    out["_sender_unique_methods"] = out.groupby(sender_col)[method_col].transform("nunique")
    out["_sender_unique_types"] = out.groupby(sender_col)[type_col].transform("nunique")

    # 是否新收款方
    first_seen_recipient_rank = out.groupby(sender_col)[recipient_col].transform(
        lambda s: pd.factorize(s)[0]
    )
    out["_new_recipient_flag"] = (first_seen_recipient_rank == 0).astype(float)

    # 是否支付方式切换
    prev_method = out.groupby(sender_col)[method_col].shift(1)
    out["_payment_method_switch"] = (
        prev_method.notna() & (prev_method != out[method_col])
    ).astype(float)

    # 是否交易类型切换
    prev_type = out.groupby(sender_col)[type_col].shift(1)
    out["_transaction_type_switch"] = (
        prev_type.notna() & (prev_type != out[type_col])
    ).astype(float)

    # 时间间隔
    out["_delta_seconds"] = out.groupby(sender_col)[time_col].diff().dt.total_seconds()
    out["_rapid_repeat"] = out["_delta_seconds"].between(0, 3600, inclusive="both").fillna(False).astype(float)

    return out