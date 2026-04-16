import pandas as pd


def _find_column(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None

    normalized = {column.strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def add_geo_features(transactions: pd.DataFrame, locations: pd.DataFrame | None) -> pd.DataFrame:
    out = transactions.copy()
    out["__row_idx"] = range(len(out))
    out["_geo_risk"] = 0.0

    if locations is None or locations.empty:
        return out.drop(columns=["__row_idx"], errors="ignore")

    biotag_col = _find_column(locations, ["BioTag", "biotag"])
    timestamp_col = _find_column(locations, ["Datetime", "timestamp", "datetime"])
    lat_col = _find_column(locations, ["Lat", "lat"])
    lng_col = _find_column(locations, ["Lng", "lng"])
    city_col = _find_column(locations, ["city", "City"])

    sender_col = _find_column(out, ["Sender ID", "sender_id", "account_id", "user_id"])
    tx_time_col = _find_column(out, ["Timestamp", "timestamp", "Datetime", "datetime"])
    tx_location_col = _find_column(out, ["location", "Location"])

    if not biotag_col or not timestamp_col or not lat_col or not lng_col or not sender_col or not tx_time_col:
        return out.drop(columns=["__row_idx"], errors="ignore")

    loc = locations.copy()
    loc[timestamp_col] = pd.to_datetime(loc[timestamp_col], errors="coerce")

    tx = out.copy()
    tx[tx_time_col] = pd.to_datetime(tx[tx_time_col], errors="coerce")

    merged_parts = []
    for sender_id, tx_group in tx.groupby(sender_col, dropna=False):
        tx_group = tx_group.sort_values([tx_time_col, "__row_idx"]).copy()
        loc_group = loc[loc[biotag_col] == sender_id].sort_values(timestamp_col).copy()

        if loc_group.empty:
            tx_group["_geo_risk"] = 0.0
            merged_parts.append(tx_group)
            continue

        merged_group = pd.merge_asof(
            tx_group,
            loc_group,
            left_on=tx_time_col,
            right_on=timestamp_col,
            direction="nearest",
            tolerance=pd.Timedelta("6h"),
        )
        merged_group.index = tx_group.index
        merged_parts.append(merged_group)

    merged = pd.concat(merged_parts, ignore_index=False)

    if tx_location_col:
        has_claimed_location = merged[tx_location_col].fillna("").astype(str).str.strip().ne("")
    else:
        has_claimed_location = pd.Series(False, index=merged.index)

    merged["_geo_risk"] = (merged[lat_col].isna() & has_claimed_location).astype(float) * 0.25

    if tx_location_col and city_col:
        claimed_city = (
            merged[tx_location_col]
            .fillna("")
            .astype(str)
            .str.split(" - ")
            .str[0]
            .str.strip()
            .str.lower()
        )
        observed_city = merged[city_col].fillna("").astype(str).str.strip().str.lower()
        city_mismatch = (
            claimed_city.ne("")
            & observed_city.ne("")
            & claimed_city.ne(observed_city)
        )
        merged["_geo_risk"] += city_mismatch.astype(float) * 0.60

    merged = merged.sort_values("__row_idx").drop(columns=["__row_idx"], errors="ignore")
    return merged
