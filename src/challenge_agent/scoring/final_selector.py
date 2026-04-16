def select_final_ids(ranked_frame, cols, review_result, base_candidate_ids=None):
    tx_id_col = cols["tx_id"]
    if tx_id_col is None:
        raise ValueError("Could not infer transaction ID column")

    if base_candidate_ids is None:
        selected_ids = set(ranked_frame[tx_id_col].astype(str).tolist())
    else:
        selected_ids = {str(value) for value in base_candidate_ids}

    promote = set(review_result.get("promote_tx_ids", []))

    selected_ids |= promote

    ranked = ranked_frame.copy()
    ranked[tx_id_col] = ranked[tx_id_col].astype(str)
    ranked = ranked[ranked[tx_id_col].isin(selected_ids)]

    if "_candidate_score" in ranked.columns:
        ranked = ranked.sort_values("_candidate_score", ascending=False)

    return ranked[tx_id_col].tolist()
