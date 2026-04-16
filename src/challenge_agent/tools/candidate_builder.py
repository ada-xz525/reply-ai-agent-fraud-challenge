from math import ceil

import pandas as pd


def minmax_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = numeric.min(), numeric.max()
    if hi <= lo:
        return pd.Series(0.0, index=numeric.index)
    return (numeric - lo) / (hi - lo)


def build_candidate_pool(
    df: pd.DataFrame,
    candidate_percentile: float = 0.90,
    review_top_k: int = 15,
):
    out = df.copy()

    for col in ["_risk_score", "_amount", "_text_risk", "_geo_risk"]:
        if col not in out.columns:
            out[col] = 0.0

    out["_economic_priority"] = minmax_scale(out["_amount"])
    out["_text_norm"] = minmax_scale(out["_text_risk"])
    out["_geo_norm"] = minmax_scale(out["_geo_risk"])

    out["_candidate_score"] = (
        0.82 * out["_risk_score"]
        + 0.08 * out["_economic_priority"]
        + 0.07 * out["_text_norm"]
        + 0.03 * out["_geo_norm"]
    )

    out = out.sort_values(["_candidate_score", "_amount"], ascending=False)

    # Keep the LLM advisory bounded so it cannot collapse recall on larger datasets.
    candidate_percentile = min(max(candidate_percentile, 0.70), 0.90)
    min_candidates = max(1, ceil(len(out) * 0.03))
    max_candidates = min(len(out), max(min_candidates, int(len(out) * 0.15) or 1))

    cut = out["_candidate_score"].quantile(candidate_percentile)
    candidates = out[out["_candidate_score"] >= cut].copy()
    if len(candidates) < min_candidates:
        candidates = out.head(min_candidates).copy()
    if len(candidates) > max_candidates:
        candidates = out.head(max_candidates).copy()

    if candidates.empty:
        return out, candidates, candidates

    lower_border = candidates["_candidate_score"].quantile(0.40)
    review_pool = out[out["_candidate_score"] >= lower_border].copy()
    review_pool = review_pool.head(max(len(candidates), review_top_k * 4))
    review_df = review_pool.sort_values(["_amount", "_candidate_score"], ascending=False).head(review_top_k)

    return out, candidates, review_df
