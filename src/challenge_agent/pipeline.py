from math import ceil
from pathlib import Path

from .agents.orchestrator import run_orchestrator
from .agents.reviewer import run_reviewer
from .io.loader import load_challenge_data
from .io.output_writer import write_output
from .io.schema import infer_columns
from .llm.client import build_chat_model
from .observability.tracing import build_callback_handler
from .scoring.final_selector import select_final_ids
from .scoring.risk_score import build_risk_score
from .settings import load_settings
from .tools.candidate_builder import build_candidate_pool
from .tools.geo_behavior_tool import add_geo_features
from .tools.schema_profiler import profile_dataset
from .tools.text_signal_tool import build_text_risk_features
from .tools.velocity_tool import build_velocity_features


def _enforce_non_empty_not_all(final_ids: list[str], ranked_df, tx_id_col: str) -> list[str]:
    if ranked_df.empty:
        return final_ids

    if not final_ids:
        return ranked_df[tx_id_col].astype(str).head(1).tolist()

    if len(final_ids) >= len(ranked_df) and len(ranked_df) > 1:
        return ranked_df[tx_id_col].astype(str).head(len(ranked_df) - 1).tolist()

    return final_ids


def run_pipeline(dataset_dir: Path, output_path: Path, session_id: str, settings=None) -> dict:
    settings = settings or load_settings()
    data = load_challenge_data(dataset_dir)

    tx = data.transactions.copy()
    cols = infer_columns(tx)
    tx_id_col = cols["tx_id"]

    dataset_summary = profile_dataset(tx, cols)

    tx = build_velocity_features(tx, cols)
    if data.locations is not None:
        tx = add_geo_features(tx, data.locations)

    text_risk_df = build_text_risk_features(
        transactions=tx,
        cols=cols,
        users=data.users,
        conversations=data.conversations,
        messages=data.messages,
        audio_events=data.audio_events,
    )
    if not text_risk_df.empty:
        tx = tx.merge(text_risk_df, on=tx_id_col, how="left")
        tx["_text_risk"] = tx["_text_risk"].fillna(0.0)
    else:
        tx["_text_risk"] = 0.0

    for col in [
        "_geo_risk",
        "_text_risk",
        "_rapid_repeat",
        "_device_accounts_norm",
        "_ip_accounts_norm",
        "_country_switch",
        "_status_failed",
        "_is_night",
        "_behavior_change",
        "_recipient_rare",
        "_counterparty_instability",
    ]:
        if col not in tx.columns:
            tx[col] = 0.0

    tool_summary = {
        "rapid_repeat_count": int(tx["_rapid_repeat"].sum()),
        "geo_risk_nonzero": int((tx["_geo_risk"] > 0).sum()),
        "text_risk_nonzero": int((tx["_text_risk"] > 0).sum()),
        "behavior_change_nonzero": int((tx["_behavior_change"] > 0.1).sum()),
        "counterparty_instability_nonzero": int((tx["_counterparty_instability"] > 0).sum()),
        "audio_event_count": int(len(data.audio_events)) if data.audio_events is not None else 0,
    }

    model_cfg = settings.model_config.get("orchestrator", {})
    model = build_chat_model(
        model_name=model_cfg.get("model", settings.model_id),
        temperature=float(model_cfg.get("temperature", 0.1)),
        max_tokens=int(model_cfg.get("max_tokens", 900)),
    )
    callback_handler = build_callback_handler()

    strategy = run_orchestrator(
        model=model,
        callback_handler=callback_handler,
        session_id=session_id,
        dataset_summary=dataset_summary,
        tool_summary=tool_summary,
    )

    scored = build_risk_score(tx, cols, strategy.get("weights", {}))
    ranked_df, candidates, review_df = build_candidate_pool(
        scored,
        candidate_percentile=float(strategy.get("candidate_percentile", 0.90)),
        review_top_k=int(strategy.get("review_top_k_high_value", 15)),
    )

    review_records = []
    if tx_id_col and not review_df.empty:
        keep_cols = [
            tx_id_col,
            "_amount",
            "_risk_score",
            "_candidate_score",
            "_text_risk",
            "_behavior_change",
            "_counterparty_instability",
            "_recipient_rare",
            "_type_risk",
            "_method_risk",
            "_description_missing",
        ]
        keep_cols = [column for column in keep_cols if column in review_df.columns]
        review_records = review_df[keep_cols].to_dict(orient="records")

    review_result = run_reviewer(
        model=model,
        callback_handler=callback_handler,
        session_id=session_id,
        records=review_records,
    )

    final_ids = select_final_ids(
        ranked_frame=ranked_df,
        cols=cols,
        review_result=review_result,
        base_candidate_ids=candidates[tx_id_col].astype(str).tolist() if tx_id_col else [],
    )
    final_ids = _enforce_non_empty_not_all(final_ids, ranked_df, tx_id_col)

    max_allowed = min(len(ranked_df) - 1, max(1, int(len(ranked_df) * 0.15) or 1)) if len(ranked_df) > 1 else 1
    if len(final_ids) > max_allowed:
        ranked_subset = ranked_df[ranked_df[tx_id_col].astype(str).isin(set(final_ids))]
        final_ids = ranked_subset[tx_id_col].astype(str).head(max_allowed).tolist()

    write_output(final_ids, str(output_path))

    return {
        "total_transactions": len(tx),
        "predicted_count": len(final_ids),
        "candidate_count": len(candidates),
        "strategy": strategy,
        "final_ids": final_ids,
        "valid_tx_ids": set(tx[tx_id_col].astype(str)),
    }
