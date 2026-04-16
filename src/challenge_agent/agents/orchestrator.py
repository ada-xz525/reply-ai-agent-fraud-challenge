import json
from pathlib import Path

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    HumanMessage = None

from ..llm.json_parser import extract_json_object
from ..observability.tracing import observe


PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


def _load_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def default_strategy() -> dict:
    return {
        "focus_patterns": [
            "behavior_change",
            "recipient_id_to_iban_instability",
            "phishing_proximity",
            "digital_channel_spend",
            "rare_counterparties",
        ],
        "weights": {
            "amount_z": 0.35,
            "rapid_repeat": 0.60,
            "device_accounts": 0.80,
            "ip_accounts": 0.70,
            "country_switch": 0.40,
            "status_failed": 0.50,
            "night": 0.35,
            "text_risk": 2.40,
            "geo_risk": 0.70,
            "high_amount_bonus": 0.20,
            "balance_drain": 0.60,
            "behavior_change": 2.20,
            "recipient_rare": 1.80,
            "counterparty_instability": 2.60,
            "description_missing": 1.00,
            "method_risk": 1.10,
            "type_risk": 1.30,
        },
        "candidate_percentile": 0.90,
        "risk_threshold_quantile": 0.85,
        "review_top_k_high_value": 15,
        "reasoning_short": "behavior-first strategy",
    }


@observe()
def run_orchestrator(
    model,
    callback_handler,
    session_id: str,
    dataset_summary: dict,
    tool_summary: dict,
) -> dict:
    system_prompt = _load_prompt(PROMPTS_DIR / "orchestrator_system.txt")
    user_prompt = _load_prompt(PROMPTS_DIR / "orchestrator_strategy.txt")
    user_prompt = user_prompt.replace(
        "{summary_json}",
        json.dumps(dataset_summary, ensure_ascii=True, indent=2),
    )
    user_prompt = user_prompt.replace(
        "{tool_summary_json}",
        json.dumps(tool_summary, ensure_ascii=True, indent=2),
    )
    prompt = system_prompt + "\n\n" + user_prompt

    try:
        if HumanMessage is None:
            raise RuntimeError("langchain_core is not installed")

        response = model.invoke(
            [HumanMessage(content=prompt)],
            config={
                "callbacks": [callback_handler],
                "metadata": {"langfuse_session_id": session_id},
            },
        )
        parsed = extract_json_object(response.content)

        merged = default_strategy()
        merged.update({key: value for key, value in parsed.items() if key != "weights"})
        if isinstance(parsed.get("weights"), dict):
            merged["weights"].update(parsed["weights"])
        return merged
    except Exception:
        return default_strategy()
