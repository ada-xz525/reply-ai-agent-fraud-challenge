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


def default_review_result() -> dict:
    return {
        "promote_tx_ids": [],
        "demote_tx_ids": [],
        "reasoning_short": "fallback review",
    }


@observe()
def run_reviewer(
    model,
    callback_handler,
    session_id: str,
    records: list[dict],
) -> dict:
    if not records:
        return default_review_result()

    system_prompt = _load_prompt(PROMPTS_DIR / "reviewer_system.txt")
    user_prompt = _load_prompt(PROMPTS_DIR / "reviewer_high_value.txt")
    user_prompt = user_prompt.replace(
        "{records_json}",
        json.dumps(records, ensure_ascii=True, indent=2),
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
        return {
            "promote_tx_ids": list(parsed.get("promote_tx_ids", [])),
            "demote_tx_ids": list(parsed.get("demote_tx_ids", [])),
            "reasoning_short": str(parsed.get("reasoning_short", "")),
        }
    except Exception:
        return default_review_result()
