import os


class DummyChatModel:
    def __init__(self, reason: str):
        self.reason = reason

    def invoke(self, *_args, **_kwargs):
        raise RuntimeError(self.reason)


def build_chat_model(model_name: str = "gpt-4o-mini", temperature: float = 0.1, max_tokens: int = 800):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return DummyChatModel("OPENROUTER_API_KEY is not configured")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return DummyChatModel("langchain_openai is not installed")

    return ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
