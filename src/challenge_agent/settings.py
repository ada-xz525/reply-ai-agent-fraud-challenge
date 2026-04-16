from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv


@dataclass
class Settings:
    project_root: Path
    openrouter_api_key: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str
    team_name: str
    model_id: str
    default_config: dict
    model_config: dict
    scoring_config: dict


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings() -> Settings:
    load_dotenv()

    project_root = Path(__file__).resolve().parents[2]

    default_config = _load_yaml(project_root / "configs" / "default.yaml")
    model_config = _load_yaml(project_root / "configs" / "models.yaml")
    scoring_config = _load_yaml(project_root / "configs" / "scoring.yaml")

    return Settings(
        project_root=project_root,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        langfuse_host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
        team_name=os.getenv("TEAM_NAME", "team"),
        model_id=os.getenv("MODEL_ID", "gpt-4o-mini"),
        default_config=default_config,
        model_config=model_config,
        scoring_config=scoring_config,
    )