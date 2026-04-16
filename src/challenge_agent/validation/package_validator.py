from pathlib import Path


REQUIRED_ROOT_FILES = [
    "README.md",
    "requirements.txt",
    "run.py",
]

REQUIRED_ROOT_DIRS = [
    "src",
    "configs",
    "prompts",
]

REQUIRED_PACKAGE_FILES = [
    "src/challenge_agent/pipeline.py",
    "src/challenge_agent/io/loader.py",
    "src/challenge_agent/io/schema.py",
    "src/challenge_agent/io/output_writer.py",
    "src/challenge_agent/scoring/risk_score.py",
    "src/challenge_agent/scoring/final_selector.py",
    "src/challenge_agent/agents/orchestrator.py",
    "src/challenge_agent/agents/reviewer.py",
]


def validate_project_structure(project_root: Path) -> None:
    missing = []

    for rel in REQUIRED_ROOT_FILES:
        if not (project_root / rel).exists():
            missing.append(rel)

    for rel in REQUIRED_ROOT_DIRS:
        if not (project_root / rel).exists():
            missing.append(rel)

    for rel in REQUIRED_PACKAGE_FILES:
        if not (project_root / rel).exists():
            missing.append(rel)

    if missing:
        raise FileNotFoundError(
            "Project structure incomplete. Missing:\n" + "\n".join(missing)
        )