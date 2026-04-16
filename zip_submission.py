from pathlib import Path
import zipfile


INCLUDE_PATHS = [
    "README.md",
    "requirements.txt",
    "run.py",
    "validate_submission.py",
    "zip_submission.py",
    "configs",
    "prompts",
    "src",
    "tests",
]

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".csv",
}

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".git",
    ".venv",
    "venv",
    ".mypy_cache",
}

EXCLUDED_FILE_NAMES = {
    ".DS_Store",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".zip",
}


def _should_include(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return False

    if path.name in EXCLUDED_FILE_NAMES:
        return False

    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False

    if path.is_dir():
        return False

    if path.suffix:
        return path.suffix.lower() in TEXT_SUFFIXES

    return path.name in {"LICENSE", "Makefile"}


def _iter_files(root: Path, rel: str):
    path = root / rel
    if path.is_file():
        if _should_include(path):
            yield path
        return

    if path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and _should_include(file_path):
                yield file_path


def _validate_zip_is_utf8(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            payload = zf.read(name)
            payload.decode("utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    out_zip = root / "submission_source.zip"

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        seen = set()
        for rel in INCLUDE_PATHS:
            for file_path in _iter_files(root, rel):
                arcname = str(file_path.relative_to(root))
                if arcname in seen:
                    continue
                seen.add(arcname)
                zf.write(file_path, arcname=arcname)

    _validate_zip_is_utf8(out_zip)
    print(f"[OK] created {out_zip}")


if __name__ == "__main__":
    main()
