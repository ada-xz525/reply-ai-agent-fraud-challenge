"""
AI Agent Fraud Detection Challenge
-----------------------------------
Usage:
    python main.py --input <input_csv> --output <output_txt>

Environment variables (set in .env or shell):
    OPENROUTER_API_KEY   - API key for OpenRouter
    OPENROUTER_MODEL     - model to use (default: openai/gpt-4o-mini)
    LANGFUSE_PUBLIC_KEY  - Langfuse public key  (optional, enables tracing)
    LANGFUSE_SECRET_KEY  - Langfuse secret key  (optional, enables tracing)
    LANGFUSE_HOST        - Langfuse host        (optional, default: https://cloud.langfuse.com)
"""

import argparse
import os
import sys
import uuid

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a financial fraud analyst. "
    "You will be given a batch of transactions in CSV format. "
    "Analyze each transaction and identify which ones are potentially fraudulent. "
    "For each suspicious transaction provide: "
    "(1) the transaction ID, "
    "(2) a brief reason for suspicion, and "
    "(3) a risk level: LOW / MEDIUM / HIGH. "
    "At the end provide a short executive summary. "
    "Reply only in plain ASCII text with no markdown."
)


# ---------------------------------------------------------------------------
# Langfuse tracing (optional – gracefully disabled if keys are not set)
# ---------------------------------------------------------------------------

def _try_init_langfuse(session_id: str):
    """Initialise Langfuse callback handler if credentials are present."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    try:
        from langfuse.callback import CallbackHandler  # type: ignore
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            session_id=session_id,
        )
        return handler
    except Exception as exc:  # pragma: no cover
        print(f"[warning] Langfuse tracing could not be initialised: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_transactions(csv_path: str) -> pd.DataFrame:
    """Load transaction data from a CSV file."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    return df


# ---------------------------------------------------------------------------
# Fraud analysis
# ---------------------------------------------------------------------------

def analyse_transactions(df: pd.DataFrame, session_id: str) -> str:
    """
    Send the transactions to the LLM and return the analysis as plain text.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Please set it in your environment or .env file."
        )

    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )

    langfuse_handler = _try_init_langfuse(session_id)

    csv_text = df.to_csv(index=False)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Here are the transactions to analyze:\n\n{csv_text}"),
    ]

    invoke_kwargs: dict = {}
    if langfuse_handler:
        invoke_kwargs["config"] = {"callbacks": [langfuse_handler]}

    response = llm.invoke(messages, **invoke_kwargs)
    return str(response.content)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(text: str, output_path: str) -> None:
    """Write ASCII plain text to the output file."""
    # Ensure the text is ASCII-safe by replacing non-ASCII characters
    ascii_text = text.encode("ascii", errors="replace").decode("ascii")
    with open(output_path, "w", encoding="ascii") as fh:
        fh.write(ascii_text)
        if not ascii_text.endswith("\n"):
            fh.write("\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="AI Agent Fraud Detection - analyzes transactions from a CSV file."
    )
    parser.add_argument(
        "--input", required=True, metavar="INPUT_CSV",
        help="Path to the input CSV file containing transactions."
    )
    parser.add_argument(
        "--output", required=True, metavar="OUTPUT_TXT",
        help="Path for the output plain-text report."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    print(f"Loading transactions from: {args.input}")
    df = load_transactions(args.input)
    print(f"Loaded {len(df)} transaction(s).")

    print("Analyzing transactions with AI agent ...")
    analysis = analyse_transactions(df, session_id)

    write_output(analysis, args.output)
    print(f"Analysis written to: {args.output}")


if __name__ == "__main__":
    main()
