# reply-ai-agent-fraud-challenge

AI Agent Fraud Detection Challenge submission.

## Run

```bash
pip install -r requirements.txt
python main.py --input <input_csv> --output <output_txt>
```

## Environment variables

Copy `.env.example` to `.env` and fill in your keys before running.

| Variable              | Required | Default                        | Description                          |
|-----------------------|----------|--------------------------------|--------------------------------------|
| `OPENROUTER_API_KEY`  | Yes      | –                              | API key for OpenRouter               |
| `OPENROUTER_MODEL`    | No       | `openai/gpt-4o-mini`           | Model to use via OpenRouter          |
| `LANGFUSE_PUBLIC_KEY` | No       | –                              | Enables Langfuse tracing when set    |
| `LANGFUSE_SECRET_KEY` | No       | –                              | Enables Langfuse tracing when set    |
| `LANGFUSE_HOST`       | No       | `https://cloud.langfuse.com`   | Langfuse host                        |

## Notes

- Uses OpenRouter via langchain-openai
- Uses Langfuse tracing with session ID printed at runtime
- Output is ASCII plain text