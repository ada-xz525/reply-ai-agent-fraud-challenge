# reply-ai-agent-fraud-challenge

## Overview
This repository contains a Reply Mirror fraud-detection pipeline aligned with the Reply AI Agent Challenge 2026 task.

What it does:
- reads a dataset directory containing `transactions.csv` plus optional JSON side tables such as `locations.json`, `sms.json`, `mails.json`, and `users.json`
- scores transactions with behavior-change, counterparty-instability, phishing-proximity, and digital-channel signals
- writes an ASCII submission file with one suspicious `transaction_id` per line
- keeps running even when optional LLM dependencies are not installed by falling back to the deterministic path

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run
You can point the runner at either the dataset folder or directly at the `transactions.csv` file:

```bash
python run.py --dataset-dir "/path/to/The Truman Show - train" --mode training
```

Equivalent:

```bash
python run.py --dataset "/path/to/The Truman Show - train/transactions.csv" --mode training
```

The output is written under `outputs/<mode>/<dataset_name>_output.txt`.

## Validate
Basic output validation:

```bash
python validate_submission.py
```

Validation against a specific dataset:

```bash
python validate_submission.py \
  --dataset-dir "/path/to/The Truman Show - train" \
  --output "outputs/training/The Truman Show - train_output.txt"
```

## Package
Create the source archive for submission:

```bash
python zip_submission.py
```
