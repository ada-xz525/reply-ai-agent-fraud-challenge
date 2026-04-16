class BudgetGuard:
    def __init__(self, max_llm_calls_per_dataset: int = 2, max_review_records: int = 20):
        self.max_llm_calls_per_dataset = max_llm_calls_per_dataset
        self.max_review_records = max_review_records
        self.calls_used = 0

    def can_call_llm(self) -> bool:
        return self.calls_used < self.max_llm_calls_per_dataset

    def register_call(self) -> None:
        self.calls_used += 1

    def cap_review_records(self, records: list) -> list:
        return records[: self.max_review_records]