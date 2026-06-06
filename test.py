import json
from pathlib import Path
from pydantic import BaseModel, Field

# Resolve path relative to this file so the project works from any working directory
TEST_FILE = str(Path(__file__).parent / "evaluation" / "tests.jsonl")

# Pydantic model so each test question is validated and typed on load rather than failing silently at eval time
class TestQuestion(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str
    source: str = ""   # gold-relevant doc; "" -> fall back to keyword metrics

# Parse JSONL line-by-line so the file can be appended to without rewriting the whole thing
def load_tests():
    tests = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            tests.append(TestQuestion(**json.loads(line)))
    return tests
