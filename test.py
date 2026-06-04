import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "evaluation" / "tests.jsonl")

class TestQuestion(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str
    source: str = ""   # gold-relevant doc; "" -> fall back to keyword metrics

def load_tests():
    tests = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            tests.append(TestQuestion(**json.loads(line)))
    return tests
