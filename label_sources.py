import json, re
import config
from test import load_tests

# Count term frequency rather than just presence so documents that discuss the topic more get a higher score
def _score(text, terms):
    text = text.lower()
    return sum(text.count(t.lower()) for t in terms)

def label():
    # Load every KB document into memory so we can score all of them against each test question
    files = {f.relative_to(config.KB_PATH).as_posix(): f.read_text(encoding="utf-8")
             for f in config.KB_PATH.rglob("*.md")}
    rows = []
    for t in load_tests():
        # Combine keywords and reference answer words so the scorer uses all available signal
        terms = t.keywords + re.findall(r"\w+", t.reference_answer)
        # Pick the document with the highest term overlap as the gold source for retrieval evaluation
        best = max(files, key=lambda s: _score(files[s], terms))
        row = t.model_dump(); row["source"] = best
        rows.append(row)
    # Write back to the same file so eval.py picks up the labelled sources automatically
    out = str(config.ROOT / "evaluation" / "tests.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Labelled {len(rows)} tests -> {out}")

if __name__ == "__main__":
    label()
