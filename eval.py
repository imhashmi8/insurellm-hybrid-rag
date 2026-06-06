import math, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from litellm import completion

import config
from test import TestQuestion, load_tests
from answer import answer_question, fetch_context

# Structured output so the judge returns machine-readable scores rather than prose
class RetrievalEval(BaseModel):
    mrr: float; ndcg: float; hit_rate: float; keyword_coverage: float

# Structured output so the judge scores are directly usable without parsing free text
class AnswerEval(BaseModel):
    feedback: str = Field(description="Concise feedback vs the reference answer")
    accuracy: float = Field(description="1 (wrong) to 5 (perfect). Any wrong answer scores 1.")
    completeness: float = Field(description="1 to 5. Only 5 if ALL reference info is present.")
    relevance: float = Field(description="1 to 5. Only 5 if on-topic with no extra info.")

# DCG helper used by nDCG, penalises relevant results that appear lower in the ranking
def _dcg(rels): return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


# Measure how well the retrieval pipeline surfaces the gold-labelled source document
def evaluate_retrieval(test: TestQuestion, k=config.FINAL_K) -> RetrievalEval:
    docs = fetch_context(test.question)[:k]

    # Keyword coverage catches cases where the right document was retrieved but gold source labelling is imperfect
    blob = " ".join(d.page_content.lower() for d in docs)
    coverage = (sum(1 for kw in test.keywords if kw.lower() in blob) / len(test.keywords) * 100
                if test.keywords else 0.0)

    if test.source:                                   # source-based (true relevance)
        rels = [1 if d.metadata.get("source") == test.source else 0 for d in docs]
        mrr = next((1 / r for r, v in enumerate(rels, 1) if v), 0.0)
        idcg = _dcg(sorted(rels, reverse=True))
        ndcg = _dcg(rels) / idcg if idcg else 0.0
        hit = 1.0 if any(rels) else 0.0
    else:                                             # fallback: keyword-based
        mrr = ndcg = hit = 0.0
        for kw in test.keywords:
            for r, d in enumerate(docs, 1):
                if kw.lower() in d.page_content.lower():
                    mrr += 1 / r; hit += 1; ndcg += 1 / math.log2(r + 1); break
        n = max(len(test.keywords), 1)
        mrr, ndcg, hit = mrr / n, ndcg / n, hit / n

    return RetrievalEval(mrr=mrr, ndcg=ndcg, hit_rate=hit, keyword_coverage=coverage)


# Use an LLM as judge to score answer quality against a reference, avoids needing exact string matches
def evaluate_answer(test: TestQuestion):
    start = time.perf_counter()
    answer, docs = answer_question(test.question)
    latency = time.perf_counter() - start
    judge = [
        {"role": "system", "content": "You are an expert evaluator. Compare to the reference. Only 5/5 for perfect answers."},
        {"role": "user", "content": f"""Question:
{test.question}

Generated Answer:
{answer}

Reference Answer:
{test.reference_answer}

Score accuracy, completeness, relevance 1 (very poor) to 5 (ideal). If wrong, accuracy must be 1."""},
    ]
    resp = completion(model=config.JUDGE_MODEL, messages=judge, response_format=AnswerEval)
    return AnswerEval.model_validate_json(resp.choices[0].message.content), latency


# Run all eval functions in parallel to avoid waiting on each LLM call sequentially
def _aggregate(tests, fn, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(zip(tests, ex.map(fn, tests)))


def run(workers=8):
    tests = load_tests()
    print(f"Evaluating {len(tests)} tests with the PRO pipeline...\n")

    r = _aggregate(tests, evaluate_retrieval, workers)
    # Break MRR down by category to spot which document types the retriever struggles with
    by_cat = defaultdict(list)
    for t, res in r:
        by_cat[t.category].append(res.mrr)
    avg = lambda key: sum(getattr(x, key) for _, x in r) / len(r)
    print("RETRIEVAL")
    print(f"  MRR        {avg('mrr'):.4f}")
    print(f"  nDCG       {avg('ndcg'):.4f}")
    print(f"  Hit@{config.FINAL_K}     {avg('hit_rate'):.4f}")
    print(f"  Coverage   {avg('keyword_coverage'):.1f}%")
    for c, v in by_cat.items():
        print(f"    {c:<14} MRR {sum(v)/len(v):.3f}")

    a = _aggregate(tests, evaluate_answer, workers)
    n = len(a)
    acc = sum(e.accuracy for _, (e, _) in a) / n
    comp = sum(e.completeness for _, (e, _) in a) / n
    rel = sum(e.relevance for _, (e, _) in a) / n
    lat = sum(l for _, (_, l) in a) / n
    print("\nANSWER QUALITY")
    print(f"  Accuracy     {acc:.2f}/5")
    print(f"  Completeness {comp:.2f}/5")
    print(f"  Relevance    {rel:.2f}/5")
    print(f"  Avg latency  {lat:.2f}s")

if __name__ == "__main__":
    run()
