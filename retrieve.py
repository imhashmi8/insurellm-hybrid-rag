import re, functools
from openai import OpenAI
from chromadb import PersistentClient
from litellm import completion
from tenacity import retry, wait_exponential
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

import config
from config import Result

WAIT = wait_exponential(multiplier=1, min=10, max=240)
openai = OpenAI()

collection = PersistentClient(path=config.DB_NAME).get_collection(config.COLLECTION)


_data = collection.get(include=["documents", "metadatas"])
_ids = _data["ids"]
_id_to_result = {i: Result(page_content=d, metadata=m)
                 for i, d, m in zip(_ids, _data["documents"], _data["metadatas"])}
_tok = lambda t: re.findall(r"\w+", t.lower())
_bm25 = BM25Okapi([_tok(d) for d in _data["documents"]])



class RankOrder(BaseModel):
    order: list[int] = Field(description="Chunk ids most-relevant first")


@functools.lru_cache(maxsize=1024)
def _embed_query(text):
    return tuple(openai.embeddings.create(model=config.EMBED_MODEL, input=[text]).data[0].embedding)

def dense_search(query, k):
    res = collection.query(query_embeddings=[list(_embed_query(query))], n_results=k)
    return res["ids"][0]

def sparse_search(query, k):
    scores = _bm25.get_scores(_tok(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_ids[i] for i in ranked]


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


@retry(wait=WAIT)
def rewrite_query(question, history=[]):
    prompt = f"""
You are answering questions about the company Insurellm and are about to search a knowledge base.
Conversation so far:
{history}
Current question:
{question}
Respond ONLY with a short, specific search query most likely to surface the relevant content.
"""
    return completion(model=config.GEN_MODEL,
                      messages=[{"role": "system", "content": prompt}]).choices[0].message.content


@retry(wait=WAIT)
def _llm_rerank(question, chunks):
    system = ("You are a document re-ranker. Re-order the retrieved chunks most-relevant first. "
              "Reply only with the list of ranked chunk ids, including every id you were given.")
    user = f"Question:\n\n{question}\n\nRe-order all chunks by relevance.\n\nChunks:\n\n"
    for i, c in enumerate(chunks, 1):
        user += f"# CHUNK ID: {i}:\n\n{c.page_content}\n\n"
    resp = completion(model=config.GEN_MODEL,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                      response_format=RankOrder)
    order = RankOrder.model_validate_json(resp.choices[0].message.content).order

    seen, ranked = set(), []
    for i in order:
        if 1 <= i <= len(chunks) and i not in seen:
            ranked.append(chunks[i - 1]); seen.add(i)
    for idx, c in enumerate(chunks, 1):
        if idx not in seen:
            ranked.append(c)
    return ranked


@functools.lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-base")

def _cross_rerank(question, chunks):
    scores = _cross_encoder().predict([(question, c.page_content) for c in chunks])
    return [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]

def rerank(question, chunks):
    return _cross_rerank(question, chunks) if config.RERANKER == "cross" else _llm_rerank(question, chunks)


def fetch_context(question, history=[]):
    rewritten = rewrite_query(question, history)
    fused = rrf([
        dense_search(question, config.RETRIEVE_K),
        dense_search(rewritten, config.RETRIEVE_K),
        sparse_search(question, config.RETRIEVE_K),
    ])[:config.RETRIEVE_K]
    chunks = [_id_to_result[i] for i in fused]
    return rerank(question, chunks)[:config.FINAL_K]