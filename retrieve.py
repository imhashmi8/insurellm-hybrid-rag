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


