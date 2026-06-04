# Import necessary libraries

import json, hashlib
from openai import OpenAI
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
import config

WAIT = wait_exponential(multiplier=1, min=10, max=240)
WORKERS = 4
openai = OpenAI()

# Chunk model definition
class Chunk(BaseModel):
    headline: str = Field(description="A brief heading likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing this chunk")
    original_text: str = Field(description="The original text of this chunk, exactly as-is")

    def as_result(self, document):
        return {
            "page_content": f"{self.headline}\n\n{self.summary}\n\n{self.original_text}",
            "metadata": {"source": document["source"], "type": document["type"]},
        }

# Chunks model definition
class Chunks(BaseModel):
    chunks: list[Chunk]


# Function to fetch documents from the knowledge base
def fetch_documents():
    documents = []

    for folder in config.KB_PATH.iterdir():
        if not folder.is_dir():
            continue
        for file in folder.rglob("*.md"):
            documents.append({
                "type": folder.name,
                "source": file.relative_to(config.KB_PATH).as_posix(),
                "text": file.read_text(encoding="utf-8"),
            })
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
# Chunking instruction template
def _make_prompt(document):
    how_many = (len(document["text"]) // config.AVERAGE_CHUNK_SIZE) + 1
    return f"""
You split a document into overlapping chunks for a knowledge base.
The document is from the shared drive of a company called Insurellm.
Type: {document["type"]}  Source: {document["source"]}
A chatbot will use these chunks to answer questions about Insurellm.
Split the whole document - leave nothing out. Aim for about {how_many} chunks (use judgement),
with ~25% overlap so the same text appears in adjacent chunks for best retrieval.
For each chunk give a headline, a summary, and the original text.

Document:
{document["text"]}

Respond with the chunks.
""" 

# Process a single document to create chunks, with caching and retry logic
@retry(wait=WAIT)
def process_document(document):
    key = hashlib.sha1((document["source"] + document["text"]).encode()).hexdigest()
    cached = config.CHUNK_CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    resp = completion(model=config.CHUNK_MODEL,
                      messages=[{"role": "user", "content": _make_prompt(document)}],
                      response_format=Chunks)
    parsed = Chunks.model_validate_json(resp.choices[0].message.content).chunks
    results = [c.as_result(document) for c in parsed]
    cached.write_text(json.dumps(results))
    return results

# Create chunks for all documents using multiprocessing
def create_chunks(documents):
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(res)
    return chunks



# Create embeddings for the chunks and store them in a vector database
def create_embeddings(chunks):
    chroma = PersistentClient(path=config.DB_NAME)
    if config.COLLECTION in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(config.COLLECTION)
    collection = chroma.get_or_create_collection(config.COLLECTION)

    texts = [c["page_content"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    for i in range(0, len(texts), config.EMBED_BATCH):
        sl = slice(i, i + config.EMBED_BATCH)
        vectors = [e.embedding for e in
                   openai.embeddings.create(model=config.EMBED_MODEL, input=texts[sl]).data]
        collection.add(ids=ids[sl], embeddings=vectors, documents=texts[sl], metadatas=metas[sl])
    print(f"Vectorstore created with {collection.count()} chunks")

if __name__ == "__main__":
    docs = fetch_documents()
    create_embeddings(create_chunks(docs))
    print("Ingestion complete")