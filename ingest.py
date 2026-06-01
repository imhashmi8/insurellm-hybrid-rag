# Import necessary libraries

import json, hashlib
from openai import OpenAI
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdb import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
import config

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

