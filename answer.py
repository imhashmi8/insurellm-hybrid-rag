from litellm import completion
from tenacity import retry, wait_exponential
import config
from retrieve import fetch_context

WAIT = wait_exponential(multiplier=1, min=10, max=240)
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
Your answer is evaluated for accuracy, relevance and completeness, so only answer the question
and fully answer it. If the context does not contain the answer, say you don't know.
Ground every claim in the extracts below and do not invent facts.

Relevant extracts from the Knowledge Base:
{context}
"""


def _messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {c.metadata['source']}:\n{c.page_content}" for c in chunks
    )
    return ([{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
            + history + [{"role": "user", "content": question}])




@retry(wait=WAIT)
def answer_question(question, history=[]):
    chunks = fetch_context(question, history)
    resp = completion(model=config.GEN_MODEL, messages=_messages(question, history, chunks))
    return resp.choices[0].message.content, chunks