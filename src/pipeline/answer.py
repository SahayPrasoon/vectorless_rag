"""
src/pipeline/answer.py
──────────────────────
Answer-generation pipeline: 1 LLM call total per user query.

Public API
──────────
  answer_query(query, document_id, conn, llm, top_k) -> AnswerResult
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import psycopg
from langchain_core.language_models import BaseChatModel

from src.pipeline.retrieval import retrieve, RetrievalResult

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

ANSWER_PROMPT = """\
You are answering a question using ONLY the context pages provided below.
This is a Vectorless RAG system — there is no other source of truth.

Rules:
1. Answer using ONLY the provided context. Do not use outside knowledge.
2. If the answer is not in the context, say so explicitly — do not guess.
3. Cite the page number inline whenever you use information from a page, e.g. (Page 12).
4. Be concise: answer the question directly first, then add supporting detail.

Question:
{query}

Context:
{context}

Answer:"""


def _build_context(pages_content: dict[int, str]) -> str:
    blocks = []
    for page_number, text in sorted(pages_content.items()):
        blocks.append(f"========== Page {page_number} ==========\n{text}")
    return "\n\n".join(blocks)


def _generate_answer(
    query: str,
    pages_content: dict[int, str],
    llm: BaseChatModel,
) -> str:
    """Single LLM call — the only one in the entire query pipeline."""
    if not pages_content:
        return (
            "I couldn't find any relevant pages in the document to answer this question."
        )

    context = _build_context(pages_content)
    prompt = ANSWER_PROMPT.format(query=query, context=context)

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise


# ── Public result type ────────────────────────────────────────────────────────

@dataclass
class AnswerResult:
    query: str
    answer: str
    sources: list[int]            # page numbers used
    tree_path: list[dict]         # [{type, title, path}, ...]
    latency_seconds: float
    llm_calls: int = 1


# ── Public function ───────────────────────────────────────────────────────────

def answer_query(
    query: str,
    document_id: str,
    conn: psycopg.Connection,
    llm: BaseChatModel,
    top_k: int = 5,
) -> AnswerResult:
    """
    Full pipeline:
      retrieve()        — 0 LLM calls (keyword-overlap routing)
      _generate_answer() — 1 LLM call

    Total LLM calls per user query: 1.
    """
    start = time.perf_counter()

    retrieval: RetrievalResult = retrieve(query, document_id, conn, top_k=top_k)
    answer = _generate_answer(query, retrieval.pages_content, llm)

    elapsed = round(time.perf_counter() - start, 3)

    logger.info(
        "answer_query | doc=%s | pages=%s | latency=%.3fs",
        document_id,
        retrieval.top_pages,
        elapsed,
    )

    return AnswerResult(
        query=query,
        answer=answer,
        sources=retrieval.top_pages,
        tree_path=retrieval.tree_path,
        latency_seconds=elapsed,
        llm_calls=1,
    )