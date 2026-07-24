"""
src/pipeline/answer.py
──────────────────────
Answer-generation pipeline: 1 LLM call total per query.

Public API
──────────
  answer_query(query, db_document_id, conn, llm, top_k)  -> AnswerResult
  generate_answer_from_retrieval(query, retrieval, llm)   -> AnswerResult
    ↑ used by query router for the multi-document path where retrieval
      has already been done externally.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import psycopg
from langchain_core.language_models import BaseChatModel

from src.pipeline.retrieval import retrieve, RetrievalResult

logger = logging.getLogger(__name__)

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


def _call_llm(query: str, pages_content: dict[int, str], llm: BaseChatModel) -> str:
    """Single LLM call. Returns empty-context message if no pages available."""
    if not pages_content:
        return "I couldn't find any relevant pages in the document to answer this question."

    prompt = ANSWER_PROMPT.format(query=query, context=_build_context(pages_content))
    try:
        return llm.invoke(prompt).content.strip()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise


@dataclass
class AnswerResult:
    query: str
    answer: str
    sources: list[int]
    tree_path: list[dict]
    latency_seconds: float
    llm_calls: int = 1


def answer_query(
    query: str,
    db_document_id: int,
    conn: psycopg.Connection,
    llm: BaseChatModel,
    top_k: int = 5,
) -> AnswerResult:
    """
    Full single-document pipeline:
      retrieve()   — 0 LLM calls (keyword-overlap routing)
      _call_llm()  — 1 LLM call
    """
    start = time.perf_counter()

    retrieval: RetrievalResult = retrieve(query, db_document_id, conn, top_k=top_k)
    answer = _call_llm(query, retrieval.pages_content, llm)

    elapsed = round(time.perf_counter() - start, 3)
    logger.info(
        "answer_query | doc_id=%d | pages=%s | latency=%.3fs",
        db_document_id, retrieval.top_pages, elapsed,
    )

    return AnswerResult(
        query=query,
        answer=answer,
        sources=retrieval.top_pages,
        tree_path=retrieval.tree_path,
        latency_seconds=elapsed,
        llm_calls=1,
    )


def generate_answer_from_retrieval(
    query: str,
    retrieval: RetrievalResult,
    llm: BaseChatModel,
) -> AnswerResult:
    """
    Generate an answer when retrieval has already been done externally.
    Used by the multi-document query path in query.py.
    Still exactly 1 LLM call.
    """
    start = time.perf_counter()
    answer = _call_llm(query, retrieval.pages_content, llm)
    elapsed = round(time.perf_counter() - start, 3)

    logger.info(
        "generate_answer_from_retrieval | pages=%s | latency=%.3fs",
        retrieval.top_pages, elapsed,
    )

    return AnswerResult(
        query=query,
        answer=answer,
        sources=retrieval.top_pages,
        tree_path=retrieval.tree_path,
        latency_seconds=elapsed,
        llm_calls=1,
    )