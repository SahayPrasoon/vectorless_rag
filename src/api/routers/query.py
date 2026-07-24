"""
src/api/routers/query.py
────────────────────────
Query / search endpoint.

POST /query/search
  JSON body:
    tenant_id   — company slug
    user_id     — logged only
    question    — natural-language question
    document_id — (optional) restrict to one logical document
    top_k       — (optional, default 5)

If document_id is omitted the system scores all tenant document trees
and answers from the highest-matching one. Still 1 LLM call total.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, status

from src.config.db import get_conn
from src.config.llm import llm
from src.api.models import APIResponse, QueryRequest, QueryOut, TreePathNodeOut
from src.pipeline.answer import answer_query, generate_answer_from_retrieval, AnswerResult
from src.pipeline.retrieval import (
    load_tree_from_db,
    traverse_tree,
    keyword_overlap,
    get_candidate_pages,
    fetch_pages_content,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_company_id(tenant_id: str, conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute('SELECT id FROM "Company" WHERE slug = %s', (tenant_id,))
        row = cur.fetchone()
    return row[0] if row else None


def _get_all_docs_for_tenant(company_id: int, conn) -> list[dict]:
    """Return [{id, sourceFile}, ...] only for documents that have a tree."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d."sourceFile"
            FROM "Document" d
            JOIN "Tree" t ON t."documentId" = d.id
            WHERE d."companyId" = %s
            ORDER BY d.id
            """,
            (company_id,),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "sourceFile": r[1]} for r in rows]


def _get_doc_by_logical_id(company_id: int, document_id: str, conn) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d."sourceFile"
            FROM "Document" d
            JOIN "Tree" t ON t."documentId" = d.id
            WHERE d."companyId" = %s AND d."sourceFile" = %s
            """,
            (company_id, document_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {"id": row[0], "sourceFile": row[1]}


# ── Multi-document retrieval ──────────────────────────────────────────────────

def _best_retrieval_across_docs(
    query: str,
    docs: list[dict],
    conn,
    top_k: int,
) -> tuple[dict, RetrievalResult]:
    """
    Run keyword-overlap tree traversal on every document.
    Returns the doc + RetrievalResult for the highest-scoring section.
    0 LLM calls.
    """
    best_doc: dict | None = None
    best_result: RetrievalResult | None = None
    best_score: float = -1.0

    for doc in docs:
        try:
            tree_root = load_tree_from_db(conn, doc["id"])
            traversal = traverse_tree(query, tree_root)
            leaf = traversal["leaf"]
            score = keyword_overlap(query, leaf)

            if score > best_score:
                best_score = score
                best_doc = doc

                candidates = get_candidate_pages(conn, leaf)
                page_numbers = [c["pageNumber"] for c in candidates]
                top_pages = page_numbers[:top_k]
                pages_content = fetch_pages_content(conn, doc["id"], top_pages)

                best_result = RetrievalResult(
                    tree_path=[
                        {"type": n.type, "title": n.title, "path": n.path}
                        for n in traversal["path"]
                    ],
                    leaf=leaf.data,
                    candidates=candidates,
                    ranked_pages=page_numbers,
                    top_pages=top_pages,
                    pages_content=pages_content,
                )
        except Exception as exc:
            logger.warning("Skipping doc id=%s during multi-doc search: %s", doc["id"], exc)
            continue

    if best_doc is None or best_result is None:
        raise ValueError("No searchable documents found for this tenant.")

    return best_doc, best_result


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/search",
    response_model=APIResponse,
    summary="Ask a question (Vectorless RAG — 1 LLM call)",
)
def search(body: QueryRequest):
    """
    Full Vectorless-RAG query:

    1. Resolve tenant → company.
    2. Resolve document(s) with built trees.
    3. Keyword-overlap tree traversal (0 LLM calls).
    4. Generate answer from retrieved pages (1 LLM call).

    Total LLM calls per request: **1**.
    """
    logger.info(
        "query/search | tenant=%s | user=%s | doc=%s | q=%r | top_k=%d",
        body.tenant_id, body.user_id,
        body.document_id or "ALL",
        body.question[:80], body.top_k,
    )

    # Safe defaults — prevents UnboundLocalError if an unexpected branch is hit
    result: AnswerResult | None = None
    matched_doc_logical: str = ""

    with get_conn() as conn:
        # ── 1. resolve tenant ─────────────────────────────────────────────────
        company_id = _get_company_id(body.tenant_id, conn)
        if company_id is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Tenant '{body.tenant_id}' not found. "
                    "Ingest at least one document first via POST /documentmcp/document/ingest."
                ),
            )

        # ── 2. resolve document(s) ────────────────────────────────────────────
        if body.document_id:
            doc = _get_doc_by_logical_id(company_id, body.document_id, conn)
            if doc is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=(
                        f"Document '{body.document_id}' not found for tenant "
                        f"'{body.tenant_id}', or its tree has not been built yet."
                    ),
                )
            docs_to_search = [doc]
        else:
            docs_to_search = _get_all_docs_for_tenant(company_id, conn)
            if not docs_to_search:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"No indexed documents found for tenant '{body.tenant_id}'. "
                        "Ingest a document first via POST /documentmcp/document/ingest."
                    ),
                )

        # ── 3 + 4. retrieval + answer generation ──────────────────────────────
        try:
            if len(docs_to_search) == 1:
                # ── single-document fast path ─────────────────────────────────
                matched_doc_logical = docs_to_search[0]["sourceFile"]
                result = answer_query(
                    query=body.question,
                    db_document_id=docs_to_search[0]["id"],
                    conn=conn,
                    llm=llm,
                    top_k=body.top_k,
                )
            else:
                # ── multi-document: score all trees, pick best ────────────────
                best_doc, retrieval = _best_retrieval_across_docs(
                    body.question, docs_to_search, conn, body.top_k,
                )
                matched_doc_logical = best_doc["sourceFile"]
                result = generate_answer_from_retrieval(
                    query=body.question,
                    retrieval=retrieval,
                    llm=llm,
                )

        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception(
                "Query failed | tenant=%s | q=%r", body.tenant_id, body.question
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {exc}",
            ) from exc

    # result is always set if no exception was raised above
    out = QueryOut(
        tenant_id=body.tenant_id,
        user_id=body.user_id,
        question=body.question,
        answer=result.answer,
        sources=result.sources,
        document_searched=matched_doc_logical,
        tree_path=[TreePathNodeOut(**n) for n in result.tree_path],
        latency_seconds=result.latency_seconds,
        llm_calls=result.llm_calls,
    )
    return APIResponse(data=out.model_dump())