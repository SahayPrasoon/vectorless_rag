"""
src/api/routers/answer.py
─────────────────────────
RAG answer endpoint.

POST /companies/{slug}/documents/{document_id}/answer
     — retrieve relevant pages (0 LLM calls) then generate an answer (1 LLM call)
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from src.config.db import get_conn
from src.config.llm import llm
from src.api.models import APIResponse, AnswerRequest, AnswerOut, TreePathNodeOut
from src.pipeline.answer import answer_query, AnswerResult

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Answer"])


# ── helper ────────────────────────────────────────────────────────────────────

def _assert_company_owns_doc(slug: str, document_id: str, conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id
            FROM documents d
            JOIN companies c ON c.id = d."companyId"
            WHERE d.id = %s AND c.slug = %s
            """,
            (document_id, slug),
        )
        if cur.fetchone() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found for company '{slug}'.",
            )


# ── route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/companies/{slug}/documents/{document_id}/answer",
    response_model=APIResponse,
    summary="Ask a question about a document (RAG)",
)
def ask_question(slug: str, document_id: str, body: AnswerRequest):
    """
    Full Vectorless-RAG pipeline:

    1. Validate company owns the document.
    2. Check document status is READY (tree exists).
    3. retrieve()      — keyword-overlap tree traversal (0 LLM calls).
    4. answer_query()  — LLM generates answer from retrieved pages (1 LLM call).

    Total LLM calls per request: **1**.
    """
    with get_conn() as conn:
        _assert_company_owns_doc(slug, document_id, conn)

        # Check document is indexed and ready
        with conn.cursor() as cur:
            cur.execute(
                'SELECT status FROM documents WHERE id = %s',
                (document_id,),
            )
            row = cur.fetchone()

        doc_status = row[0] if row else None
        if doc_status == "PROCESSING":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Document '{document_id}' is still being processed. "
                    "Please wait until status is READY."
                ),
            )
        if doc_status == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Document '{document_id}' failed during processing. "
                    "Re-upload or rebuild the tree via POST .../tree/build"
                ),
            )

        # Check tree exists
        with conn.cursor() as cur:
            cur.execute(
                'SELECT 1 FROM trees WHERE "documentId" = %s',
                (document_id,),
            )
            if cur.fetchone() is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"No tree found for document '{document_id}'. "
                        "Build it first via POST .../tree/build"
                    ),
                )

        logger.info(
            "answer | company=%s | doc=%s | query=%r | top_k=%d",
            slug, document_id, body.query[:80], body.top_k,
        )

        try:
            result: AnswerResult = answer_query(
                query=body.query,
                document_id=document_id,
                conn=conn,
                llm=llm,
                top_k=body.top_k,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception(
                "answer_query failed | doc=%s | query=%r", document_id, body.query
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Answer generation failed: {exc}",
            ) from exc

    answer_out = AnswerOut(
        query=result.query,
        answer=result.answer,
        sources=result.sources,
        treePath=[TreePathNodeOut(**n) for n in result.tree_path],
        latencySeconds=result.latency_seconds,
        llmCalls=result.llm_calls,
    )
    return APIResponse(data=answer_out.model_dump())
