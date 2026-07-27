"""
src/api/routers/ingest.py
─────────────────────────
Document ingestion endpoint.

POST /document/ingest
  Form fields:
    tenant_id   — maps to Company.slug (auto-created if missing)
    user_id     — logged only, not stored
    document_id — caller-supplied logical name for the document
                  (used as sourceFile key; appended if already exists)
    file        — the PDF binary

Behaviour
─────────
  • If the tenant (Company) does not exist → create it automatically.
  • If a Document with this (tenant_id, document_id) exists → append pages
    and rebuild the tree.
  • If it is a brand-new document → create Document + pages then build tree.
"""
from __future__ import annotations

import io
import logging

import pdfplumber
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.config.db import get_conn
from src.config.llm import llm
from src.api.models import APIResponse, IngestOut
from src.pipeline.page_metadata import generate_and_store_metadata
from src.pipeline.tree_builder import build_tree, upsert_tree

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document", tags=["Ingest"])


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_or_create_company(tenant_id: str, conn) -> str:
    """
    Return the Company.id (cuid string) for tenant_id.
    Creates the company row automatically if it does not exist.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM companies WHERE slug = %s",
            (tenant_id,),
        )
        row = cur.fetchone()

    if row:
        return row[0]

    # Auto-create: name = slug = tenant_id
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO companies (id, name, slug, "createdAt", "updatedAt")
            VALUES (gen_random_uuid()::text, %s, %s, NOW(), NOW())
            RETURNING id
            """,
            (tenant_id, tenant_id),
        )
        new_id = cur.fetchone()[0]
    conn.commit()

    logger.info("Auto-created company: slug=%s id=%s", tenant_id, new_id)
    return new_id


def _get_document(company_id: str, document_id: str, conn) -> dict | None:
    """
    Return the Document row as a dict if (company_id, sourceFile=document_id) exists.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, "sourceFile", "totalPages"
            FROM documents
            WHERE "companyId" = %s AND "sourceFile" = %s
            """,
            (company_id, document_id),
        )
        row = cur.fetchone()

    if row is None:
        return None
    return {"id": row[0], "title": row[1], "sourceFile": row[2], "totalPages": row[3]}


def _extract_pages(pdf_bytes: bytes) -> list[dict]:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append({
                "page_number": i,
                "content":     text,
                "token_count": len(text.split()),
            })
    return pages


# ── route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=APIResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF document (auto-creates tenant / appends if exists)",
)
def ingest_document(
    tenant_id:   str        = Form(..., description="Tenant identifier (Company slug)."),
    user_id:     str        = Form(..., description="Caller's user identifier (logged only)."),
    document_id: str        = Form(..., description="Logical document name / key."),
    file:        UploadFile = File(...),
):
    """
    Full ingestion pipeline in one call:

    1. Validate the PDF.
    2. Auto-create the Company (tenant) if it does not exist.
    3. If the Document already exists for this tenant → append new pages
       (page numbers continue from where the last page left off) and rebuild tree.
    4. If the Document is new → create it, insert all pages, build tree.

    Result: the document is immediately queryable via POST /query/search.
    """
    logger.info(
        "ingest | tenant=%s | user=%s | doc=%s | file=%s",
        tenant_id, user_id, document_id, file.filename,
    )

    # ── 1. validate file ──────────────────────────────────────────────────────
    pdf_bytes = file.file.read()

    if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file does not appear to be a valid PDF.",
        )

    # ── 2. extract pages ──────────────────────────────────────────────────────
    try:
        extracted = _extract_pages(pdf_bytes)
    except Exception as exc:
        logger.exception("PDF extraction failed.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract text from PDF: {exc}",
        ) from exc

    if not extracted:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="PDF appears to be empty — no pages extracted.",
        )

    # Safe defaults — prevent UnboundLocalError if an unexpected path is hit
    db_document_id: str = ""
    total_pages: int = 0
    version: int = 0

    with get_conn() as conn:
        # ── 3. resolve / create company ───────────────────────────────────────
        company_id = _get_or_create_company(tenant_id, conn)

        # ── 4. resolve / create document ──────────────────────────────────────
        existing_doc = _get_document(company_id, document_id, conn)
        action = "appended" if existing_doc else "created"

        if existing_doc:
            # ── APPEND branch ─────────────────────────────────────────────────
            db_document_id = existing_doc["id"]
            existing_total: int = existing_doc["totalPages"]

            # Page numbers continue from after the last existing page
            page_offset = existing_total

            with conn.cursor() as cur:
                for p in extracted:
                    new_page_number = page_offset + p["page_number"]
                    cur.execute(
                        """
                        INSERT INTO pages
                            (id, "documentId", "pageNumber", content, "tokenCount", "createdAt", "updatedAt")
                        VALUES (gen_random_uuid()::text, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT ("documentId", "pageNumber") DO UPDATE
                            SET content      = EXCLUDED.content,
                                "tokenCount" = EXCLUDED."tokenCount",
                                "updatedAt"  = NOW()
                        """,
                        (db_document_id, new_page_number, p["content"], p["token_count"]),
                    )

                # Update totalPages
                new_total = existing_total + len(extracted)
                cur.execute(
                    'UPDATE documents SET "totalPages" = %s, "updatedAt" = NOW() WHERE id = %s',
                    (new_total, db_document_id),
                )
            conn.commit()

            logger.info(
                "Appended %d pages to doc id=%s (was %d, now %d).",
                len(extracted), db_document_id, existing_total, new_total,
            )
            total_pages = new_total

        else:
            # ── CREATE branch ─────────────────────────────────────────────────
            title = document_id  # use logical document_id as the title

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                        (id, "companyId", title, "sourceFile", "totalPages", status, "createdAt", "updatedAt")
                    VALUES (gen_random_uuid()::text, %s, %s, %s, %s, 'READY', NOW(), NOW())
                    RETURNING id
                    """,
                    (company_id, title, document_id, len(extracted)),
                )
                db_document_id = cur.fetchone()[0]
            conn.commit()

            with conn.cursor() as cur:
                for p in extracted:
                    cur.execute(
                        """
                        INSERT INTO pages
                            (id, "documentId", "pageNumber", content, "tokenCount", "createdAt", "updatedAt")
                        VALUES (gen_random_uuid()::text, %s, %s, %s, %s, NOW(), NOW())
                        ON CONFLICT ("documentId", "pageNumber") DO NOTHING
                        """,
                        (db_document_id, p["page_number"], p["content"], p["token_count"]),
                    )
            conn.commit()

            total_pages = len(extracted)
            logger.info(
                "Created doc id=%s with %d pages (tenant=%s, doc=%s).",
                db_document_id, total_pages, tenant_id, document_id,
            )

        # ── 5. generate page metadata ─────────────────────────────────────────
        try:
            generate_and_store_metadata(db_document_id, conn, llm)
        except Exception as exc:
            logger.exception("Metadata generation failed for doc id=%s.", db_document_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Page metadata generation failed: {exc}",
            ) from exc

        # ── 6. build / rebuild tree ───────────────────────────────────────────
        try:
            tree_dict = build_tree(db_document_id, conn, llm)
            version   = upsert_tree(db_document_id, tree_dict, conn)
        except Exception as exc:
            logger.exception("Tree build failed for doc id=%s.", db_document_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Tree construction failed: {exc}",
            ) from exc

    return APIResponse(
        message=(
            f"Document '{document_id}' {action} successfully for tenant '{tenant_id}'. "
            f"{total_pages} total pages indexed, tree version {version}."
        ),
        data=IngestOut(
            tenant_id=tenant_id,
            document_id=document_id,
            db_document_id=db_document_id,
            title=document_id,
            total_pages=total_pages,
            tree_version=version,
            action=action,
        ).model_dump(),
    )