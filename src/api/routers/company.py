"""
src/api/routers/company.py
──────────────────────────
Company-scoped endpoints.

GET  /companies                                          — list all companies
POST /companies                                          — create a company
GET  /companies/{slug}                                   — get single company by slug
GET  /companies/{slug}/documents                         — list documents for a company
GET  /companies/{slug}/documents/{document_id}           — single document detail
POST /companies/{slug}/documents                         — upload PDF → extract → metadata → tree
DELETE /companies/{slug}/documents/{document_id}         — delete document + cascade
"""
from __future__ import annotations

import io
import logging
import os
import re
from datetime import datetime, timezone

import pdfplumber
import psycopg
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.config.db import get_conn
from src.config.llm import llm
from src.api.models import (
    APIResponse,
    CompanyOut,
    CompanyCreate,
    DocumentOut,
    DocumentListOut,
)
from src.pipeline.page_metadata import generate_and_store_metadata
from src.pipeline.tree_builder import build_tree, upsert_tree

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/companies", tags=["Companies"])

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_company_by_slug(slug: str, conn: psycopg.Connection) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            'SELECT id, name, slug, "createdAt" FROM companies WHERE slug = %s',
            (slug,),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with slug '{slug}' not found.",
        )
    return {"id": row[0], "name": row[1], "slug": row[2], "createdAt": row[3]}


def _has_tree(conn: psycopg.Connection, document_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            'SELECT 1 FROM trees WHERE "documentId" = %s LIMIT 1',
            (document_id,),
        )
        return cur.fetchone() is not None


def _get_document_status(conn: psycopg.Connection, document_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            'SELECT status FROM documents WHERE id = %s',
            (document_id,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def _set_document_status(
    conn: psycopg.Connection, document_id: str, status_val: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            'UPDATE documents SET status = %s, "updatedAt" = NOW() WHERE id = %s',
            (status_val, document_id),
        )
    conn.commit()


def _slugify(text: str) -> str:
    """Convert a string to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_pages_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text from every page of a PDF.
    Returns list of {page_number, content, token_count}.
    Uses pdfplumber for accurate text extraction.
    """
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            pages.append({
                "page_number": i,
                "content": text,
                "token_count": len(text.split()),
            })
    return pages


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=APIResponse, summary="List all companies")
def list_companies():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id, name, slug, "createdAt" FROM companies ORDER BY name'
            )
            rows = cur.fetchall()

    companies = [
        CompanyOut(id=r[0], name=r[1], slug=r[2], createdAt=r[3])
        for r in rows
    ]
    return APIResponse(data={"companies": [c.model_dump() for c in companies], "total": len(companies)})


@router.post("", response_model=APIResponse, status_code=status.HTTP_201_CREATED,
             summary="Create a company")
def create_company(body: CompanyCreate):
    slug = body.slug or _slugify(body.name)
    with get_conn() as conn:
        # Check uniqueness
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id FROM companies WHERE slug = %s OR name = %s',
                (slug, body.name),
            )
            if cur.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Company with name '{body.name}' or slug '{slug}' already exists.",
                )
            cur.execute(
                """
                INSERT INTO companies (id, name, slug, description, "updatedAt")
                VALUES (gen_random_uuid()::text, %s, %s, %s, NOW())
                RETURNING id, name, slug, "createdAt"
                """,
                (body.name, slug, body.description),
            )
            row = cur.fetchone()
        conn.commit()

    company = CompanyOut(id=row[0], name=row[1], slug=row[2], createdAt=row[3])
    return APIResponse(
        message="Company created.",
        data=company.model_dump(),
    )


@router.get("/{slug}", response_model=APIResponse, summary="Get company by slug")
def get_company(slug: str):
    with get_conn() as conn:
        company = _get_company_by_slug(slug, conn)
    return APIResponse(data=CompanyOut(**company).model_dump())


@router.get(
    "/{slug}/documents",
    response_model=APIResponse,
    summary="List documents for a company",
)
def list_company_documents(slug: str):
    with get_conn() as conn:
        company = _get_company_by_slug(slug, conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, "companyId", title, "sourceFile", "totalPages", status, "createdAt"
                FROM documents
                WHERE "companyId" = %s
                ORDER BY "createdAt" DESC
                """,
                (company["id"],),
            )
            rows = cur.fetchall()

        documents = []
        for r in rows:
            doc_id = r[0]
            has_tree = _has_tree(conn, doc_id)
            documents.append(
                DocumentOut(
                    id=doc_id,
                    companyId=r[1],
                    title=r[2],
                    sourceFile=r[3],
                    totalPages=r[4],
                    status=r[5],
                    hasTree=has_tree,
                    createdAt=r[6],
                ).model_dump()
            )

    payload = DocumentListOut(
        companyId=company["id"],
        companySlug=slug,
        documents=documents,
        total=len(documents),
    )
    return APIResponse(data=payload.model_dump())


@router.get(
    "/{slug}/documents/{document_id}",
    response_model=APIResponse,
    summary="Get single document detail",
)
def get_company_document(slug: str, document_id: str):
    with get_conn() as conn:
        company = _get_company_by_slug(slug, conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, "companyId", title, "sourceFile", "totalPages", status, "createdAt"
                FROM documents
                WHERE id = %s AND "companyId" = %s
                """,
                (document_id, company["id"]),
            )
            row = cur.fetchone()

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found for company '{slug}'.",
            )

        has_tree = _has_tree(conn, document_id)

    doc = DocumentOut(
        id=row[0],
        companyId=row[1],
        title=row[2],
        sourceFile=row[3],
        totalPages=row[4],
        status=row[5],
        hasTree=has_tree,
        createdAt=row[6],
    )
    return APIResponse(data=doc.model_dump())


@router.post(
    "/{slug}/documents",
    response_model=APIResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload PDF → extract pages → generate metadata → build tree",
    description="""
Full synchronous pipeline triggered by a single multipart upload:

1. **Validate** — check company exists, file is a PDF, size ≤ MAX_UPLOAD_SIZE_MB.
2. **Extract** — parse every page with pdfplumber; store raw text in `pages`.
3. **Metadata** — batch LLM calls generate title/summary/keywords per page.
4. **Tree** — bottom-up LLM pipeline: chapters → sections → chapter summaries → root summary.
5. **Persist** — tree stored in `trees`; document status set to `READY`.

The document is queryable via `POST /{slug}/documents/{id}/answer` once this returns.
For large PDFs (>100 pages) consider offloading steps 3-4 to a background worker.
""",
)
def upload_document(slug: str, file: UploadFile = File(...)):
    # ── 1. validate ──────────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted.",
        )

    pdf_bytes = file.file.read()

    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB upload limit.",
        )

    if len(pdf_bytes) < 4 or pdf_bytes[:4] != b"%PDF":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file does not appear to be a valid PDF.",
        )

    with get_conn() as conn:
        company = _get_company_by_slug(slug, conn)
        company_id = company["id"]

        # ── 2. extract pages ─────────────────────────────────────────────────
        logger.info("Extracting pages from '%s' for company '%s'.", file.filename, slug)
        try:
            extracted_pages = _extract_pages_from_pdf(pdf_bytes)
        except Exception as exc:
            logger.exception("PDF extraction failed for '%s'.", file.filename)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not extract text from PDF: {exc}",
            ) from exc

        total_pages = len(extracted_pages)
        doc_title = file.filename.rsplit(".", 1)[0]  # filename without extension

        # Check for duplicate within this company
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id FROM documents WHERE "companyId" = %s AND "sourceFile" = %s',
                (company_id, file.filename),
            )
            existing = cur.fetchone()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"A document named '{file.filename}' already exists for company '{slug}'. "
                    f"Existing document id: {existing[0]}"
                ),
            )

        # ── 3. create document + pages rows ──────────────────────────────────
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (id, "companyId", title, "sourceFile", "totalPages", status, "updatedAt")
                VALUES
                    (gen_random_uuid()::text, %s, %s, %s, %s, 'PROCESSING', NOW())
                RETURNING id, "companyId", title, "sourceFile", "totalPages", status, "createdAt"
                """,
                (company_id, doc_title, file.filename, total_pages),
            )
            doc_row = cur.fetchone()
        conn.commit()

        document_id  = doc_row[0]
        logger.info(
            "Document created: id=%s title='%s' pages=%d",
            document_id, doc_title, total_pages,
        )

        # Bulk insert pages
        with conn.cursor() as cur:
            for p in extracted_pages:
                page_id = f"{document_id}_P{p['page_number']:04d}"
                cur.execute(
                    """
                    INSERT INTO pages
                        (id, "documentId", "pageNumber", content, "tokenCount", "updatedAt")
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT ("documentId", "pageNumber") DO NOTHING
                    """,
                    (
                        page_id,
                        document_id,
                        p["page_number"],
                        p["content"],
                        p["token_count"],
                    ),
                )
        conn.commit()
        logger.info("Pages stored: %d pages for doc %s.", total_pages, document_id)

        # ── 4. generate page metadata (LLM batch calls) ───────────────────────
        logger.info("Generating page metadata for doc %s …", document_id)
        try:
            page_count = generate_and_store_metadata(document_id, conn, llm)
            logger.info("Metadata done: %d pages.", page_count)
        except Exception as exc:
            _set_document_status(conn, document_id, "FAILED")
            logger.exception("Metadata generation failed for doc %s.", document_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Page metadata generation failed: {exc}",
            ) from exc

        # ── 5. build tree (bottom-up LLM pipeline) ────────────────────────────
        logger.info("Building tree for doc %s …", document_id)
        try:
            tree_dict = build_tree(document_id, conn, llm)
            version   = upsert_tree(document_id, tree_dict, conn)
            logger.info("Tree built: version=%d for doc %s.", version, document_id)
        except Exception as exc:
            _set_document_status(conn, document_id, "FAILED")
            logger.exception("Tree build failed for doc %s.", document_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Tree construction failed: {exc}",
            ) from exc

        # ── 6. mark document READY ────────────────────────────────────────────
        _set_document_status(conn, document_id, "READY")

    doc_out = DocumentOut(
        id=doc_row[0],
        companyId=doc_row[1],
        title=doc_row[2],
        sourceFile=doc_row[3],
        totalPages=doc_row[4],
        status="READY",
        hasTree=True,
        createdAt=doc_row[6],
    )

    return APIResponse(
        message=(
            f"Document uploaded, indexed, and tree built successfully. "
            f"{total_pages} pages processed, tree version {version}."
        ),
        data={
            "document": doc_out.model_dump(),
            "treeVersion": version,
            "pagesProcessed": total_pages,
        },
    )


@router.delete(
    "/{slug}/documents/{document_id}",
    response_model=APIResponse,
    summary="Delete a document and all its data (cascade)",
)
def delete_document(slug: str, document_id: str):
    with get_conn() as conn:
        company = _get_company_by_slug(slug, conn)

        with conn.cursor() as cur:
            cur.execute(
                'DELETE FROM documents WHERE id = %s AND "companyId" = %s RETURNING id',
                (document_id, company["id"]),
            )
            deleted = cur.fetchone()

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found for company '{slug}'.",
            )
        conn.commit()

    logger.info("Deleted document %s for company %s.", document_id, slug)
    return APIResponse(message=f"Document '{document_id}' deleted.")
