"""
src/api/models.py
─────────────────
Pydantic schemas for every request body and response envelope.

Design: all endpoints are flat — tenant_id / user_id / document_id
come in the request body (multipart for ingest, JSON for query).
No slugs, no path parameters for company/document identity.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Generic envelope ─────────────────────────────────────────────────────────

class APIResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Any = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: str | None = None


# ── Ingest ───────────────────────────────────────────────────────────────────

class IngestOut(BaseModel):
    tenant_id: str
    document_id: str         # caller-supplied logical document_id (= sourceFile key)
    db_document_id: str      # cuid string PK from the documents table
    title: str
    total_pages: int
    tree_version: int
    action: str              # "created" | "appended"


# ── Query / Search ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant identifier (maps to Company.slug).")
    user_id: str   = Field(..., description="Caller's user identifier (logged, not stored).")
    question: str  = Field(..., min_length=3, max_length=2000)
    document_id: str | None = Field(
        default=None,
        description=(
            "Optional: restrict search to one logical document. "
            "If omitted, all documents for the tenant are searched."
        ),
    )
    top_k: int = Field(default=5, ge=1, le=20)


class TreePathNodeOut(BaseModel):
    type: str
    title: str
    path: str


class QueryOut(BaseModel):
    tenant_id: str
    user_id: str
    question: str
    answer: str
    sources: list[int]
    document_searched: str
    tree_path: list[TreePathNodeOut]
    latency_seconds: float
    llm_calls: int = 1


# ── Tree (internal / debug) ───────────────────────────────────────────────────

class TreeNodeOut(BaseModel):
    title: str
    type: str
    path: str
    level: int
    summary: str = ""
    keywords: list[str] = []
    pageStart: int | None = None
    pageEnd: int | None = None
    pageCount: int | None = None
    pageIds: list[str] = []      # cuid strings, not ints
    children: list["TreeNodeOut"] = []

    class Config:
        populate_by_name = True


TreeNodeOut.model_rebuild()


class TreeOut(BaseModel):
    tenant_id: str
    document_id: str        # logical sourceFile key
    db_document_id: str     # cuid string PK
    version: int
    createdAt: datetime
    tree: TreeNodeOut