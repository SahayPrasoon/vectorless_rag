"""
src/api/models.py
─────────────────
Pydantic schemas for every request body and response envelope
used across the Vectorless-RAG API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Generic envelope ──────────────────────────────────────────────────────────

class APIResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Any = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: str | None = None


# ── Company ───────────────────────────────────────────────────────────────────

class CompanyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200,
                      description="Display name of the company.")
    slug: str | None = Field(
        default=None,
        description="URL-safe slug. Auto-derived from name if omitted.",
        pattern=r"^[a-z0-9-]+$",
    )
    description: str | None = Field(default=None, max_length=500)


class CompanyOut(BaseModel):
    id: str
    name: str
    slug: str
    createdAt: datetime


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: str
    companyId: str
    title: str
    sourceFile: str
    totalPages: int
    status: str = "PROCESSING"   # PROCESSING | READY | FAILED
    hasTree: bool = False
    createdAt: datetime


class DocumentListOut(BaseModel):
    companyId: str
    companySlug: str
    documents: list[DocumentOut]
    total: int


# ── Tree ──────────────────────────────────────────────────────────────────────

class TreeNodeOut(BaseModel):
    """Recursive – mirrors the treeJson shape stored in DB."""
    title: str
    type: str                         # root | chapter | section
    path: str
    level: int
    summary: str = ""
    keywords: list[str] = []
    pageStart: int | None = None
    pageEnd: int | None = None
    pageCount: int | None = None
    pageIds: list[str] = []
    children: list["TreeNodeOut"] = []

    class Config:
        populate_by_name = True


TreeNodeOut.model_rebuild()


class TreeOut(BaseModel):
    documentId: str
    version: int
    createdAt: datetime
    tree: TreeNodeOut


# ── Answer / RAG ──────────────────────────────────────────────────────────────

class AnswerRequest(BaseModel):
    query: str = Field(
        ..., min_length=3, max_length=2000,
        description="Natural-language question to answer.",
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Max number of pages to feed the LLM.",
    )


class TreePathNodeOut(BaseModel):
    type: str
    title: str
    path: str


class AnswerOut(BaseModel):
    query: str
    answer: str
    sources: list[int]               # page numbers used
    treePath: list[TreePathNodeOut]
    latencySeconds: float
    llmCalls: int = 1
