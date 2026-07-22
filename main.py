"""
main.py
───────
Vectorless-RAG  —  FastAPI application entry point.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoint map
────────────
  POST   /api/v1/companies                                          create company
  GET    /api/v1/companies                                          list companies
  GET    /api/v1/companies/{slug}                                   get company
  GET    /api/v1/companies/{slug}/documents                         list documents
  POST   /api/v1/companies/{slug}/documents                         upload PDF → full pipeline
  GET    /api/v1/companies/{slug}/documents/{id}                    document detail
  DELETE /api/v1/companies/{slug}/documents/{id}                    delete document
  GET    /api/v1/companies/{slug}/documents/{id}/tree               fetch tree
  POST   /api/v1/companies/{slug}/documents/{id}/tree/build         rebuild tree
  POST   /api/v1/companies/{slug}/documents/{id}/answer             RAG query
  GET    /health
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.db import init_pool, close_pool
from src.api.routers import company, tree, answer

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Vectorless-RAG API …")
    init_pool()
    yield
    logger.info("Shutting down Vectorless-RAG API …")
    close_pool()


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vectorless-RAG API",
    description=(
        "Company-scoped document intelligence API.\n\n"
        "**Full pipeline on upload:** `POST /companies/{slug}/documents`\n\n"
        "1. Extract pages from PDF\n"
        "2. Generate page metadata (LLM batch calls)\n"
        "3. Build hierarchical chapter → section tree (LLM bottom-up)\n"
        "4. Document is immediately queryable\n\n"
        "**Query:** `POST /companies/{slug}/documents/{id}/answer`  — 1 LLM call total"
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── CORS (tighten origins for production) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request latency logging middleware ────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "%s %s → %d  [%.1f ms]",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ── global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "message": "Internal server error.", "detail": str(exc)},
    )


# ── routers ───────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(company.router, prefix=API_PREFIX)
app.include_router(tree.router,    prefix=API_PREFIX)
app.include_router(answer.router,  prefix=API_PREFIX)


# ── health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="Health check")
def health():
    return {"status": "ok", "service": "vectorless-rag", "version": "2.0.0"}


# ── root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"], summary="API root")
def root():
    return {
        "service": "Vectorless-RAG API",
        "version": "2.0.0",
        "docs":    "/docs",
        "endpoints": {
            "companies":  "/api/v1/companies",
            "upload_doc": "POST /api/v1/companies/{slug}/documents",
            "answer":     "POST /api/v1/companies/{slug}/documents/{id}/answer",
        },
    }
