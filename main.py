"""
main.py
───────
Vectorless-RAG  —  FastAPI entry point.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoint map
────────────
  POST  /documentmcp/document/ingest       upload PDF → extract → metadata → tree
  POST  /documentmcp/query/search          RAG query (1 LLM call)
  GET   /health
  GET   /

Auth is handled by your upstream middleware — this server trusts all
incoming requests have already been validated.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.db import init_pool, close_pool
from src.api.routers import ingest, query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Vectorless-RAG API …")
    init_pool()
    yield
    logger.info("Shutting down Vectorless-RAG API …")
    close_pool()


app = FastAPI(
    title="Vectorless-RAG API",
    description=(
        "Tenant-scoped document intelligence — no vector embeddings.\n\n"
        "**Ingest:** `POST /documentmcp/document/ingest`  \n"
        "**Query:**  `POST /documentmcp/query/search`"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "%s %s → %d  [%.1f ms]",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "message": "Internal server error.", "detail": str(exc)},
    )


# ── routers — prefix matches your target URL pattern ─────────────────────────
PREFIX = "/documentmcp"

app.include_router(ingest.router, prefix=PREFIX)   # → /documentmcp/document/ingest
app.include_router(query.router,  prefix=PREFIX)   # → /documentmcp/query/search


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "vectorless-rag", "version": "2.0.0"}


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Vectorless-RAG API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "ingest": "POST /documentmcp/document/ingest",
            "query":  "POST /documentmcp/query/search",
        },
    }