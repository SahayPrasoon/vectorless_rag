"""
src/api/routers/tree.py
───────────────────────
Document-tree endpoints.

GET  /companies/{slug}/documents/{document_id}/tree
     — fetch the stored tree for a document

POST /companies/{slug}/documents/{document_id}/tree/build
     — trigger (re)build of the tree from scratch
       (runs: page_metadata → tree_builder in one request)
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, status

from src.config.db import get_conn
from src.config.llm import llm
from src.api.models import APIResponse, TreeOut, TreeNodeOut
from src.pipeline.tree_builder import build_tree, upsert_tree
from src.pipeline.page_metadata import generate_and_store_metadata

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tree"])


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _tree_node_out(node_dict: dict) -> dict:
    """Recursively convert the raw treeJson dict to a TreeNodeOut-compatible dict."""
    return {
        "title":     node_dict.get("title", ""),
        "type":      node_dict.get("type", "root"),
        "path":      node_dict.get("path", ""),
        "level":     node_dict.get("level", 0),
        "summary":   node_dict.get("summary", ""),
        "keywords":  node_dict.get("keywords", []),
        "pageStart": node_dict.get("pageStart"),
        "pageEnd":   node_dict.get("pageEnd"),
        "pageCount": node_dict.get("pageCount"),
        "pageIds":   node_dict.get("pageIds", []),
        "children":  [_tree_node_out(c) for c in node_dict.get("children", [])],
    }


# ── routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/companies/{slug}/documents/{document_id}/tree",
    response_model=APIResponse,
    summary="Fetch the stored tree for a document",
)
def get_tree(slug: str, document_id: str):
    with get_conn() as conn:
        _assert_company_owns_doc(slug, document_id, conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, "documentId", version, "treeJson", "createdAt"
                FROM trees
                WHERE "documentId" = %s
                """,
                (document_id,),
            )
            row = cur.fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No tree found for document '{document_id}'. "
                "Call POST .../tree/build first."
            ),
        )

    tree_id, doc_id, version, tree_json, created_at = row
    if isinstance(tree_json, str):
        tree_json = json.loads(tree_json)

    tree_out = TreeOut(
        documentId=doc_id,
        version=version,
        createdAt=created_at,
        tree=TreeNodeOut(**_tree_node_out(tree_json)),
    )
    return APIResponse(data=tree_out.model_dump())


@router.post(
    "/companies/{slug}/documents/{document_id}/tree/build",
    response_model=APIResponse,
    summary="Build (or rebuild) the document tree",
    status_code=status.HTTP_202_ACCEPTED,
)
def build_document_tree(slug: str, document_id: str):
    """
    Runs the full bottom-up pipeline synchronously:
      1. generate_and_store_metadata()  — LLM batch calls
      2. build_tree()                   — chapter / section / root summary
      3. upsert_tree()                  — persist to DB

    Use this to rebuild an existing tree without re-uploading the PDF.
    """
    with get_conn() as conn:
        _assert_company_owns_doc(slug, document_id, conn)

        logger.info("Rebuilding tree for doc %s (company slug=%s).", document_id, slug)

        try:
            # Step 1: page metadata
            page_count = generate_and_store_metadata(document_id, conn, llm)
            logger.info("Page metadata done: %d pages.", page_count)

            # Step 2: build tree
            tree_dict = build_tree(document_id, conn, llm)

            # Step 3: persist
            version = upsert_tree(document_id, tree_dict, conn)

        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception("Tree build failed for doc %s.", document_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Tree build failed: {exc}",
            ) from exc

    return APIResponse(
        message="Tree built successfully.",
        data={
            "documentId": document_id,
            "version":    version,
            "pageCount":  page_count,
        },
    )
