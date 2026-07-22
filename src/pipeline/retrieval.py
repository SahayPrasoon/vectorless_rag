"""
src/pipeline/retrieval.py
─────────────────────────
Zero-LLM retrieval pipeline (keyword-overlap routing).

Public API
──────────
  load_tree_from_db(conn, document_id) -> TreeNode
  retrieve(query, document_id, conn, top_k) -> RetrievalResult
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import psycopg


# ── TreeNode ─────────────────────────────────────────────────────────────────

class TreeNode:
    """Lightweight in-memory wrapper around one node of the treeJson dict."""

    def __init__(self, data: dict[str, Any], parent: "TreeNode | None" = None):
        self.data = data
        self.parent = parent
        self.children: list["TreeNode"] = []

    @property
    def level(self) -> int:       return self.data.get("level", 0)
    @property
    def title(self) -> str:       return self.data.get("title", "")
    @property
    def summary(self) -> str:     return self.data.get("summary", "")
    @property
    def keywords(self) -> list:   return self.data.get("keywords", [])
    @property
    def path(self) -> str:        return self.data.get("path", "")
    @property
    def type(self) -> str:        return self.data.get("type", "root")
    @property
    def page_ids(self) -> list:   return self.data.get("pageIds", [])
    @property
    def page_start(self) -> int | None: return self.data.get("pageStart")
    @property
    def page_end(self) -> int | None:   return self.data.get("pageEnd")

    def __repr__(self) -> str:
        return f"TreeNode(level={self.level}, path={self.path!r}, title={self.title!r})"


def build_tree_nodes(tree_dict: dict, parent: TreeNode | None = None) -> TreeNode:
    """Recursively build TreeNode objects from the raw treeJson dict."""
    node = TreeNode(tree_dict, parent)
    for child_dict in tree_dict.get("children", []):
        node.children.append(build_tree_nodes(child_dict, parent=node))
    return node


# ── DB helpers ────────────────────────────────────────────────────────────────

def load_tree_from_db(conn: psycopg.Connection, document_id: str) -> TreeNode:
    """
    Load treeJson from the trees table and return the root TreeNode.
    Raises ValueError if no tree exists for the given document.
    """
    with conn.cursor() as cur:
        cur.execute(
            'SELECT "treeJson" FROM trees WHERE "documentId" = %s',
            (document_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise ValueError(
            f"No tree found for documentId={document_id!r}. "
            "Run the tree-builder pipeline first."
        )

    tree_dict = row[0]
    if isinstance(tree_dict, str):
        tree_dict = json.loads(tree_dict)

    return build_tree_nodes(tree_dict)


# ── Similarity scoring ────────────────────────────────────────────────────────

def keyword_overlap(query: str, node: TreeNode) -> float:
    """
    Score a tree node against a query using word-level overlap.
    Checks: node.summary + node.keywords + node.title.
    Returns overlap ratio in [0, 1] (normalised by query length).
    """
    query_words = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())
    if not query_words:
        return 0.0

    node_text = " ".join([
        node.summary,
        " ".join(node.keywords),
        node.title,
    ]).lower()
    node_words = set(re.sub(r"[^a-z0-9 ]", " ", node_text).split())

    overlap = query_words & node_words
    return len(overlap) / (len(query_words) + 1e-9)


# ── Traversal ─────────────────────────────────────────────────────────────────

def traverse_tree(query: str, root: TreeNode) -> dict:
    """
    Route a query to best chapter → best section using keyword overlap only.
    Returns a dict with keys: path (list[TreeNode]), leaf (TreeNode).
    """
    if not root.children:
        raise ValueError("Tree has no chapters — cannot traverse.")

    best_chapter = max(root.children, key=lambda ch: keyword_overlap(query, ch))

    if not best_chapter.children:
        raise ValueError(
            f"Chapter '{best_chapter.title}' has no sections — cannot traverse."
        )

    best_section = max(
        best_chapter.children,
        key=lambda sec: keyword_overlap(query, sec),
    )

    return {
        "path": [root, best_chapter, best_section],
        "leaf": best_section,
    }


# ── Page fetching ─────────────────────────────────────────────────────────────

def get_candidate_pages(
    conn: psycopg.Connection,
    leaf: TreeNode,
) -> list[dict]:
    """Return lightweight metadata for every page in the leaf section."""
    page_ids = leaf.page_ids
    if not page_ids:
        return []

    placeholders = ",".join(["%s"] * len(page_ids))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT "pageNumber", metadata
            FROM pages
            WHERE id IN ({placeholders})
            ORDER BY "pageNumber"
            """,
            page_ids,
        )
        rows = cur.fetchall()

    candidates = []
    for page_number, metadata in rows:
        meta = metadata or {}
        candidates.append({
            "pageNumber": page_number,
            "title":      meta.get("title", ""),
            "summary":    meta.get("summary", ""),
            "keywords":   meta.get("keywords", []),
        })
    return candidates


def fetch_pages_content(
    conn: psycopg.Connection,
    document_id: str,
    page_numbers: list[int],
) -> dict[int, str]:
    """Return {pageNumber: full_text} for the given pages."""
    if not page_numbers:
        return {}

    placeholders = ",".join(["%s"] * len(page_numbers))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT "pageNumber", content
            FROM pages
            WHERE "documentId" = %s
              AND "pageNumber" IN ({placeholders})
            ORDER BY "pageNumber"
            """,
            (document_id, *page_numbers),
        )
        rows = cur.fetchall()

    content_map = {pn: content for pn, content in rows}
    return {p: content_map[p] for p in page_numbers if p in content_map}


# ── Public retrieval function ─────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    tree_path: list[dict]          # [{type, title, path}, ...]
    leaf: dict                     # raw section data dict
    candidates: list[dict]         # candidate page metadata
    ranked_pages: list[int]        # page numbers in section order
    top_pages: list[int]           # top_k slice
    pages_content: dict[int, str]  # {pageNumber: text}


def retrieve(
    query: str,
    document_id: str,
    conn: psycopg.Connection,
    top_k: int = 5,
) -> RetrievalResult:
    """
    Full zero-LLM retrieval pipeline.

    1. Load tree from DB.
    2. Traverse: root → chapter → section (keyword overlap, 0 LLM calls).
    3. Fetch all pages in the winning section.
    4. Return top_k pages for answer generation.
    """
    tree_root = load_tree_from_db(conn, document_id)
    traversal = traverse_tree(query, tree_root)
    leaf = traversal["leaf"]

    candidates = get_candidate_pages(conn, leaf)
    page_numbers = [c["pageNumber"] for c in candidates]
    top_pages = page_numbers[:top_k]
    pages_content = fetch_pages_content(conn, document_id, top_pages)

    return RetrievalResult(
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
