"""
src/pipeline/tree_builder.py
─────────────────────────────
Bottom-up tree construction pipeline (v2).

Build order
───────────
  1. detect_chapters()         — 1 LLM call, identifies chapter boundaries
  2. build_sections()          — N LLM calls, groups pages into sections per chapter
                                 (batched: SECTION_BATCH_SIZE pages per call)
  3. merge_chapter_summaries() — 1 LLM call per chapter
  4. merge_root_summary()      — 1 LLM call

Public API
──────────
  build_tree(document_id, conn, llm) -> dict   (the enriched treeJson)
  upsert_tree(document_id, tree_dict, conn) -> int  (version number)
"""
from __future__ import annotations

import json
import logging
from typing import Any

import psycopg
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

SECTION_BATCH_SIZE = 6


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _parse_llm_json(text: str) -> Any:
    """Strip markdown fences then parse JSON."""
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _invoke_json(llm: BaseChatModel, prompt: str) -> Any:
    response = llm.invoke(prompt)
    return _parse_llm_json(response.content)


# ── Step 0: fetch pages ───────────────────────────────────────────────────────

def fetch_pages(conn: psycopg.Connection, document_id: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, "pageNumber", content, metadata
            FROM pages
            WHERE "documentId" = %s
            ORDER BY "pageNumber"
            """,
            (document_id,),
        )
        rows = cur.fetchall()

    pages = []
    for row in rows:
        meta = row[3] if row[3] else {}
        pages.append({
            "id":         row[0],
            "pageNumber": row[1],
            "content":    row[2],
            "title":      meta.get("title", ""),
            "summary":    meta.get("summary", ""),
            "keywords":   meta.get("keywords", []),
            "topics":     meta.get("topics", []),
            "pageType":   meta.get("pageType", "content"),
        })
    return pages


# ── Step 1a: detect chapter boundaries ───────────────────────────────────────

CHAPTER_PROMPT = """\
You are identifying the top-level CHAPTER structure of a document for a Vectorless RAG system.

Group the pages below into chapters (major topic groups). Every page must fall inside exactly
one chapter's page range. Chapters must be in page order and cover the full range with no gaps.
Chapter titles must be descriptive (not "Chapter 1").

Return ONLY valid JSON. No markdown, no explanation.

Schema:
[
  {{"title": "<chapter title>", "pageStart": <int>, "pageEnd": <int>}}
]

Pages (pageNumber | title | summary):
{pages}
"""


def _format_pages_for_chapters(pages: list[dict]) -> str:
    lines = []
    for p in pages:
        summary_short = (p["summary"] or "")[:200].replace("\n", " ")
        lines.append(f"Page {p['pageNumber']} | {p['title'] or '(no title)'} | {summary_short}")
    return "\n".join(lines)


def detect_chapters(pages: list[dict], llm: BaseChatModel) -> list[dict]:
    prompt = CHAPTER_PROMPT.format(pages=_format_pages_for_chapters(pages))
    chapters = _invoke_json(llm, prompt)
    logger.info("Detected %d chapters.", len(chapters))
    return chapters


# ── Step 1b: group pages into sections ────────────────────────────────────────

SECTION_PROMPT = """\
You are grouping pages from ONE chapter of a document into SECTIONS for a Vectorless RAG system.

Chapter: "{chapter_title}"

Rules:
- Every page below must appear in exactly one section.
- A section spans one or more CONSECUTIVE pages on the same sub-topic.
- Section titles must be descriptive (not "Section 1").
- "summary" must be a 1-2 sentence summary MERGED from the page summaries in that section.
- "keywords" must be a short deduplicated list merged from the pages' keywords.
- pageIds must be the exact id strings provided — do not invent new ones.

Return ONLY valid JSON. No markdown, no explanation.

Schema:
[
  {{
    "title": "<section title>",
    "summary": "<merged summary>",
    "keywords": ["..."],
    "pageStart": <int>,
    "pageEnd": <int>,
    "pageIds": ["<page_id>", ...]
  }}
]

Pages (pageNumber | page_id | title | summary | keywords):
{pages}
"""


def _format_pages_for_sections(pages_batch: list[dict]) -> str:
    lines = []
    for p in pages_batch:
        summary_short = (p["summary"] or "")[:200].replace("\n", " ")
        kw = ", ".join(p["keywords"] or [])
        lines.append(
            f"Page {p['pageNumber']} | {p['id']} | {p['title'] or '(no title)'} | {summary_short} | {kw}"
        )
    return "\n".join(lines)


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _call_sections(chapter_title: str, batch: list[dict], llm: BaseChatModel) -> list[dict]:
    prompt = SECTION_PROMPT.format(
        chapter_title=chapter_title,
        pages=_format_pages_for_sections(batch),
    )
    return _invoke_json(llm, prompt)


def build_sections(chapters: list[dict], all_pages: list[dict], llm: BaseChatModel) -> None:
    """Mutates each chapter dict: adds chapter['sections']."""
    for chapter in chapters:
        chapter_pages = [
            p for p in all_pages
            if chapter["pageStart"] <= p["pageNumber"] <= chapter["pageEnd"]
        ]
        sections: list[dict] = []
        for batch in _batches(chapter_pages, SECTION_BATCH_SIZE):
            sections.extend(_call_sections(chapter["title"], batch, llm))
        chapter["sections"] = sections
        logger.info(
            "Chapter '%s': %d sections from %d pages.",
            chapter["title"], len(sections), len(chapter_pages),
        )


# ── Step 2: merge section summaries → chapter summary ─────────────────────────

CHAPTER_SUMMARY_PROMPT = """\
You are writing a single merged summary for a document CHAPTER, based on the summaries of its
sections, for a Vectorless RAG system.

Chapter title: "{chapter_title}"

Section summaries:
{sections}

Return ONLY valid JSON. No markdown, no explanation.

Schema:
{{"summary": "<2-3 sentence merged summary covering the whole chapter>", "keywords": ["..."]}}
"""


def _format_sections_for_prompt(sections: list[dict]) -> str:
    lines = []
    for s in sections:
        kw = ", ".join(s.get("keywords", []) or [])
        lines.append(f"- {s['title']}: {s['summary']} (keywords: {kw})")
    return "\n".join(lines)


def merge_chapter_summaries(chapters: list[dict], llm: BaseChatModel) -> None:
    """Mutates each chapter dict: adds chapter['summary'] and chapter['keywords']."""
    for chapter in chapters:
        prompt = CHAPTER_SUMMARY_PROMPT.format(
            chapter_title=chapter["title"],
            sections=_format_sections_for_prompt(chapter["sections"]),
        )
        merged = _invoke_json(llm, prompt)
        chapter["summary"] = merged.get("summary", "")
        chapter["keywords"] = merged.get("keywords", [])
        logger.info("Chapter '%s' summary merged.", chapter["title"])


# ── Step 3: merge chapter summaries → root summary ────────────────────────────

ROOT_SUMMARY_PROMPT = """\
You are writing a single merged summary for an ENTIRE document, based on the summaries of its
chapters, for a Vectorless RAG system.

Document title: "{doc_title}"

Chapter summaries:
{chapters}

Return ONLY valid JSON. No markdown, no explanation.

Schema:
{{"summary": "<3-4 sentence merged summary covering the whole document>", "keywords": ["..."]}}
"""


def _format_chapters_for_prompt(chapters: list[dict]) -> str:
    lines = []
    for c in chapters:
        kw = ", ".join(c.get("keywords", []) or [])
        lines.append(f"- {c['title']}: {c['summary']} (keywords: {kw})")
    return "\n".join(lines)


def merge_root_summary(doc_title: str, chapters: list[dict], llm: BaseChatModel) -> tuple[str, list[str]]:
    prompt = ROOT_SUMMARY_PROMPT.format(
        doc_title=doc_title,
        chapters=_format_chapters_for_prompt(chapters),
    )
    merged = _invoke_json(llm, prompt)
    return merged.get("summary", ""), merged.get("keywords", [])


# ── Assemble & enrich ─────────────────────────────────────────────────────────

def _assemble_tree(
    doc_title: str,
    root_summary: str,
    root_keywords: list[str],
    chapters: list[dict],
) -> dict:
    tree: dict[str, Any] = {
        "title":    doc_title,
        "type":     "root",
        "summary":  root_summary,
        "keywords": root_keywords,
        "children": [],
    }
    for chapter in chapters:
        chapter_node: dict[str, Any] = {
            "title":    chapter["title"],
            "type":     "chapter",
            "summary":  chapter.get("summary", ""),
            "keywords": chapter.get("keywords", []),
            "children": [],
        }
        for section in chapter["sections"]:
            chapter_node["children"].append({
                "title":     section["title"],
                "type":      "section",
                "summary":   section.get("summary", ""),
                "keywords":  section.get("keywords", []),
                "pageStart": section["pageStart"],
                "pageEnd":   section["pageEnd"],
                "pageIds":   section["pageIds"],
            })
        tree["children"].append(chapter_node)
    return tree


def _enrich_and_validate(tree: dict, all_pages: list[dict]) -> dict:
    all_page_ids = {p["id"] for p in all_pages}
    page_id_to_num = {p["id"]: p["pageNumber"] for p in all_pages}
    seen_ids: set[str] = set()

    chapter_idx = 0
    for chapter in tree.get("children", []):
        chapter_idx += 1
        chapter["level"] = 1
        chapter["path"] = str(chapter_idx)

        section_idx = 0
        ch_page_nums: list[int] = []

        for section in chapter.get("children", []):
            section_idx += 1
            section["level"] = 2
            section["path"] = f"{chapter_idx}.{section_idx}"

            s_ids = section.get("pageIds", [])
            s_nums = sorted(
                [page_id_to_num[pid] for pid in s_ids if pid in page_id_to_num]
            )
            if s_nums:
                section["pageStart"] = s_nums[0]
                section["pageEnd"] = s_nums[-1]
            section["pageCount"] = len(s_ids)

            seen_ids.update(s_ids)
            ch_page_nums.extend(s_nums)

        if ch_page_nums:
            chapter["pageStart"] = min(ch_page_nums)
            chapter["pageEnd"] = max(ch_page_nums)
        chapter["pageCount"] = len(ch_page_nums)

    tree["level"] = 0
    tree["path"] = "root"
    tree["pageCount"] = len(all_pages)
    tree["pageStart"] = min(p["pageNumber"] for p in all_pages)
    tree["pageEnd"] = max(p["pageNumber"] for p in all_pages)

    missing = all_page_ids - seen_ids
    if missing:
        missing_nums = sorted(
            [page_id_to_num[pid] for pid in missing if pid in page_id_to_num]
        )
        logger.warning(
            "%d pages not assigned to any section: %s", len(missing), missing_nums
        )
    else:
        logger.info("All pages accounted for in tree.")

    return tree


# ── DB persistence ────────────────────────────────────────────────────────────

def upsert_tree(
    document_id: str,
    tree_dict: dict,
    conn: psycopg.Connection,
) -> int:
    """
    Insert or update the Tree row for document_id.
    Returns the new version number.
    """
    tree_json_str = json.dumps(tree_dict)

    with conn.cursor() as cur:
        cur.execute(
            'SELECT id, version FROM trees WHERE "documentId" = %s',
            (document_id,),
        )
        existing = cur.fetchone()

        if existing:
            tree_id, current_version = existing
            new_version = current_version + 1
            cur.execute(
                """
                UPDATE trees
                SET "treeJson" = %s,
                    version    = %s,
                    "updatedAt" = NOW()
                WHERE id = %s
                """,
                (tree_json_str, new_version, tree_id),
            )
            logger.info(
                "Tree updated for doc %s — version %d.", document_id, new_version
            )
        else:
            new_version = 1
            cur.execute(
                """
                INSERT INTO trees (id, "documentId", "treeJson", version, "updatedAt")
                VALUES (gen_random_uuid()::text, %s, %s, 1, NOW())
                """,
                (document_id, tree_json_str),
            )
            logger.info("Tree inserted for doc %s.", document_id)

    conn.commit()
    return new_version


# ── Top-level orchestrator ────────────────────────────────────────────────────

def build_tree(
    document_id: str,
    conn: psycopg.Connection,
    llm: BaseChatModel,
) -> dict:
    """
    Full bottom-up pipeline.

    Returns the enriched treeJson dict (not yet persisted).
    Call upsert_tree() to save it.
    """
    # 0. fetch pages (metadata must already be populated)
    pages = fetch_pages(conn, document_id)
    if not pages:
        raise ValueError(f"No pages found for documentId={document_id!r}.")

    # fetch document title
    with conn.cursor() as cur:
        cur.execute('SELECT title FROM documents WHERE id = %s', (document_id,))
        doc_row = cur.fetchone()
    doc_title = doc_row[0] if doc_row else "Document"

    # 1a. chapters
    chapters = detect_chapters(pages, llm)

    # 1b. sections
    build_sections(chapters, pages, llm)

    # 2. chapter summaries
    merge_chapter_summaries(chapters, llm)

    # 3. root summary
    root_summary, root_keywords = merge_root_summary(doc_title, chapters, llm)

    # 4. assemble + enrich
    raw_tree = _assemble_tree(doc_title, root_summary, root_keywords, chapters)
    enriched = _enrich_and_validate(raw_tree, pages)

    return enriched
