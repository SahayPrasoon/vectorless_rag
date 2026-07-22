"""
src/pipeline/page_metadata.py
──────────────────────────────
Batch LLM metadata generation for pages (title, summary, keywords, etc.).

Public API
──────────
  generate_and_store_metadata(document_id, conn, llm, batch_size) -> int
    Returns the number of pages processed.
"""
from __future__ import annotations

import json
import logging

import psycopg
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

BATCH_SIZE = 5

PROMPT = """\
You are building page metadata for a Vectorless RAG indexing system.

You will receive multiple pages.

For EACH page generate metadata.

Return ONLY valid JSON — an array, no markdown, no extra text.

Schema:
[
  {{
    "pageNumber": 1,
    "title": "",
    "summary": "",
    "keywords": [],
    "entities": [],
    "topics": [],
    "pageType": "cover|toc|content|appendix|references",
    "containsTable": false,
    "containsFigure": false
  }}
]

Pages

{pages}
"""


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _parse_llm_json(text: str):
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _generate_batch(batch: list[tuple], llm: BaseChatModel) -> list[dict]:
    """
    batch: list of (id, pageNumber, content) tuples
    Returns list of metadata dicts.
    """
    page_text_blocks = []
    for _, page_number, content in batch:
        page_text_blocks.append(f"Page Number: {page_number}\n\n{content}")

    prompt = PROMPT.format(pages="\n\n".join(page_text_blocks))
    response = llm.invoke(prompt)
    return _parse_llm_json(response.content)


def generate_and_store_metadata(
    document_id: str,
    conn: psycopg.Connection,
    llm: BaseChatModel,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Generate page metadata in batches and persist it to the pages table.

    Returns the number of pages processed.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, "pageNumber", content
            FROM pages
            WHERE "documentId" = %s
            ORDER BY "pageNumber"
            """,
            (document_id,),
        )
        pages = cur.fetchall()

    if not pages:
        raise ValueError(f"No pages found for documentId={document_id!r}.")

    logger.info(
        "Generating metadata for %d pages (batch_size=%d).", len(pages), batch_size
    )

    all_metadata: list[dict] = []
    for batch in _batches(list(pages), batch_size):
        metadata = _generate_batch(batch, llm)
        all_metadata.extend(metadata)
        logger.debug(
            "Processed pages %s-%s.",
            batch[0][1], batch[-1][1],
        )

    with conn.cursor() as cur:
        for meta in all_metadata:
            cur.execute(
                """
                UPDATE pages
                SET metadata = %s
                WHERE "documentId" = %s AND "pageNumber" = %s
                """,
                (json.dumps(meta), document_id, meta["pageNumber"]),
            )

    conn.commit()
    logger.info("Metadata stored for %d pages.", len(all_metadata))
    return len(all_metadata)
