"""
src/pipeline/page_metadata.py
──────────────────────────────
Batch LLM metadata generation for pages.

IDs are cuid strings (Prisma default).

Public API
──────────
  generate_and_store_metadata(db_document_id, conn, llm, batch_size) -> int
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
    # batch: [(id, pageNumber, content), ...]
    page_text_blocks = [
        f"Page Number: {page_number}\n\n{content}"
        for _, page_number, content in batch
    ]
    prompt = PROMPT.format(pages="\n\n".join(page_text_blocks))
    response = llm.invoke(prompt)
    return _parse_llm_json(response.content)


def generate_and_store_metadata(
    db_document_id: str,
    conn: psycopg.Connection,
    llm: BaseChatModel,
    batch_size: int = BATCH_SIZE,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, "pageNumber", content
            FROM pages
            WHERE "documentId" = %s
              AND metadata IS NULL
            ORDER BY "pageNumber"
            """,
            (db_document_id,),
        )
        pages = cur.fetchall()

    if not pages:
        logger.info("All pages already have metadata for documentId=%s — skipping.", db_document_id)
        return 0

    logger.info("Generating metadata for %d pages (batch=%d).", len(pages), batch_size)

    all_metadata: list[dict] = []
    for batch in _batches(list(pages), batch_size):
        all_metadata.extend(_generate_batch(batch, llm))
        logger.debug("Processed pages %s–%s.", batch[0][1], batch[-1][1])

    with conn.cursor() as cur:
        for meta in all_metadata:
            cur.execute(
                """
                UPDATE pages
                SET metadata = %s, "updatedAt" = NOW()
                WHERE "documentId" = %s AND "pageNumber" = %s
                """,
                (json.dumps(meta), db_document_id, meta["pageNumber"]),
            )

    conn.commit()
    logger.info("Metadata stored for %d pages.", len(all_metadata))
    return len(all_metadata)