#!/usr/bin/env python
"""Index XPU entries from JSONL files into PostgreSQL vector database."""

import argparse
import logging
import sys
import os
from pathlib import Path

from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add project root to path


from build_agent.xpu.xpu_adapter import XpuEntry, load_xpu_entries
from build_agent.xpu.xpu_vector_store import XpuVectorStore, build_xpu_text, text_to_embedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def index_xpu_file(jsonl_path: Path, vector_store: XpuVectorStore, batch_size: int = 10) -> None:
    """Index all XPU entries from a JSONL file."""
    logger.info("Loading XPU entries from %s", jsonl_path)
    entries = load_xpu_entries(jsonl_path)
    logger.info("Found %d XPU entries", len(entries))
    
    indexed = 0
    failed = 0
    
    for i, entry in enumerate(entries):
        try:
            # Build searchable text
            text = build_xpu_text(entry)
            
            # Generate embedding
            logger.debug("Generating embedding for %s", entry.id)
            embedding = text_to_embedding(text)
            
            # Store in database
            vector_store.upsert_entry(entry, embedding)
            indexed += 1
            
            if (i + 1) % batch_size == 0:
                logger.info("Indexed %d/%d entries", i + 1, len(entries))
        except Exception as e:
            logger.error("Failed to index %s: %s", entry.id, e, exc_info=True)
            failed += 1
    
    logger.info("Indexing complete: %d succeeded, %d failed", indexed, failed)
    if failed > 0:
        raise RuntimeError(f"Failed to index {failed} entries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index XPU entries into vector database")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to XPU JSONL file (e.g., exp/xpu_v0.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Log progress every N entries",
    )
    args = parser.parse_args()
    
    load_dotenv()
    
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    vector_store = XpuVectorStore()
    try:
        index_xpu_file(args.input, vector_store, args.batch_size)
    finally:
        vector_store.close()


if __name__ == "__main__":
    main()

