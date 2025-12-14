#!/usr/bin/env python
"""Enhanced script to index XPU entries from JSONL files into PostgreSQL vector database.

This script provides:
1. Indexing XPU entries with proper embedding generation
2. Database verification and statistics
3. Query capabilities to inspect indexed data

The embedding is generated from the text built by build_xpu_text(), which includes:
- Context: Language, Tools, Python versions, OS
- Signals: Keywords, Error patterns (regex)
- Advice: advice_nl text

This matches the retrieval strategy used during inference.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add project root to path


from build_agent.xpu.xpu_adapter import XpuEntry, load_xpu_entries
from build_agent.xpu.xpu_vector_store import (
    XpuVectorStore,
    build_xpu_text,
    text_to_embedding,
    EMBEDDING_DIM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_db_connection(dns: str):
    """Get a direct database connection for queries."""
    return psycopg2.connect(dns)


def verify_database(dns: str) -> Dict[str, Any]:
    """Verify database connection and return table statistics."""
    logger.info("Verifying database connection...")
    conn = get_db_connection(dns)
    try:
        with conn.cursor() as cur:
            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'xpu_entries'
                );
            """)
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                return {
                    "table_exists": False,
                    "message": "Table xpu_entries does not exist. Run indexing to create it.",
                }
            
            # Get table structure
            cur.execute("""
                SELECT 
                    column_name,
                    data_type,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'xpu_entries'
                ORDER BY ordinal_position;
            """)
            columns = cur.fetchall()
            
            # Get row count
            cur.execute("SELECT COUNT(*) FROM xpu_entries;")
            row_count = cur.fetchone()[0]
            
            # Get sample entries
            cur.execute("""
                SELECT id, context, signals, advice_nl, atoms, created_at
                FROM xpu_entries
                LIMIT 3;
            """)
            samples = cur.fetchall()
            
            # Check embedding dimension
            cur.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM xpu_entries) as total,
                    (SELECT COUNT(*) FROM xpu_entries WHERE embedding IS NOT NULL) as with_embedding;
            """)
            embedding_stats = cur.fetchone()
            
            return {
                "table_exists": True,
                "row_count": row_count,
                "columns": [
                    {
                        "name": col[0],
                        "type": col[1],
                        "max_length": col[2],
                    }
                    for col in columns
                ],
                "embedding_stats": {
                    "total": embedding_stats[0],
                    "with_embedding": embedding_stats[1],
                },
                "sample_entries": [
                    {
                        "id": s[0],
                        "context": s[1],
                        "signals": s[2],
                        "advice_nl": s[3][:2] if s[3] else [],  # First 2 items
                        "atoms_count": len(s[4]) if s[4] else 0,
                        "created_at": str(s[5]),
                    }
                    for s in samples
                ],
            }
    finally:
        conn.close()


def index_xpu_file(
    jsonl_path: Path,
    vector_store: XpuVectorStore,
    batch_size: int = 10,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Index all XPU entries from a JSONL file.
    
    Returns:
        Dictionary with statistics: {'indexed': int, 'failed': int, 'skipped': int}
    """
    logger.info("Loading XPU entries from %s", jsonl_path)
    entries = load_xpu_entries(jsonl_path)
    logger.info("Found %d XPU entries", len(entries))
    
    if dry_run:
        logger.info("DRY RUN: Would index %d entries", len(entries))
        # Show what would be indexed
        for i, entry in enumerate(entries[:3]):
            text = build_xpu_text(entry)
            logger.info("Entry %d: id=%s, embedding_text_length=%d", i + 1, entry.id, len(text))
            logger.debug("Embedding text preview: %s", text[:200])
        return {"indexed": 0, "failed": 0, "skipped": len(entries)}
    
    indexed = 0
    failed = 0
    skipped = 0
    
    # Use tqdm for progress bar
    with tqdm(total=len(entries), desc="Indexing XPU entries", unit="entry") as pbar:
        for i, entry in enumerate(entries):
            try:
                # Build searchable text (this is what gets embedded)
                text = build_xpu_text(entry)
                
                if not text.strip():
                    logger.warning("Skipping %s: empty embedding text", entry.id)
                    skipped += 1
                    pbar.update(1)
                    continue
                
                # Generate embedding
                logger.debug("Generating embedding for %s (text length: %d)", entry.id, len(text))
                embedding = text_to_embedding(text)
                
                if len(embedding) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}"
                    )
                
                # Store in database
                vector_store.upsert_entry(entry, embedding)
                indexed += 1
                
                if (i + 1) % batch_size == 0:
                    logger.info("Indexed %d/%d entries", i + 1, len(entries))
                
            except Exception as e:
                logger.error("Failed to index %s: %s", entry.id, e, exc_info=True)
                failed += 1
            finally:
                pbar.update(1)
    
    logger.info("Indexing complete: %d succeeded, %d failed, %d skipped", indexed, failed, skipped)
    return {"indexed": indexed, "failed": failed, "skipped": skipped}


def query_entry(dns: str, xpu_id: str) -> Optional[Dict[str, Any]]:
    """Query a specific XPU entry by ID."""
    conn = get_db_connection(dns)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, context, signals, advice_nl, atoms, created_at
                FROM xpu_entries
                WHERE id = %s;
            """, (xpu_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "context": row[1],
                "signals": row[2],
                "advice_nl": row[3],
                "atoms": row[4],
                "created_at": str(row[5]),
            }
    finally:
        conn.close()


def search_similar(
    dns: str, query_text: str, k: int = 5, min_similarity: float = 0.3
) -> List[Dict[str, Any]]:
    """Search for similar XPU entries using vector similarity."""
    logger.info("Generating embedding for query text...")
    query_embedding = text_to_embedding(query_text)
    
    conn = get_db_connection(dns)
    try:
        with conn.cursor() as cur:
            embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
            
            cur.execute("""
                SELECT 
                    id,
                    context,
                    signals,
                    advice_nl,
                    atoms,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM xpu_entries
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_str, embedding_str, min_similarity, embedding_str, k))
            
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "context": row[1],
                    "signals": row[2],
                    "advice_nl": row[3][:2] if row[3] else [],  # First 2 items
                    "atoms_count": len(row[4]) if row[4] else 0,
                    "similarity": float(row[5]),
                }
                for row in rows
            ]
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index XPU entries into vector database with verification capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index XPU entries from a JSONL file
  python exp/scripts/index_xpu_to_vector_db_enhanced.py index --input exp/xpu_v0.jsonl

  # Verify database and show statistics
  python exp/scripts/index_xpu_to_vector_db_enhanced.py verify

  # Query a specific entry
  python exp/scripts/index_xpu_to_vector_db_enhanced.py query --id xpu_5458157951

  # Search for similar entries
  python exp/scripts/index_xpu_to_vector_db_enhanced.py search --query "numpy error ufunc"

  # Dry run (show what would be indexed)
  python exp/scripts/index_xpu_to_vector_db_enhanced.py index --input exp/xpu_v0.jsonl --dry-run
        """,
    )
    
    parser.add_argument(
        "--dns",
        type=str,
        default=None,
        help="PostgreSQL connection string (default: from dns environment variable)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index XPU entries from JSONL file")
    index_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to XPU JSONL file (e.g., exp/xpu_v0.jsonl)",
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Log progress every N entries",
    )
    index_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually indexing",
    )
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify database and show statistics")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a specific XPU entry by ID")
    query_parser.add_argument("--id", type=str, required=True, help="XPU entry ID")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar XPU entries")
    search_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text to search for",
    )
    search_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    search_parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.3,
        help="Minimum similarity threshold (default: 0.3)",
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    load_dotenv()
    
    # Get DNS from args or environment
    dns = args.dns or os.environ.get("dns")
    if not dns:
        raise RuntimeError(
            "Missing database connection string. "
            "Set --dns argument or dns environment variable."
        )
    
    if args.command == "index":
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        vector_store = XpuVectorStore(connection_string=dns)
        try:
            stats = index_xpu_file(
                args.input,
                vector_store,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                logger.info("Indexing statistics: %s", stats)
        finally:
            vector_store.close()
    
    elif args.command == "verify":
        stats = verify_database(dns)
        print("\n" + "=" * 70)
        print("Database Verification Results")
        print("=" * 70)
        if not stats.get("table_exists"):
            print(f"❌ {stats['message']}")
        else:
            print(f"✅ Table exists")
            print(f"\nStatistics:")
            print(f"  Total entries: {stats['row_count']}")
            print(f"  Entries with embedding: {stats['embedding_stats']['with_embedding']}")
            print(f"\nTable structure:")
            for col in stats["columns"]:
                max_len = f", max_length={col['max_length']}" if col["max_length"] else ""
                print(f"  - {col['name']}: {col['type']}{max_len}")
            if stats["sample_entries"]:
                print(f"\nSample entries (first 3):")
                for entry in stats["sample_entries"]:
                    print(f"  - {entry['id']}: {len(entry['advice_nl'])} advice items, "
                          f"{entry['atoms_count']} atoms")
        print("=" * 70)
    
    elif args.command == "query":
        entry = query_entry(dns, args.id)
        if not entry:
            print(f"❌ Entry not found: {args.id}")
        else:
            print("\n" + "=" * 70)
            print(f"XPU Entry: {entry['id']}")
            print("=" * 70)
            print(f"Context: {json.dumps(entry['context'], indent=2, ensure_ascii=False)}")
            print(f"\nSignals: {json.dumps(entry['signals'], indent=2, ensure_ascii=False)}")
            print(f"\nAdvice ({len(entry['advice_nl'])} items):")
            for i, advice in enumerate(entry["advice_nl"], 1):
                print(f"  {i}. {advice}")
            print(f"\nAtoms ({len(entry['atoms'])} items):")
            for i, atom in enumerate(entry["atoms"], 1):
                print(f"  {i}. {atom['name']}: {atom['args']}")
            print(f"\nCreated at: {entry['created_at']}")
            print("=" * 70)
    
    elif args.command == "search":
        results = search_similar(dns, args.query, k=args.k, min_similarity=args.min_similarity)
        print("\n" + "=" * 70)
        print(f"Search Results for: '{args.query}'")
        print(f"Found {len(results)} similar entries (min_similarity={args.min_similarity})")
        print("=" * 70)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['id']} (similarity: {result['similarity']:.3f})")
            print(f"   Context: {json.dumps(result['context'], ensure_ascii=False)}")
            if result["advice_nl"]:
                print(f"   Advice preview: {result['advice_nl'][0][:100]}...")
            print(f"   Atoms: {result['atoms_count']}")


if __name__ == "__main__":
    main()

