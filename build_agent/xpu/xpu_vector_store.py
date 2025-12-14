"""Vector store for XPU entries using PostgreSQL with pgvector extension."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool

from build_agent.xpu.xpu_adapter import XpuEntry, XpuContext

logger = logging.getLogger(__name__)

# Embedding dimension (can be overridden by EMBEDDING_DIM env var)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))


def get_db_connection_string() -> str:
    """Get database connection string from environment."""
    dns = os.environ.get("dns")
    if not dns:
        raise RuntimeError("Missing required environment variable: dns (PostgreSQL connection string)")
    return dns


def create_xpu_table(conn) -> None:
    """Create XPU table with vector column if not exists."""
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS xpu_entries (
                id TEXT PRIMARY KEY,
                context JSONB NOT NULL,
                signals JSONB NOT NULL,
                advice_nl JSONB NOT NULL,
                atoms JSONB NOT NULL,
                embedding vector(%s) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """ % EMBEDDING_DIM)
        
        # Create index for vector similarity search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS xpu_entries_embedding_idx 
            ON xpu_entries 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        conn.commit()
        logger.info("XPU table and index created/verified")


def text_to_embedding(text: str, model: str = None) -> List[float]:
    """Generate embedding for text using OpenAI-compatible API.
    
    Configuration priority:
    1. EMBEDDING_API_KEY + EMBEDDING_BASE_URL (if set) - dedicated embedding service
    2. OPENAI_API_KEY + OPENAI_BASE_URL (if set) - fallback to OpenAI config
    3. OPENAI_API_KEY only - uses OpenAI official API
    """
    import openai
    
    # Check for embedding-specific configuration first
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY")
    embedding_base_url = os.environ.get("EMBEDDING_BASE_URL")
    embedding_model = os.environ.get("EMBEDDING_MODEL")
    
    if embedding_api_key:
        # Use embedding-specific configuration
        api_key = embedding_api_key
        base_url = embedding_base_url
        model = model or embedding_model or "text-embedding-3-small"
        logger.info(f"Using embedding API: {base_url or 'default'}, model: {model}")
    else:
        # Fall back to OpenAI configuration
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        model = model or "text-embedding-3-small"
        
        if not api_key:
            raise RuntimeError(
                "Missing API key for embedding generation. "
                "Set either EMBEDDING_API_KEY or OPENAI_API_KEY"
            )
    
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = openai.OpenAI(**client_kwargs)
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding


def build_xpu_text(entry: XpuEntry) -> str:
    """Build searchable text representation of XPU entry."""
    parts = []
    
    # Context
    ctx = entry.context
    if ctx.get("lang"):
        parts.append(f"Language: {ctx['lang']}")
    if ctx.get("tools"):
        parts.append(f"Tools: {', '.join(ctx['tools'])}")
    if ctx.get("python"):
        parts.append(f"Python versions: {', '.join(map(str, ctx['python']))}")
    if ctx.get("os"):
        parts.append(f"OS: {', '.join(ctx['os'])}")
    
    # Signals
    signals = entry.signals
    if signals.get("keywords"):
        parts.append(f"Keywords: {', '.join(signals['keywords'])}")
    if signals.get("regex"):
        parts.append(f"Error patterns: {', '.join(signals['regex'])}")
    
    # Advice
    if entry.advice_nl:
        parts.append("Advice: " + " ".join(entry.advice_nl))
    
    return "\n".join(parts)


class XpuVectorStore:
    """Vector store for XPU entries."""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or get_db_connection_string()
        self.pool = ThreadedConnectionPool(1, 5, self.connection_string)
        self._ensure_table()
    
    def _get_conn(self):
        """Get connection from pool."""
        return self.pool.getconn()
    
    def _put_conn(self, conn):
        """Return connection to pool."""
        self.pool.putconn(conn)
    
    def _ensure_table(self) -> None:
        """Ensure table exists."""
        conn = self._get_conn()
        try:
            create_xpu_table(conn)
        finally:
            self._put_conn(conn)
    
    def upsert_entry(self, entry: XpuEntry, embedding: List[float]) -> None:
        """Insert or update XPU entry with embedding."""
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}")
        
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Convert embedding list to string format for pgvector: '[0.1,0.2,...]'
                embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                
                cur.execute("""
                    INSERT INTO xpu_entries (id, context, signals, advice_nl, atoms, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        context = EXCLUDED.context,
                        signals = EXCLUDED.signals,
                        advice_nl = EXCLUDED.advice_nl,
                        atoms = EXCLUDED.atoms,
                        embedding = EXCLUDED.embedding;
                """, (
                    entry.id,
                    json.dumps(entry.context),
                    json.dumps(entry.signals),
                    json.dumps(entry.advice_nl),
                    json.dumps([{"name": a.name, "args": a.args} for a in entry.atoms]),
                    embedding_str,
                ))
                conn.commit()
        finally:
            self._put_conn(conn)
    
    def search(
        self,
        query_embedding: List[float],
        ctx: Optional[XpuContext] = None,
        k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for similar XPU entries using vector similarity."""
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(f"Query embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(query_embedding)}")
        
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Build WHERE clause for context filtering
                where_clauses = []
                where_params = []
                
                if ctx:
                    if ctx.lang:
                        where_clauses.append("context->>'lang' = %s")
                        where_params.append(ctx.lang)
                    if ctx.python:
                        # Match any Python version in the list
                        py_conditions = []
                        for py_ver in ctx.python:
                            py_conditions.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(context->'python') AS v WHERE v LIKE %s)")
                            where_params.append(f"{py_ver}%")
                        if py_conditions:
                            where_clauses.append(f"({' OR '.join(py_conditions)})")
                    if ctx.tools:
                        # Match if any tool in context matches
                        tool_conditions = []
                        for tool in ctx.tools:
                            tool_conditions.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(context->'tools') AS t WHERE t = %s)")
                            where_params.append(tool)
                        if tool_conditions:
                            where_clauses.append(f"({' OR '.join(tool_conditions)})")
                
                where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""
                
                # Convert embedding list to string format for pgvector: '[0.1,0.2,...]'
                embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
                
                # Build query with proper parameter order
                query = f"""
                    SELECT 
                        id,
                        context,
                        signals,
                        advice_nl,
                        atoms,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM xpu_entries
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    {where_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                # Parameters: query_embedding (for SELECT), query_embedding (for WHERE), min_similarity, 
                #             where_params..., query_embedding (for ORDER BY), k
                params = [embedding_str, embedding_str, min_similarity] + where_params + [embedding_str, k]
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        "id": row[0],
                        "context": row[1],
                        "signals": row[2],
                        "advice_nl": row[3],
                        "atoms": row[4],
                        "similarity": float(row[5]),
                    })
                
                return results
        finally:
            self._put_conn(conn)
    
    def get_entry(self, xpu_id: str) -> Optional[Dict[str, Any]]:
        """Get single XPU entry by ID."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, context, signals, advice_nl, atoms
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
                }
        finally:
            self._put_conn(conn)
    
    def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            self.pool.closeall()

    # 在 XpuVectorStore 类中添加这个方法
    def increment_telemetry(self, xpu_ids: List[str], field: str):
        """
        原子操作：给指定 ID 列表的 telemetry 某个字段 +1
        field 只能是 'hits', 'successes', 'failures'
        """
        if not xpu_ids: return
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # 使用 Postgres 的 jsonb_set 和 COALESCE 实现原子 +1
                sql = f"""
                    UPDATE xpu_entries 
                    SET telemetry = jsonb_set(
                        COALESCE(telemetry, '{{}}'::jsonb), 
                        '{{{field}}}', 
                        (COALESCE(telemetry->>'{field}', '0')::int + 1)::text::jsonb
                    )
                    WHERE id = ANY(%s);
                """
                cur.execute(sql, (xpu_ids,))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update telemetry ({field}): {e}")
        finally:
            self._put_conn(conn)