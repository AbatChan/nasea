"""
Context Memory - Manages conversation history and code context.
Implements retrieval-augmented generation (RAG) for large codebases.

Features:
- SQLite-based persistent storage
- Semantic code search using embeddings
- Code caching for reuse
- Task history tracking
"""

import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from loguru import logger

# Optional embedding support
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.debug("NumPy not available - semantic search will use text matching")


@dataclass
class ContextEntry:
    """A single entry in the context memory."""
    id: Optional[int]
    timestamp: str
    role: str  # 'user', 'manager', 'developer', 'verifier', 'system'
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class ContextMemory:
    """
    Manages conversation history and code context across agents.
    Provides persistent memory beyond LLM context window.
    """

    def __init__(self, database_path: Path, embed_fn: Optional[Callable[[str], List[float]]] = None):
        """
        Initialize context memory with SQLite backend.

        Args:
            database_path: Path to SQLite database file
            embed_fn: Optional embedding function that takes text and returns a vector.
                     If not provided, semantic search falls back to text matching.
        """
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Embedding function for semantic search
        self._embed_fn = embed_fn
        self._embedding_cache: Dict[str, List[float]] = {}

        # Connect to database
        self.conn = sqlite3.connect(str(self.database_path))
        self.cursor = self.conn.cursor()

        # Initialize tables
        self._init_database()
        logger.info(f"ContextMemory initialized: {self.database_path}")

    def _init_database(self):
        """Create database tables if they don't exist."""
        # Context entries table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB
            )
        """)

        # Task history table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                result TEXT,
                error TEXT
            )
        """)

        # Code snippets cache (with embedding for semantic search)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash TEXT UNIQUE NOT NULL,
                prompt TEXT NOT NULL,
                generated_code TEXT NOT NULL,
                language TEXT,
                created_at TEXT NOT NULL,
                quality_score REAL,
                reuse_count INTEGER DEFAULT 0,
                embedding TEXT
            )
        """)

        # Add embedding column if it doesn't exist (migration for existing DBs)
        try:
            self.cursor.execute("ALTER TABLE code_cache ADD COLUMN embedding TEXT")
            logger.debug("Added embedding column to code_cache table")
        except sqlite3.OperationalError:
            pass  # Column already exists

        self.conn.commit()

    def add_entry(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a new entry to context memory.

        Args:
            role: Role of the entity creating this entry
            content: Content of the entry
            metadata: Additional metadata

        Returns:
            ID of the created entry
        """
        entry = ContextEntry(
            id=None,
            timestamp=datetime.now().isoformat(),
            role=role,
            content=content,
            metadata=metadata or {}
        )

        self.cursor.execute("""
            INSERT INTO context_entries (timestamp, role, content, metadata)
            VALUES (?, ?, ?, ?)
        """, (entry.timestamp, entry.role, entry.content, json.dumps(entry.metadata)))

        self.conn.commit()
        entry_id = self.cursor.lastrowid

        logger.debug(f"Added context entry: {role} ({len(content)} chars)")
        return entry_id

    def get_recent_entries(
        self,
        limit: int = 10,
        role: Optional[str] = None
    ) -> List[ContextEntry]:
        """
        Get recent context entries.

        Args:
            limit: Maximum number of entries to return
            role: Filter by role (optional)

        Returns:
            List of context entries
        """
        if role:
            self.cursor.execute("""
                SELECT id, timestamp, role, content, metadata
                FROM context_entries
                WHERE role = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (role, limit))
        else:
            self.cursor.execute("""
                SELECT id, timestamp, role, content, metadata
                FROM context_entries
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        entries = []
        for row in self.cursor.fetchall():
            entries.append(ContextEntry(
                id=row[0],
                timestamp=row[1],
                role=row[2],
                content=row[3],
                metadata=json.loads(row[4]) if row[4] else {}
            ))

        return list(reversed(entries))  # Return in chronological order

    def get_conversation_history(self, limit: int = 20) -> str:
        """
        Get formatted conversation history for LLM context.

        Args:
            limit: Maximum number of entries

        Returns:
            Formatted conversation history
        """
        entries = self.get_recent_entries(limit=limit)

        history = []
        for entry in entries:
            history.append(f"[{entry.role.upper()}] {entry.content}")

        return "\n\n".join(history)

    def search_entries(
        self,
        query: str,
        limit: int = 5
    ) -> List[ContextEntry]:
        """
        Search entries by content (simple text search for MVP).

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Matching entries
        """
        self.cursor.execute("""
            SELECT id, timestamp, role, content, metadata
            FROM context_entries
            WHERE content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{query}%", limit))

        entries = []
        for row in self.cursor.fetchall():
            entries.append(ContextEntry(
                id=row[0],
                timestamp=row[1],
                role=row[2],
                content=row[3],
                metadata=json.loads(row[4]) if row[4] else {}
            ))

        return entries

    # -------------------------------------------------------------------------
    # Semantic Code Search
    # -------------------------------------------------------------------------

    def set_embedding_function(self, embed_fn: Callable[[str], List[float]]):
        """
        Set or update the embedding function for semantic search.

        Args:
            embed_fn: Function that takes text and returns embedding vector
        """
        self._embed_fn = embed_fn
        self._embedding_cache.clear()
        logger.info("Embedding function configured for semantic search")

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text, using cache if available.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if no embed function
        """
        if not self._embed_fn:
            return None

        # Use hash as cache key (text can be very long)
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            embedding = self._embed_fn(text)
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between -1 and 1
        """
        if not NUMPY_AVAILABLE:
            # Fallback: simple dot product (assumes normalized vectors)
            return sum(a * b for a, b in zip(vec1, vec2))

        arr1 = np.array(vec1)
        arr2 = np.array(vec2)

        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(arr1, arr2) / (norm1 * norm2))

    def search_similar_code(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.5
    ) -> List[Tuple[str, str, str, float]]:
        """
        Search for similar code using semantic embeddings.

        Falls back to text search if embeddings are not available.

        Args:
            query: Natural language query or code snippet
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of tuples: (prompt, code, language, similarity_score)
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)

        if query_embedding is None:
            # Fallback to text search
            logger.debug("No embedding function - using text search fallback")
            return self._text_search_code(query, limit)

        # Get all cached code with embeddings
        self.cursor.execute("""
            SELECT prompt, generated_code, language, embedding
            FROM code_cache
            WHERE embedding IS NOT NULL
        """)

        results = []
        for row in self.cursor.fetchall():
            prompt, code, language, embedding_blob = row

            if embedding_blob:
                try:
                    cached_embedding = json.loads(embedding_blob)
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)

                    if similarity >= min_similarity:
                        results.append((prompt, code, language, similarity))
                except (json.JSONDecodeError, TypeError):
                    continue

        # Sort by similarity (highest first) and limit
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:limit]

    def _text_search_code(
        self,
        query: str,
        limit: int = 5
    ) -> List[Tuple[str, str, str, float]]:
        """
        Fallback text search for code cache.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of tuples: (prompt, code, language, relevance_score)
        """
        # Search in prompts and code
        keywords = query.lower().split()

        self.cursor.execute("""
            SELECT prompt, generated_code, language
            FROM code_cache
            ORDER BY reuse_count DESC, created_at DESC
        """)

        results = []
        for row in self.cursor.fetchall():
            prompt, code, language = row
            text = (prompt + " " + code).lower()

            # Simple relevance: count keyword matches
            matches = sum(1 for kw in keywords if kw in text)
            if matches > 0:
                # Normalize to 0-1 range
                relevance = min(1.0, matches / len(keywords))
                results.append((prompt, code, language, relevance))

        # Sort by relevance and limit
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:limit]

    def cache_code_with_embedding(
        self,
        prompt: str,
        code: str,
        language: str,
        quality_score: Optional[float] = None
    ):
        """
        Cache generated code with its embedding for semantic search.

        Args:
            prompt: Original prompt
            code: Generated code
            language: Programming language
            quality_score: Quality score (0-1)
        """
        # Generate embedding for prompt + code summary
        embedding_text = f"{prompt}\n\n{code[:500]}"  # Limit code for embedding
        embedding = self._get_embedding(embedding_text)
        embedding_blob = json.dumps(embedding) if embedding else None

        # Use stable hash
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        try:
            self.cursor.execute("""
                INSERT INTO code_cache (prompt_hash, prompt, generated_code, language, created_at, quality_score, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (prompt_hash, prompt, code, language, datetime.now().isoformat(), quality_score, embedding_blob))
            self.conn.commit()
            logger.debug(f"Cached code with embedding for: {prompt[:50]}...")
        except sqlite3.IntegrityError:
            # Update existing entry
            self.cursor.execute("""
                UPDATE code_cache
                SET reuse_count = reuse_count + 1,
                    embedding = COALESCE(?, embedding)
                WHERE prompt_hash = ?
            """, (embedding_blob, prompt_hash))
            self.conn.commit()

    def find_reusable_code(
        self,
        task_description: str,
        language: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find potentially reusable code snippets for a given task.

        This is the main entry point for semantic code search, designed
        to reduce LLM calls by finding similar past code.

        Args:
            task_description: Description of what needs to be built
            language: Optional language filter
            limit: Maximum results

        Returns:
            List of dicts with: prompt, code, language, similarity, reuse_count
        """
        # Search semantically
        similar = self.search_similar_code(task_description, limit=limit * 2)

        # Filter by language if specified
        if language:
            similar = [s for s in similar if s[2] == language]

        results = []
        for prompt, code, lang, similarity in similar[:limit]:
            # Get reuse count
            self.cursor.execute("""
                SELECT reuse_count FROM code_cache
                WHERE prompt = ?
            """, (prompt,))
            row = self.cursor.fetchone()
            reuse_count = row[0] if row else 0

            results.append({
                "prompt": prompt,
                "code": code,
                "language": lang,
                "similarity": similarity,
                "reuse_count": reuse_count
            })

        return results

    def add_task(
        self,
        task_id: str,
        description: str,
        status: str = "pending"
    ):
        """
        Add a task to history.

        Args:
            task_id: Unique task identifier
            description: Task description
            status: Current status
        """
        try:
            self.cursor.execute("""
                INSERT INTO task_history (task_id, description, status, created_at)
                VALUES (?, ?, ?, ?)
            """, (task_id, description, status, datetime.now().isoformat()))
            self.conn.commit()
            logger.debug(f"Added task: {task_id}")
        except sqlite3.IntegrityError:
            logger.warning(f"Task already exists: {task_id}")

    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        result: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Update task status.

        Args:
            task_id: Task identifier
            status: New status (optional)
            result: Task result (optional)
            error: Error message (optional)
        """
        updates = []
        values = []

        if status:
            updates.append("status = ?")
            values.append(status)

        if result:
            updates.append("result = ?")
            values.append(result)

        if error:
            updates.append("error = ?")
            values.append(error)

        if status in ["completed", "failed"]:
            updates.append("completed_at = ?")
            values.append(datetime.now().isoformat())

        if not updates:
            return

        values.append(task_id)
        query = f"UPDATE task_history SET {', '.join(updates)} WHERE task_id = ?"

        self.cursor.execute(query, tuple(values))
        self.conn.commit()
        logger.debug(f"Updated task: {task_id} -> {status}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status.

        Args:
            task_id: Task identifier

        Returns:
            Task information or None
        """
        self.cursor.execute("""
            SELECT task_id, description, status, created_at, completed_at, result, error
            FROM task_history
            WHERE task_id = ?
        """, (task_id,))

        row = self.cursor.fetchone()
        if not row:
            return None

        return {
            "task_id": row[0],
            "description": row[1],
            "status": row[2],
            "created_at": row[3],
            "completed_at": row[4],
            "result": row[5],
            "error": row[6]
        }

    def cache_code(
        self,
        prompt: str,
        code: str,
        language: str,
        quality_score: Optional[float] = None
    ):
        """
        Cache generated code for reuse.

        Args:
            prompt: Original prompt
            code: Generated code
            language: Programming language
            quality_score: Quality score (0-1)
        """
        # Simple hash of prompt (could use better hashing in production)
        prompt_hash = str(hash(prompt))

        try:
            self.cursor.execute("""
                INSERT INTO code_cache (prompt_hash, prompt, generated_code, language, created_at, quality_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (prompt_hash, prompt, code, language, datetime.now().isoformat(), quality_score))
            self.conn.commit()
            logger.debug(f"Cached code for prompt: {prompt[:50]}...")
        except sqlite3.IntegrityError:
            # Update reuse count if already exists
            self.cursor.execute("""
                UPDATE code_cache
                SET reuse_count = reuse_count + 1
                WHERE prompt_hash = ?
            """, (prompt_hash,))
            self.conn.commit()

    def get_cached_code(self, prompt: str) -> Optional[Tuple[str, str]]:
        """
        Retrieve cached code if available.

        Args:
            prompt: Original prompt

        Returns:
            Tuple of (code, language) or None
        """
        prompt_hash = str(hash(prompt))

        self.cursor.execute("""
            SELECT generated_code, language
            FROM code_cache
            WHERE prompt_hash = ?
        """, (prompt_hash,))

        row = self.cursor.fetchone()
        if row:
            # Increment reuse count
            self.cursor.execute("""
                UPDATE code_cache
                SET reuse_count = reuse_count + 1
                WHERE prompt_hash = ?
            """, (prompt_hash,))
            self.conn.commit()

            logger.info(f"Cache hit for prompt: {prompt[:50]}...")
            return (row[0], row[1])

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with statistics
        """
        # Total entries
        self.cursor.execute("SELECT COUNT(*) FROM context_entries")
        total_entries = self.cursor.fetchone()[0]

        # Entries by role
        self.cursor.execute("""
            SELECT role, COUNT(*) FROM context_entries GROUP BY role
        """)
        by_role = dict(self.cursor.fetchall())

        # Task statistics
        self.cursor.execute("""
            SELECT status, COUNT(*) FROM task_history GROUP BY status
        """)
        tasks_by_status = dict(self.cursor.fetchall())

        # Cache statistics
        self.cursor.execute("SELECT COUNT(*), SUM(reuse_count) FROM code_cache")
        cache_row = self.cursor.fetchone()
        cached_prompts = cache_row[0]
        cache_hits = cache_row[1] or 0

        return {
            "total_entries": total_entries,
            "by_role": by_role,
            "tasks_by_status": tasks_by_status,
            "cached_prompts": cached_prompts,
            "cache_hits": cache_hits
        }

    def clear_old_entries(self, days: int = 7):
        """
        Clear entries older than specified days.

        Args:
            days: Number of days to keep
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()

        self.cursor.execute("""
            DELETE FROM context_entries
            WHERE timestamp < ?
        """, (cutoff_iso,))

        deleted = self.cursor.rowcount
        self.conn.commit()
        logger.info(f"Cleared {deleted} old entries")

    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.debug("ContextMemory closed")

    def __enter__(self):
        """Context manager entry - return self for use in 'with' statements."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure connection is closed."""
        self.close()
        # Don't suppress exceptions - return None/False
        return False

    def __del__(self):
        """Destructor - ensure connection is closed when object is garbage collected."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return f"ContextMemory(entries={stats['total_entries']}, tasks={sum(stats['tasks_by_status'].values())})"
