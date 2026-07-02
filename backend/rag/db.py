import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import redis.asyncio as redis
from neo4j import GraphDatabase

logger = logging.getLogger("rip.db")

# Read configurations from environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "research_metadata")

# ----------------- QDRANT CLIENT -----------------
def get_qdrant_client() -> QdrantClient:
    """Gets Qdrant client, falls back to memory mode if connection fails."""
    try:
        if QDRANT_HOST == "mock" or os.getenv("MOCK_DB", "false").lower() == "true":
            logger.info("Initializing Qdrant client in memory-mode (Mock)")
            return QdrantClient(":memory:")
        
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2.0)
        # Quick healthcheck
        client.get_collections()
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return client
    except Exception as e:
        logger.warning(f"Failed to connect to Qdrant ({e}). Falling back to memory mode.")
        return QdrantClient(":memory:")

# Initialize a default Qdrant Client
qdrant_client = get_qdrant_client()

# Ensure RAG collection exists in Qdrant
COLLECTION_NAME = "research_documents"
try:
    collections = qdrant_client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    if not exists:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=768,  # text-embedding-004 output size
                distance=qmodels.Distance.COSINE
            )
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Error initializing Qdrant collection: {e}")

# ----------------- REDIS CACHE -----------------
class MockRedis:
    """In-memory Redis mock for fallback."""
    def __init__(self):
        self.store = {}
    
    async def get(self, key: str) -> Optional[str]:
        return self.store.get(key)
        
    async def set(self, key: str, value: str, ex: int = None) -> None:
        self.store[key] = value
        if ex:
            # Simple TTL simulation in a background task
            async def expire():
                await asyncio.sleep(ex)
                self.store.pop(key, None)
            asyncio.create_task(expire())
            
    async def ping(self) -> bool:
        return True

async def get_redis_client():
    """Gets Redis client, falls back to MockRedis if connection fails."""
    if os.getenv("MOCK_DB", "false").lower() == "true":
        return MockRedis()
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_connect_timeout=2.0)
        await client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return client
    except Exception as e:
        logger.warning(f"Failed to connect to Redis ({e}). Falling back to MockRedis.")
        return MockRedis()

# ----------------- NEO4J GRAPH DB -----------------
class MockNeo4jSession:
    def __init__(self):
        self.nodes = {}
        self.relationships = []

    def run(self, query: str, **kwargs) -> Any:
        # Simple parser for mock queries
        query = query.lower()
        if "merge" in query:
            # We mock the return of node creation
            return []
        elif "match" in query:
            # Mock graph search output
            entity = kwargs.get("entity", "unknown")
            return [
                {
                    "path": [
                        {"name": entity, "type": "Entity"},
                        {"type": "RELATED_TO"},
                        {"name": "Artificial Intelligence", "type": "Technology"}
                    ]
                }
            ]
        return []

class MockNeo4jDriver:
    def session(self, **kwargs):
        return MockNeo4jSession()
    
    def close(self):
        pass

def get_neo4j_driver() -> Any:
    """Gets Neo4j driver, falls back to MockNeo4jDriver if connection fails."""
    if os.getenv("MOCK_DB", "false").lower() == "true":
        return MockNeo4jDriver()
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {NEO4J_URI}")
        return driver
    except Exception as e:
        logger.warning(f"Failed to connect to Neo4j ({e}). Falling back to MockNeo4jDriver.")
        return MockNeo4jDriver()

# ----------------- POSTGRESQL METADATA -----------------
class MockPostgresConn:
    """Mock connection for PostgreSQL."""
    def __init__(self):
        self.tables = {
            "documents": [],
            "chunks": [],
            "answer_evaluations": []
        }

    async def execute(self, query: str, *args) -> str:
        return "COMMAND OK"

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        # Returns empty or mocked queries
        if "answer_evaluations" in query:
            return [
                {
                    "session_id": "mock-session",
                    "query": "What is RAG?",
                    "answer": "RAG stands for Retrieval-Augmented Generation.",
                    "faithfulness": 0.9,
                    "relevancy": 0.85,
                    "recall": 0.8,
                    "overall": 0.85,
                    "timestamp": "2026-06-18 12:00:00"
                }
            ]
        return []

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        return None

class MockPostgresPool:
    async def acquire(self):
        return MockPostgresConn()
    
    async def release(self, conn):
        pass
        
    async def close(self):
        pass

async def get_postgres_pool() -> Any:
    """Gets PostgreSQL connection pool, falls back to MockPostgresPool."""
    if os.getenv("MOCK_DB", "false").lower() == "true":
        return MockPostgresPool()
    import asyncpg
    try:
        pool = await asyncpg.create_pool(
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            timeout=2.0
        )
        logger.info(f"Connected to PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}")
        # Initialize tables
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    title TEXT,
                    source_url TEXT,
                    source_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    text TEXT,
                    page_num INT,
                    chunk_index INT,
                    source_url TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS answer_evaluations (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    query TEXT,
                    answer TEXT,
                    faithfulness FLOAT,
                    relevancy FLOAT,
                    recall FLOAT,
                    overall FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        return pool
    except Exception as e:
        logger.warning(f"Failed to connect to PostgreSQL ({e}). Falling back to MockPostgresPool.")
        return MockPostgresPool()
