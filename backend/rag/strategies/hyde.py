import os
import json
import hashlib
import logging
from typing import List, Dict, Any
from google import genai
from google.genai.errors import APIError

from backend.rag.db import qdrant_client, get_redis_client, COLLECTION_NAME
from backend.rag.embeddings.embedder import embedder

logger = logging.getLogger("rip.strategies.hyde")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IS_MOCK_ENV = os.getenv("MOCK_DB", "false").lower() == "true" or not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_")

class HyDEStrategy:
    def __init__(self):
        if IS_MOCK_ENV:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client for HyDE: {e}")
                self.client = None

    async def generate_hypothesis(self, query: str) -> str:
        """Call Gemini Flash to generate a hypothetical ideal answer to the query."""
        if self.client is None:
            # Mock hypothesis answer
            return f"This is a hypothetical detailed answer to explain {query} in depth, covering key concepts and references."
            
        prompt = f"""Given the user query, write a single paragraph that represents a highly detailed, hypothetical ideal answer to the query. 
Do not include metadata, preambles, or conversational filler. Write only the factual hypothesis.

Query: {query}
Hypothesis:"""
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            text = response.text.strip() if response.text else ""
            if not text:
                return f"Placeholder hypothesis for query: {query}"
            return text
        except APIError as e:
            logger.error(f"Gemini API error generating HyDE hypothesis: {e}")
            return f"Fallback hypothesis response regarding {query}."
        except Exception as e:
            logger.error(f"Error generating HyDE hypothesis: {e}")
            return f"Fallback hypothesis response regarding {query}."

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents using HyDE strategy with Redis caching for embeddings."""
        # 1. Compute hash of the query for Redis key
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        redis_key = f"hyde:vector:{query_hash}"
        
        redis_client = await get_redis_client()
        cached_vector_str = await redis_client.get(redis_key)
        
        vector = None
        if cached_vector_str:
            try:
                vector = json.loads(cached_vector_str)
                logger.info("Found cached HyDE embedding in Redis.")
            except Exception as e:
                logger.warning(f"Error parsing cached HyDE vector: {e}")
                vector = None
                
        if vector is None:
            # 2. Generate hypothesis
            hypothesis = await self.generate_hypothesis(query)
            logger.info(f"Generated hypothesis for HyDE: {hypothesis[:100]}...")
            
            # 3. Embed hypothesis
            vector = await embedder.embed_query(hypothesis)
            
            # 4. Cache in Redis
            try:
                await redis_client.set(redis_key, json.dumps(vector), ex=3600)
                logger.info("Cached new HyDE embedding in Redis (TTL=3600s).")
            except Exception as e:
                logger.warning(f"Failed to cache vector in Redis: {e}")

        # 5. Search Qdrant using hypothesis embedding
        try:
            response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=top_k,
                with_payload=True
            )
            results = response.points
        except Exception as e:
            logger.error(f"Qdrant search error in HyDE: {e}")
            return []

        formatted_results = []
        for res in results:
            formatted_results.append({
                "id": str(res.id),
                "text": res.payload.get("text", ""),
                "source_url": res.payload.get("source_url", ""),
                "page_num": res.payload.get("page_num", 1),
                "chunk_index": res.payload.get("chunk_index", 0),
                "title": res.payload.get("title", "Document"),
                "score": res.score
            })
            
        return formatted_results

# Singleton instance
hyde_strategy = HyDEStrategy()
