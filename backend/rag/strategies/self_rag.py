import os
import json
import logging
from typing import List, Dict, Any
from google import genai
from prometheus_client import Counter

from backend.rag.retrieval.hybrid_search import hybrid_searcher

logger = logging.getLogger("rip.strategies.self_rag")

# Prometheus counter
SELF_RAG_RETRY_COUNT = Counter(
    "self_rag_retry_count",
    "Number of times Self-RAG triggered query expansion and re-retrieved documents"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IS_MOCK_ENV = os.getenv("MOCK_DB", "false").lower() == "true" or not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_")

class SelfRAGStrategy:
    def __init__(self):
        if IS_MOCK_ENV:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client for Self-RAG: {e}")
                self.client = None

    async def score_passage(self, query: str, passage: str) -> Dict[str, Any]:
        """Calls Gemini Flash to score relevance on a 1-5 scale."""
        if self.client is None:
            # Mock scoring based on some keyword heuristics
            common_words = set(query.lower().split()) & set(passage.lower().split())
            score = min(5, max(1, len(common_words) + 1))
            return {"score": score, "reason": "Mocked score based on common keywords"}

        prompt = f"""On a scale of 1-5, how relevant is this passage to answering: {query}?
Passage: {passage}
Reply with only a JSON: {{score: int, reason: str}}"""

        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            text = response.text.strip() if response.text else ""
            return json.loads(text)
        except Exception as e:
            logger.error(f"Error scoring passage with Gemini: {e}")
            # Fallback score to pass
            return {"score": 3, "reason": "Fallback score due to api error"}

    async def expand_query(self, query: str) -> str:
        """Call Gemini Flash to generate synonyms/expanded queries."""
        if self.client is None:
            return f"{query} research analysis guide"
            
        prompt = f"""Given the search query: "{query}", expand it by adding synonyms, technical terms, or broader concepts that can help find relevant documentation in a search engine. 
Reply with only the expanded query string, keeping it concise and optimized for search.

Expanded Query:"""
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text.strip() if response.text else query
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return f"{query} synonyms"

    async def search(self, query: str, top_k: int = 5, retry_count: int = 0) -> List[Dict[str, Any]]:
        """Perform Self-RAG retrieval with validation scoring and query expansion on failure."""
        logger.info(f"Self-RAG search initiated for query: '{query}' (Attempt {retry_count + 1})")
        
        # 1. Retrieve initial candidate chunks
        candidates = await hybrid_searcher.search(query, top_k=top_k * 2)
        if not candidates:
            if retry_count < 1:
                SELF_RAG_RETRY_COUNT.inc()
                expanded = await self.expand_query(query)
                return await self.search(expanded, top_k=top_k, retry_count=retry_count + 1)
            return []

        # 2. Score relevance for each chunk
        passed_chunks = []
        for c in candidates:
            scoring = await self.score_passage(query, c["text"])
            score = scoring.get("score", 1)
            reason = scoring.get("reason", "")
            
            logger.info(f"Chunk {c['id'][:8]} scored {score}/5. Reason: {reason}")
            
            if score >= 3:
                c["self_rag_score"] = score
                passed_chunks.append(c)

        # 3. If fewer than 2 chunks remain, expand query and re-retrieve
        if len(passed_chunks) < 2 and retry_count < 1:
            logger.info("Fewer than 2 chunks passed relevance threshold. Expanding query and retrying...")
            SELF_RAG_RETRY_COUNT.inc()
            expanded_query = await self.expand_query(query)
            logger.info(f"Expanded query: '{expanded_query}'")
            return await self.search(expanded_query, top_k=top_k, retry_count=retry_count + 1)

        return passed_chunks[:top_k]

# Singleton instance
self_rag_strategy = SelfRAGStrategy()
