import time
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from prometheus_client import Histogram

from backend.rag.db import qdrant_client, get_postgres_pool, COLLECTION_NAME
from backend.rag.embeddings.embedder import embedder

logger = logging.getLogger("rip.retrieval.hybrid")

# Prometheus Metrics
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent performing hybrid search retrieval",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

class HybridSearcher:
    def __init__(self):
        pass

    def _tokenize(self, text: str) -> List[str]:
        """Simple lower-case word tokenizer."""
        return [w.strip() for w in text.lower().split() if w.strip()]

    async def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all ingested chunks from database for BM25 indexing."""
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT id, text, source_url, page_num, chunk_index FROM chunks")
            if rows:
                return [dict(r) for r in rows]
                
        # Fallback to Qdrant scrolling if postgres is empty/mocked
        try:
            records, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            return [
                {
                    "id": r.id,
                    "text": r.payload.get("text", ""),
                    "source_url": r.payload.get("source_url", ""),
                    "page_num": r.payload.get("page_num", 1),
                    "chunk_index": r.payload.get("chunk_index", 0),
                    "title": r.payload.get("title", "")
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"Error fetching fallback chunks from Qdrant: {e}")
            return []

    async def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Performs RRF-fused Hybrid Search."""
        start_time = time.time()
        
        # 1. Semantic Search
        query_vector = await embedder.embed_query(query)
        try:
            response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k * 2,  # Retrieve more for fusion
                with_payload=True
            )
            vector_results = response.points
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            vector_results = []

        # Map vector search results by ID for RRF rank mapping
        semantic_rankings = []
        for idx, res in enumerate(vector_results):
            semantic_rankings.append({
                "id": str(res.id),
                "rank": idx + 1,
                "score": res.score,
                "text": res.payload.get("text", ""),
                "source_url": res.payload.get("source_url", ""),
                "page_num": res.payload.get("page_num", 1),
                "chunk_index": res.payload.get("chunk_index", 0),
                "title": res.payload.get("title", "Document")
            })

        # 2. BM25 Local Keyword Search
        corpus_chunks = await self.get_all_chunks()
        bm25_rankings = []
        
        if corpus_chunks:
            tokenized_corpus = [self._tokenize(c["text"]) for c in corpus_chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            
            tokenized_query = self._tokenize(query)
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Rank chunks based on BM25 scores
            chunk_scores = []
            for idx, score in enumerate(bm25_scores):
                if score > 0.0:  # Only count relevant matches
                    chunk_scores.append((corpus_chunks[idx], score))
                    
            # Sort by score descending
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            for idx, (chunk, score) in enumerate(chunk_scores[:top_k * 2]):
                bm25_rankings.append({
                    "id": str(chunk["id"]),
                    "rank": idx + 1,
                    "score": score,
                    "text": chunk["text"],
                    "source_url": chunk.get("source_url", ""),
                    "page_num": chunk.get("page_num", 1),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "title": chunk.get("title", "Document")
                })

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, Dict[str, Any]] = {}
        
        # Helper to register a hit and update RRF score
        def add_rrf_hit(item: Dict[str, Any], rank: int, scorer_name: str, raw_score: float):
            doc_id = item["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "id": doc_id,
                    "text": item["text"],
                    "source_url": item["source_url"],
                    "page_num": item["page_num"],
                    "chunk_index": item["chunk_index"],
                    "title": item["title"],
                    "rrf_score": 0.0,
                    "scores": {}
                }
            rrf_scores[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank)
            rrf_scores[doc_id]["scores"][scorer_name] = raw_score

        for hit in semantic_rankings:
            add_rrf_hit(hit, hit["rank"], "semantic", hit["score"])
            
        for hit in bm25_rankings:
            add_rrf_hit(hit, hit["rank"], "bm25", hit["score"])

        # Create sorted list
        merged_results = list(rrf_scores.values())
        merged_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # Log latency metric
        latency = time.time() - start_time
        RETRIEVAL_LATENCY.observe(latency)
        logger.info(f"Hybrid search finished in {latency:.4f}s. Found {len(merged_results)} hits.")
        
        return merged_results[:top_k]

# Singleton instance
hybrid_searcher = HybridSearcher()
