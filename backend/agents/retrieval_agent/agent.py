import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
import httpx

from backend.rag.retrieval.hybrid_search import hybrid_searcher
from backend.rag.strategies.hyde import hyde_strategy
from backend.rag.strategies.self_rag import self_rag_strategy
from backend.rag.strategies.graph_rag import graph_rag_strategy
from backend.rag.retrieval.compressor import compressor

logger = logging.getLogger("rip.agents.retrieval")

app = FastAPI(title="Retrieval Agent", version="1.0")

WEB_AGENT_URL = os.getenv("WEB_AGENT_URL", "http://localhost:8104/tasks/send")

class RetrievalRequest(BaseModel):
    query: str

class RetrievalResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    strategy_used: str

class RetrievalAgent:
    def __init__(self):
        pass

    async def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        # 1. Classify query to select strategy
        entities = await graph_rag_strategy.extract_query_entities(query)
        strategy = "hybrid"
        
        if len(query) < 40:
            strategy = "hyde"
            logger.info("Query classified as short factual. Using HyDE strategy.")
            chunks = await hyde_strategy.search(query, top_k=5)
        elif len(entities) >= 2:
            strategy = "graph_rag"
            logger.info(f"Query contains multiple entities ({entities}). Using Graph RAG + Hybrid strategy.")
            # Combine graph RAG chunks with hybrid baseline
            graph_chunks = await graph_rag_strategy.search(query)
            hybrid_chunks = await hybrid_searcher.search(query, top_k=3)
            chunks = graph_chunks + hybrid_chunks
        else:
            strategy = "hybrid"
            logger.info("Using baseline Hybrid search strategy.")
            chunks = await hybrid_searcher.search(query, top_k=5)

        # 2. Self-RAG quality gate (filter chunks scoring < 3)
        validated_chunks = []
        for c in chunks:
            scoring = await self_rag_strategy.score_passage(query, c["text"])
            score = scoring.get("score", 3)
            if score >= 3:
                c["relevance_score"] = score / 5.0
                validated_chunks.append(c)

        # 3. Call Web Agent fallback if validation filters out too many chunks (< 3 chunks)
        if len(validated_chunks) < 3:
            logger.info(f"Only {len(validated_chunks)} chunks passed the quality gate. Invoking Web Agent for live search...")
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(WEB_AGENT_URL, json={"query": query})
                    if response.status_code == 200:
                        web_data = response.json()
                        web_chunks = web_data.get("chunks", [])
                        logger.info(f"Web Agent returned {len(web_chunks)} live web chunks.")
                        # Validate web chunks as well
                        for wc in web_chunks:
                            scoring = await self_rag_strategy.score_passage(query, wc["text"])
                            if scoring.get("score", 3) >= 3:
                                wc["relevance_score"] = scoring.get("score", 3) / 5.0
                                validated_chunks.append(wc)
            except Exception as e:
                logger.error(f"Failed to call Web Agent: {e}")

        # 4. Contextual Compression
        compressed_chunks = await compressor.compress(query, validated_chunks)
        
        # Format scores/metadata
        for c in compressed_chunks:
            if "score" not in c:
                c["score"] = c.get("relevance_score", 0.7)
            if "graph_source" not in c:
                c["graph_source"] = False
            if "live_web" not in c:
                c["live_web"] = False

        return compressed_chunks, strategy

retrieval_agent = RetrievalAgent()

@app.post("/tasks/send", response_model=RetrievalResponse)
async def execute_task(request: RetrievalRequest):
    try:
        chunks, strategy = await retrieval_agent.retrieve(request.query)
        return RetrievalResponse(chunks=chunks, strategy_used=strategy)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "RetrievalAgent",
        "description": "Orchestrates document search strategies and filters context",
        "version": "1.0",
        "capabilities": ["hybrid_search", "hyde", "graph_rag", "compression"],
        "inputModes": ["text"],
        "outputModes": ["structured_data"]
    }

if __name__ == "__main__":
    import sys
    # Import Tuple
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8101
    uvicorn.run(app, host="0.0.0.0", port=port)
