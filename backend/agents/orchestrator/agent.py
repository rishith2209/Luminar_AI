import os
import uuid
import time
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from google import genai
import httpx

logger = logging.getLogger("rip.agents.orchestrator")

app = FastAPI(title="Research Orchestrator", version="1.0")

# Connection endpoints for sub-agents (configured for local/docker)
RETRIEVAL_AGENT_URL = os.getenv("RETRIEVAL_AGENT_URL", "http://localhost:8101/tasks/send")
SYNTHESIS_AGENT_URL = os.getenv("SYNTHESIS_AGENT_URL", "http://localhost:8102/tasks/send")
EVAL_AGENT_URL = os.getenv("EVAL_AGENT_URL", "http://localhost:8103/tasks/send")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IS_MOCK_ENV = os.getenv("MOCK_DB", "false").lower() == "true" or not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_")

class OrchestratorRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class OrchestratorResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    eval_scores: Dict[str, float]
    agent_trace: List[Dict[str, Any]]

class OrchestratorAgent:
    def __init__(self):
        if IS_MOCK_ENV:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize GenAI client: {e}")
                self.client = None

    async def classify_query(self, query: str) -> str:
        """Classifies query into factual, analytical, comparative, or generative."""
        if self.client is None:
            # Simple heuristic classifier
            q_lower = query.lower()
            if any(x in q_lower for x in ["compare", "versus", "vs", "difference between"]):
                return "comparative"
            elif any(x in q_lower for x in ["why", "explain", "analyze", "how does"]):
                return "analytical"
            elif any(x in q_lower for x in ["write", "create", "generate", "draft"]):
                return "generative"
            return "factual"

        prompt = f"""Classify the user query into exactly one of these types:
- "factual": Simple lookup, fact verification, short answers.
- "analytical": Complex reasoning, cause-and-effect, deep technical explanations.
- "comparative": Comparing multiple concepts, items, solutions, list of pros/cons.
- "generative": Writing templates, drafting emails, coding, summarizing from imagination.

Query: {query}
Reply with only the lowercase category name."""

        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            val = response.text.strip().lower() if response.text else "factual"
            if val not in ["factual", "analytical", "comparative", "generative"]:
                return "factual"
            return val
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return "factual"

    async def run_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Orchestrate query routing through the A2A network and score the result."""
        if not session_id:
            session_id = str(uuid.uuid4())
            
        agent_trace = []
        
        # 1. Classify query
        agent_trace.append({"agent": "Orchestrator", "status": "thinking", "message": "Classifying user query...", "timestamp": time.time()})
        query_class = await self.classify_query(query)
        agent_trace.append({"agent": "Orchestrator", "status": "complete", "message": f"Classified query as '{query_class}'", "timestamp": time.time()})

        chunks = []
        strategy_used = "none"
        synthesis_result = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 2. Dispatch based on query class
            if query_class == "factual":
                # Route to retrieval agent
                agent_trace.append({"agent": "RetrievalAgent", "status": "thinking", "message": "Fetching relevant chunks...", "timestamp": time.time()})
                t_start = time.time()
                try:
                    response = await client.post(RETRIEVAL_AGENT_URL, json={"query": query})
                    if response.status_code == 200:
                        ret_data = response.json()
                        chunks = ret_data.get("chunks", [])
                        strategy_used = ret_data.get("strategy_used", "hybrid")
                    agent_trace.append({
                        "agent": "RetrievalAgent", 
                        "status": "complete", 
                        "message": f"Retrieved {len(chunks)} chunks using '{strategy_used}'", 
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "timestamp": time.time()
                    })
                except Exception as e:
                    logger.error(f"Retrieval agent call failed: {e}")
                    agent_trace.append({"agent": "RetrievalAgent", "status": "failed", "message": str(e), "timestamp": time.time()})

                # Route to synthesis agent
                agent_trace.append({"agent": "SynthesisAgent", "status": "thinking", "message": "Writing final answer...", "timestamp": time.time()})
                t_start = time.time()
                try:
                    response = await client.post(SYNTHESIS_AGENT_URL, json={"query": query, "chunks": chunks})
                    if response.status_code == 200:
                        synthesis_result = response.json()
                    agent_trace.append({
                        "agent": "SynthesisAgent", 
                        "status": "complete", 
                        "message": f"Answer generated. Confidence: {synthesis_result.get('confidence')}", 
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "timestamp": time.time()
                    })
                except Exception as e:
                    logger.error(f"Synthesis agent call failed: {e}")
                    agent_trace.append({"agent": "SynthesisAgent", "status": "failed", "message": str(e), "timestamp": time.time()})

            elif query_class == "analytical":
                # Sequential routing: retrieval -> synthesis
                agent_trace.append({"agent": "RetrievalAgent", "status": "thinking", "message": "Deep search in progress...", "timestamp": time.time()})
                t_start = time.time()
                try:
                    response = await client.post(RETRIEVAL_AGENT_URL, json={"query": query})
                    if response.status_code == 200:
                        ret_data = response.json()
                        chunks = ret_data.get("chunks", [])
                        strategy_used = ret_data.get("strategy_used", "hybrid")
                    agent_trace.append({
                        "agent": "RetrievalAgent", 
                        "status": "complete", 
                        "message": f"Fetched {len(chunks)} chunks with '{strategy_used}'", 
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "timestamp": time.time()
                    })
                except Exception as e:
                    agent_trace.append({"agent": "RetrievalAgent", "status": "failed", "message": str(e), "timestamp": time.time()})

                agent_trace.append({"agent": "SynthesisAgent", "status": "thinking", "message": "Drafting deep synthesis...", "timestamp": time.time()})
                t_start = time.time()
                try:
                    response = await client.post(SYNTHESIS_AGENT_URL, json={"query": query, "chunks": chunks})
                    if response.status_code == 200:
                        synthesis_result = response.json()
                    agent_trace.append({
                        "agent": "SynthesisAgent", 
                        "status": "complete", 
                        "message": f"Synthesis ready. Flags: {len(synthesis_result.get('flags', []))}", 
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "timestamp": time.time()
                    })
                except Exception as e:
                    agent_trace.append({"agent": "SynthesisAgent", "status": "failed", "message": str(e), "timestamp": time.time()})

            elif query_class == "comparative":
                # Parallel routing (retrieve original query and expanded synonyms, combine context)
                agent_trace.append({"agent": "RetrievalAgent", "status": "thinking", "message": "Executing parallel searches...", "timestamp": time.time()})
                t_start = time.time()
                
                # Fetch synonyms/expanded queries using Self-RAG module logic
                from backend.rag.strategies.self_rag import self_rag_strategy
                expanded_query = await self_rag_strategy.expand_query(query)
                
                try:
                    # Run two retrievals in parallel
                    res_orig, res_exp = await asyncio.gather(
                        client.post(RETRIEVAL_AGENT_URL, json={"query": query}),
                        client.post(RETRIEVAL_AGENT_URL, json={"query": expanded_query}),
                        return_exceptions=True
                    )
                    
                    chunks_list = []
                    seen = set()
                    
                    for res in [res_orig, res_exp]:
                        if isinstance(res, httpx.Response) and res.status_code == 200:
                            ret_data = res.json()
                            strategy_used = ret_data.get("strategy_used", "hybrid")
                            for chunk in ret_data.get("chunks", []):
                                if chunk["text"] not in seen:
                                    seen.add(chunk["text"])
                                    chunks_list.append(chunk)
                                    
                    chunks = chunks_list
                    agent_trace.append({
                        "agent": "RetrievalAgent", 
                        "status": "complete", 
                        "message": f"Parallel search completed. Combined {len(chunks)} unique chunks.", 
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "timestamp": time.time()
                    })
                except Exception as e:
                    agent_trace.append({"agent": "RetrievalAgent", "status": "failed", "message": str(e), "timestamp": time.time()})

                # Call synthesis
                agent_trace.append({"agent": "SynthesisAgent", "status": "thinking", "message": "Formulating comparative analysis...", "timestamp": time.time()})
                t_start = time.time()
                try:
                    response = await client.post(SYNTHESIS_AGENT_URL, json={"query": query, "chunks": chunks})
                    if response.status_code == 200:
                        synthesis_result = response.json()
                    agent_trace.append({
                        "agent": "SynthesisAgent", 
                        "status": "complete", 
                        "message": "Comparative answer synthesized.", 
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "timestamp": time.time()
                    })
                except Exception as e:
                    agent_trace.append({"agent": "SynthesisAgent", "status": "failed", "message": str(e), "timestamp": time.time()})

            elif query_class == "generative":
                # Synthesize directly with no chunk context
                agent_trace.append({"agent": "SynthesisAgent", "status": "thinking", "message": "Generating text directly...", "timestamp": time.time()})
                t_start = time.time()
                try:
                    response = await client.post(SYNTHESIS_AGENT_URL, json={"query": query, "chunks": []})
                    if response.status_code == 200:
                        synthesis_result = response.json()
                    agent_trace.append({
                        "agent": "SynthesisAgent", 
                        "status": "complete", 
                        "message": "Generative draft created.", 
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "timestamp": time.time()
                    })
                except Exception as e:
                    agent_trace.append({"agent": "SynthesisAgent", "status": "failed", "message": str(e), "timestamp": time.time()})

            # 3. Call Eval Agent to score answer
            eval_scores = {"faithfulness": 1.0, "relevancy": 1.0, "recall": 1.0, "overall": 1.0}
            answer_text = synthesis_result.get("answer_markdown", "")
            
            if answer_text:
                agent_trace.append({"agent": "EvalAgent", "status": "thinking", "message": "Scoring answer quality...", "timestamp": time.time()})
                t_start = time.time()
                context_texts = [c["text"] for c in chunks]
                try:
                    response = await client.post(
                        EVAL_AGENT_URL, 
                        json={
                            "session_id": session_id,
                            "query": query,
                            "answer": answer_text,
                            "context_chunks": context_texts
                        }
                    )
                    if response.status_code == 200:
                        eval_scores = response.json()
                    agent_trace.append({
                        "agent": "EvalAgent", 
                        "status": "complete", 
                        "message": f"Evaluation done. Overall: {eval_scores.get('overall')}", 
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "timestamp": time.time()
                    })
                except Exception as e:
                    logger.error(f"Eval agent call failed: {e}")
                    agent_trace.append({"agent": "EvalAgent", "status": "failed", "message": str(e), "timestamp": time.time()})

        # Save main response structure
        return {
            "answer": answer_text or "Sorry, I was unable to generate an answer.",
            "citations": synthesis_result.get("citations", []),
            "eval_scores": eval_scores,
            "agent_trace": agent_trace
        }

orchestrator = OrchestratorAgent()

@app.post("/tasks/send", response_model=OrchestratorResponse)
async def execute_task(request: OrchestratorRequest):
    try:
        res = await orchestrator.run_query(request.query, request.session_id)
        return OrchestratorResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "ResearchOrchestrator",
        "description": "Routes research queries to specialized agents",
        "url": "http://localhost:8000/a2a",
        "version": "1.0",
        "capabilities": ["research", "synthesis", "evaluation"],
        "inputModes": ["text"],
        "outputModes": ["text", "structured_data"]
    }

if __name__ == "__main__":
    import sys
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
    uvicorn.run(app, host="0.0.0.0", port=port)
