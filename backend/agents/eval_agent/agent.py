import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from backend.evaluation.ragas_eval import evaluator
from backend.rag.db import get_postgres_pool

logger = logging.getLogger("rip.agents.eval")

app = FastAPI(title="Eval Agent", version="1.0")

class EvalRequest(BaseModel):
    session_id: str
    query: str
    answer: str
    context_chunks: List[str]
    ground_truth: Optional[str] = None

class EvalResponse(BaseModel):
    faithfulness: float
    relevancy: float
    recall: float
    overall: float

class EvalAgent:
    def __init__(self):
        pass

    async def evaluate(self, session_id: str, query: str, answer: str, context_chunks: List[str], ground_truth: str = None) -> Dict[str, float]:
        # 1. Compute faithfulness
        faithfulness = await evaluator.score_faithfulness(answer, context_chunks)
        
        # 2. Compute relevancy
        relevancy = await evaluator.score_answer_relevancy(query, answer)
        
        # 3. Compute recall (if ground truth is absent, we use context precision as recall estimate)
        if ground_truth:
            recall = await evaluator.score_context_recall(ground_truth, context_chunks)
        else:
            recall = await evaluator.score_context_precision(query, context_chunks)
            
        overall = float((faithfulness + relevancy + recall) / 3.0)
        
        # 4. Save results to Postgres
        try:
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO answer_evaluations (session_id, query, answer, faithfulness, relevancy, recall, overall)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    session_id, query, answer, faithfulness, relevancy, recall, overall
                )
                logger.info("Saved evaluations to PostgreSQL successfully.")
        except Exception as e:
            logger.error(f"Failed to save evaluation to PostgreSQL: {e}")

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "recall": recall,
            "overall": overall
        }

eval_agent = EvalAgent()

@app.post("/tasks/send", response_model=EvalResponse)
async def execute_task(request: EvalRequest):
    try:
        results = await eval_agent.evaluate(
            session_id=request.session_id,
            query=request.query,
            answer=request.answer,
            context_chunks=request.context_chunks,
            ground_truth=request.ground_truth
        )
        return EvalResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "EvalAgent",
        "description": "Scores RAG answers against faithfulness, recall, and relevancy metrics",
        "version": "1.0",
        "capabilities": ["faithfulness_scoring", "relevancy_scoring", "recall_scoring", "logging"],
        "inputModes": ["text", "structured_data"],
        "outputModes": ["structured_data"]
    }

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8103
    uvicorn.run(app, host="0.0.0.0", port=port)
