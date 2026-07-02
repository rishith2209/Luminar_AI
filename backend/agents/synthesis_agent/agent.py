import os
import json
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from google import genai
from google.genai import types

logger = logging.getLogger("rip.agents.synthesis")

app = FastAPI(title="Synthesis Agent", version="1.0")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IS_MOCK_ENV = os.getenv("MOCK_DB", "false").lower() == "true" or not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_")

class ChunkInput(BaseModel):
    id: str
    text: str
    source_url: str
    title: str
    page_num: int = 1
    chunk_index: int = 0
    graph_source: bool = False
    live_web: bool = False

class SynthesisRequest(BaseModel):
    query: str
    chunks: List[ChunkInput]

class SynthesisResponse(BaseModel):
    answer_markdown: str
    citations: List[Dict[str, Any]]
    confidence: float
    flags: List[str]

class SynthesisAgent:
    def __init__(self):
        if IS_MOCK_ENV:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize GenAI client: {e}")
                self.client = None

    async def synthesize(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize answer with citations and detect hallucinations using Gemini Flash."""
        if not chunks:
            return {
                "answer_markdown": "No relevant context documents were found to formulate an answer.",
                "citations": [],
                "confidence": 0.0,
                "flags": ["No context chunks provided"]
            }

        # Format context for Gemini
        context_str = ""
        for idx, c in enumerate(chunks):
            src = c.get("source_url") or c.get("title") or "Unknown"
            context_str += f"[{idx + 1}] Source: {src}\nContent: {c.get('text')}\n\n"

        if self.client is None:
            # Mock synthesis response
            citation_list = []
            for idx, c in enumerate(chunks[:3]):
                citation_list.append({
                    "index": idx + 1,
                    "title": c.get("title", "Document"),
                    "source_url": c.get("source_url", ""),
                    "snippet": c.get("text", "")[:100] + "..."
                })
            
            answer = f"According to the provided research documents, the answer is formulated based on contextual chunks. Specifically, we know that: {chunks[0].get('text')[:150]} [1]. Furthermore, research indicates that: {chunks[1].get('text')[:150] if len(chunks) > 1 else 'more data exists'} [2]."
            return {
                "answer_markdown": answer,
                "citations": citation_list,
                "confidence": 0.9,
                "flags": []
            }

        prompt = f"""You are an expert synthesis agent. Your task is to write a comprehensive, markdown-formatted answer to the user query based ONLY on the provided context chunks.

For every factual claim you make, you MUST append an inline citation pointing to the source index, for example [1], [2], [1][3], etc. 
Do not make any claims that cannot be directly sourced from the provided chunks. If you make any statements that are NOT supported by the chunks, you must list them in the "flags" field.

Context Chunks:
{context_str}

Query: {query}

Reply with ONLY a JSON object containing these keys:
- "answer_markdown": "your markdown answer here"
- "citations": [
    {{"index": 1, "title": "...", "source_url": "...", "snippet": "..."}}
  ]
- "confidence": float (between 0.0 and 1.0)
- "flags": ["list any unsupported or speculative statements made, otherwise empty array"]"""

        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            text_resp = response.text.strip() if response.text else ""
            data = json.loads(text_resp)
            return data
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            return {
                "answer_markdown": "Error formulating response via Gemini Flash.",
                "citations": [],
                "confidence": 0.1,
                "flags": [f"Synthesis pipeline error: {e}"]
            }

synthesis_agent = SynthesisAgent()

@app.post("/tasks/send", response_model=SynthesisResponse)
async def execute_task(request: SynthesisRequest):
    try:
        # Map ChunkInput to dict
        chunks_dict = [c.model_dump() for c in request.chunks]
        result = await synthesis_agent.synthesize(request.query, chunks_dict)
        return SynthesisResponse(
            answer_markdown=result.get("answer_markdown", ""),
            citations=result.get("citations", []),
            confidence=result.get("confidence", 0.5),
            flags=result.get("flags", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "SynthesisAgent",
        "description": "Synthesizes research answers with inline citations",
        "version": "1.0",
        "capabilities": ["citation_mapping", "hallucination_flagging", "synthesis"],
        "inputModes": ["text", "structured_data"],
        "outputModes": ["text", "structured_data"]
    }

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8102
    uvicorn.run(app, host="0.0.0.0", port=port)
