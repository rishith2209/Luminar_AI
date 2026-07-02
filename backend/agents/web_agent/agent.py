import os
import re
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from google import genai
from google.genai import types
from bs4 import BeautifulSoup
import httpx

from backend.rag.ingestion.pipeline import ingestion_pipeline

logger = logging.getLogger("rip.agents.web")

app = FastAPI(title="Web Agent", version="1.0")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IS_MOCK_ENV = os.getenv("MOCK_DB", "false").lower() == "true" or not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_")

class WebQueryRequest(BaseModel):
    query: str

class WebQueryResponse(BaseModel):
    chunks: List[Dict[str, Any]]

class WebAgent:
    def __init__(self):
        if IS_MOCK_ENV:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize GenAI client: {e}")
                self.client = None

    async def search_and_scrape(self, query: str) -> List[Dict[str, Any]]:
        """Perform search using Google Search grounding tool and scrape top results."""
        if self.client is None:
            # Mock web search result
            return [
                {
                    "id": "web-mock-1",
                    "text": f"Web search details about {query}: Current web consensus indicates rapid advancements in multi-agent orchestration and protocols like Model Context Protocol (MCP).",
                    "source_url": "https://example.com/mock-search",
                    "title": "Mock Search Results",
                    "page_num": 1,
                    "chunk_index": 0,
                    "live_web": True
                }
            ]

        try:
            # Call Gemini with Google Search tool enabled
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Search the web and list the top 3 most relevant URLs discussing: {query}",
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
                )
            )
            
            urls = []
            # Extract grounded URLs if available in candidates
            if response.candidates and response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                if metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        if chunk.web and chunk.web.uri:
                            urls.append((chunk.web.title, chunk.web.uri))
            
            # Fallback regex search for URLs in response text if grounding metadata is empty
            if not urls and response.text:
                found_urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', response.text)
                for u in found_urls[:3]:
                    urls.append(("Web Page", u))

            if not urls:
                # Add default search URLs
                urls = [("Search Grounding", f"https://www.google.com/search?q={query.replace(' ', '+')}")]

            # Scrape top 3 URLs
            scraped_chunks = []
            async with httpx.AsyncClient(timeout=8.0) as client:
                for idx, (title, url) in enumerate(urls[:3]):
                    try:
                        headers = {"User-Agent": "Mozilla/5.0 RIP/1.0"}
                        res = await client.get(url, headers=headers)
                        if res.status_code == 200:
                            soup = BeautifulSoup(res.text, "html.parser")
                            for s in soup(["script", "style", "nav", "footer"]):
                                s.decompose()
                            text = soup.get_text(separator=" ").strip()
                            # Clean text
                            text = " ".join(text.split())
                            
                            # Use semantic chunker directly
                            chunks = await ingestion_pipeline.semantic_chunking(text[:5000])  # limit to first 5k chars to keep fast
                            for c_idx, c_text in enumerate(chunks[:2]):  # Top 2 chunks from each site
                                scraped_chunks.append({
                                    "id": f"web-{idx}-{c_idx}",
                                    "text": c_text,
                                    "source_url": url,
                                    "title": title or "Web Source",
                                    "page_num": 1,
                                    "chunk_index": c_idx,
                                    "live_web": True
                                })
                    except Exception as e:
                        logger.error(f"Error scraping {url}: {e}")

            # If scraping failed, return Gemini text response as a chunk
            if not scraped_chunks:
                scraped_chunks.append({
                    "id": "web-summary",
                    "text": response.text or "No search results returned.",
                    "source_url": "https://www.google.com/search",
                    "title": "Google Search Grounding Summary",
                    "page_num": 1,
                    "chunk_index": 0,
                    "live_web": True
                })

            return scraped_chunks
        except Exception as e:
            logger.error(f"Web agent search error: {e}")
            return []

web_agent = WebAgent()

@app.post("/tasks/send", response_model=WebQueryResponse)
async def execute_task(request: WebQueryRequest):
    try:
        chunks = await web_agent.search_and_scrape(request.query)
        return WebQueryResponse(chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "WebAgent",
        "description": "Performs Google Search grounding and scraps web sources",
        "version": "1.0",
        "capabilities": ["web_search", "web_scraping"],
        "inputModes": ["text"],
        "outputModes": ["structured_data"]
    }

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8104
    uvicorn.run(app, host="0.0.0.0", port=port)
