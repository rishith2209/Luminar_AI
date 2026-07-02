import os
import logging
from typing import List, Dict, Any
from google import genai
from google.genai.errors import APIError

logger = logging.getLogger("rip.retrieval.compressor")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IS_MOCK_ENV = os.getenv("MOCK_DB", "false").lower() == "true" or not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_")

class ContextualCompressor:
    def __init__(self):
        if IS_MOCK_ENV:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client for Compressor: {e}")
                self.client = None

    async def compress_chunk(self, query: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Compress a single chunk using Gemini Flash to extract relevant sentences."""
        text = chunk.get("text", "")
        if not text:
            return chunk

        if self.client is None:
            # Mock compression: Return the original text but slice it a bit
            sentences = text.split(". ")
            compressed_text = ". ".join(sentences[:min(len(sentences), 3)])
            chunk_copy = chunk.copy()
            chunk_copy["text"] = f"[RELEVANT CONTEXT] {compressed_text}"
            chunk_copy["original_text"] = text
            return chunk_copy

        prompt = f"""Extract only the sentences from this passage that directly answer: "{query}". 
If none of the passage is relevant, return an empty string. 
Do not explain, do not add filler text, do not write a summary. Write ONLY the exact relevant sentences.

Passage: {text}
Extracted Sentences:"""

        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            compressed_text = response.text.strip() if response.text else ""
            if not compressed_text or len(compressed_text) < 10:
                # If compression indicates no relevance
                return None
                
            chunk_copy = chunk.copy()
            chunk_copy["text"] = f"[RELEVANT CONTEXT] {compressed_text}"
            chunk_copy["original_text"] = text
            return chunk_copy
        except Exception as e:
            logger.error(f"Error compressing chunk: {e}")
            # Fallback to returning original chunk with label
            chunk_copy = chunk.copy()
            chunk_copy["text"] = f"[RELEVANT CONTEXT] {text}"
            chunk_copy["original_text"] = text
            return chunk_copy

    async def compress(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress list of retrieved chunks in parallel, filtering out irrelevant items."""
        if not chunks:
            return []
            
        logger.info(f"Compressing {len(chunks)} chunks for query: '{query}'")
        
        # Parallel execution using asyncio.gather
        tasks = [self.compress_chunk(query, chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results (non-relevant chunks)
        compressed_chunks = [res for res in results if res is not None]
        
        logger.info(f"Compressed down from {len(chunks)} to {len(compressed_chunks)} chunks.")
        return compressed_chunks

# Singleton instance
compressor = ContextualCompressor()
