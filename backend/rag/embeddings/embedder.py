import os
import logging
from typing import List
import numpy as np
from google import genai
from google.genai.errors import APIError

logger = logging.getLogger("rip.embeddings")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IS_MOCK_ENV = os.getenv("MOCK_DB", "false").lower() == "true" or not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_")

class GeminiEmbedder:
    def __init__(self):
        if IS_MOCK_ENV:
            logger.info("Initializing mock GeminiEmbedder (generating random vectors).")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
                logger.info("Google GenAI client initialized for embeddings.")
            except Exception as e:
                logger.warning(f"Could not initialize real Gemini Client: {e}. Falling back to mock embedder.")
                self.client = None

    async def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query."""
        if not text:
            return [0.0] * 768
            
        if self.client is None:
            # Return pseudo-random vector based on hash of text for deterministic testing
            rng = np.random.default_rng(hash(text) & 0xffffffff)
            vec = rng.standard_normal(768)
            norm = np.linalg.norm(vec)
            return (vec / (norm if norm > 0 else 1.0)).tolist()

        try:
            response = await self.client.aio.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            # Response handling depending on structure
            if hasattr(response, "embeddings") and response.embeddings:
                return response.embeddings[0].values
            elif hasattr(response, "embedding") and response.embedding:
                return response.embedding.values
            else:
                # Fallback response inspection
                values = response.embedding.values if hasattr(response.embedding, "values") else response.embedding
                return values
        except APIError as e:
            logger.error(f"Gemini embedding API error: {e}. Falling back to mock vector.")
            rng = np.random.default_rng(hash(text) & 0xffffffff)
            return rng.standard_normal(768).tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}. Falling back to mock vector.")
            rng = np.random.default_rng(hash(text) & 0xffffffff)
            return rng.standard_normal(768).tolist()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents/chunks."""
        if not texts:
            return []
            
        if self.client is None:
            results = []
            for text in texts:
                rng = np.random.default_rng(hash(text) & 0xffffffff)
                vec = rng.standard_normal(768)
                norm = np.linalg.norm(vec)
                results.append((vec / (norm if norm > 0 else 1.0)).tolist())
            return results

        try:
            # Batch embedding
            response = await self.client.aio.models.embed_content(
                model="text-embedding-004",
                contents=texts
            )
            if hasattr(response, "embeddings"):
                return [emb.values for emb in response.embeddings]
            return [emb.values for emb in response.embedding]
        except Exception as e:
            logger.error(f"Error batch embedding: {e}. Generating mock embeddings.")
            results = []
            for text in texts:
                rng = np.random.default_rng(hash(text) & 0xffffffff)
                results.append(rng.standard_normal(768).tolist())
            return results

# Singleton instance
embedder = GeminiEmbedder()
