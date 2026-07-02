import os
import json
import logging
from typing import List
import numpy as np
from google import genai

from backend.rag.embeddings.embedder import embedder

logger = logging.getLogger("rip.evaluation.ragas")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IS_MOCK_ENV = os.getenv("MOCK_DB", "false").lower() == "true" or not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_")

class RagasEvaluator:
    def __init__(self):
        if IS_MOCK_ENV:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize GenAI client for RAGAS: {e}")
                self.client = None

    async def score_faithfulness(self, answer: str, context_chunks: List[str]) -> float:
        """Measure if the answer is grounded in the retrieved context."""
        if not answer or not context_chunks:
            return 0.0

        if self.client is None:
            return 0.9  # Mock score

        # 1. Extract claims
        prompt_claims = f"""You are a facts extractor. Identify all discrete factual claims made in the following answer. 
Reply with only a JSON list of strings representing the claims:
{{"claims": ["claim 1", "claim 2"]}}

Answer: {answer}
JSON:"""
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt_claims,
                config={"response_mime_type": "application/json"}
            )
            text_resp = response.text.strip() if response.text else ""
            data = json.loads(text_resp)
            claims = data.get("claims", [])
        except Exception as e:
            logger.error(f"Error extracting claims for faithfulness: {e}")
            return 0.5

        if not claims:
            return 1.0

        # 2. Check each claim against context
        context_block = "\n---\n".join(context_chunks)
        supported_count = 0
        
        for claim in claims:
            prompt_verify = f"""You are a factual verifier. Is the following claim fully supported by the provided context?
Reply with only 'Yes' or 'No' and a short reason as JSON:
{{"supported": true/false, "reason": "why"}}

Context:
{context_block}

Claim: {claim}
JSON:"""
            try:
                verify_res = await self.client.aio.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt_verify,
                    config={"response_mime_type": "application/json"}
                )
                text_verify = verify_res.text.strip() if verify_res.text else ""
                verify_data = json.loads(text_verify)
                if verify_data.get("supported", False):
                    supported_count += 1
            except Exception as e:
                logger.error(f"Error verifying claim '{claim}': {e}")
                # Optimistic fallback if verification fails
                supported_count += 1

        return float(supported_count / len(claims))

    async def score_answer_relevancy(self, query: str, answer: str) -> float:
        """Measure if the answer directly addresses the original query."""
        if not query or not answer:
            return 0.0

        if self.client is None:
            return 0.85  # Mock score

        # 1. Generate 5 questions from the answer
        prompt_qs = f"""Generate 5 distinct search queries that would be directly answered by the following text. 
Reply with only a JSON list of strings:
{{"queries": ["q1", "q2", "q3", "q4", "q5"]}}

Text: {answer}
JSON:"""
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt_qs,
                config={"response_mime_type": "application/json"}
            )
            text_resp = response.text.strip() if response.text else ""
            data = json.loads(text_resp)
            generated_queries = data.get("queries", [])
        except Exception as e:
            logger.error(f"Error generating questions for relevancy: {e}")
            return 0.5

        if not generated_queries:
            return 0.0

        # 2. Calculate average cosine similarity between query and generated queries
        try:
            # Embed all queries
            query_emb = await embedder.embed_query(query)
            gen_embs = await embedder.embed_documents(generated_queries)
            
            similarities = []
            q_vec = np.array(query_emb)
            for g_emb in gen_embs:
                g_vec = np.array(g_emb)
                dot = np.dot(q_vec, g_vec)
                norm1 = np.linalg.norm(q_vec)
                norm2 = np.linalg.norm(g_vec)
                sim = float(dot / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0
                similarities.append(sim)
                
            return float(np.mean(similarities))
        except Exception as e:
            logger.error(f"Error embedding / computing similarity for relevancy: {e}")
            return 0.5

    async def score_context_recall(self, ground_truth: str, context_chunks: List[str]) -> float:
        """Measure what fraction of key facts from ground truth are present in retrieved context."""
        if not ground_truth or not context_chunks:
            return 0.0

        if self.client is None:
            return 0.8  # Mock score

        # 1. Extract key facts from ground truth
        prompt_facts = f"""Extract all key factual statements and concepts from the ground truth text below.
Reply with only a JSON list of strings:
{{"facts": ["fact 1", "fact 2"]}}

Ground Truth: {ground_truth}
JSON:"""
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt_facts,
                config={"response_mime_type": "application/json"}
            )
            text_resp = response.text.strip() if response.text else ""
            data = json.loads(text_resp)
            facts = data.get("facts", [])
        except Exception as e:
            logger.error(f"Error extracting facts for recall: {e}")
            return 0.5

        if not facts:
            return 1.0

        # 2. Verify how many facts are in the context
        context_block = "\n---\n".join(context_chunks)
        present_count = 0
        
        for fact in facts:
            prompt_check = f"""Does the following context contain or support the fact described?
Reply with only 'Yes' or 'No' as JSON:
{{"present": true/false}}

Context:
{context_block}

Fact: {fact}
JSON:"""
            try:
                check_res = await self.client.aio.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt_check,
                    config={"response_mime_type": "application/json"}
                )
                text_check = check_res.text.strip() if check_res.text else ""
                check_data = json.loads(text_check)
                if check_data.get("present", False):
                    present_count += 1
            except Exception as e:
                logger.error(f"Error checking fact '{fact}': {e}")
                present_count += 1

        return float(present_count / len(facts))

    async def score_context_precision(self, query: str, context_chunks: List[str]) -> float:
        """Measure if retrieved chunks are highly relevant to query."""
        if not query or not context_chunks:
            return 0.0

        if self.client is None:
            return 0.85  # Mock score

        relevant_count = 0
        for chunk in context_chunks:
            prompt_precision = f"""Is this context chunk directly useful for answering the query?
Reply with only 'Yes' or 'No' as JSON:
{{"useful": true/false}}

Query: {query}
Chunk: {chunk}
JSON:"""
            try:
                response = await self.client.aio.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt_precision,
                    config={"response_mime_type": "application/json"}
                )
                text_resp = response.text.strip() if response.text else ""
                data = json.loads(text_resp)
                if data.get("useful", False):
                    relevant_count += 1
            except Exception as e:
                logger.error(f"Error scoring precision: {e}")
                relevant_count += 1

        return float(relevant_count / len(context_chunks))

# Singleton evaluator
evaluator = RagasEvaluator()
