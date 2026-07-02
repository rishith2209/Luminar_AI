import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.rag.retrieval.hybrid_search import HybridSearcher
from backend.rag.strategies.self_rag import SelfRAGStrategy
from backend.rag.strategies.hyde import HyDEStrategy
from backend.evaluation.ragas_eval import RagasEvaluator
from backend.rag.db import qdrant_client

@pytest.mark.asyncio
async def test_reciprocal_rank_fusion():
    """Verify RRF correctly merges and ranks BM25 and vector search outputs."""
    searcher = HybridSearcher()
    
    # Mock database retrieval returning chunks
    mock_chunks = [
        {"id": "chunk-1", "text": "Model Context Protocol or MCP is standard", "source_url": "url-1", "page_num": 1, "chunk_index": 0, "title": "MCP Specs"},
        {"id": "chunk-2", "text": "A2A defines agent messaging rules", "source_url": "url-2", "page_num": 1, "chunk_index": 0, "title": "A2A Specs"}
    ]
    
    # Mocking Qdrant search
    mock_point_1 = MagicMock()
    mock_point_1.id = "chunk-1"
    mock_point_1.score = 0.9
    mock_point_1.payload = {"text": "Model Context Protocol or MCP is standard", "source_url": "url-1", "page_num": 1, "chunk_index": 0, "title": "MCP Specs"}
    
    mock_query_res = MagicMock(points=[mock_point_1])
    
    with patch.object(searcher, "get_all_chunks", return_value=mock_chunks), \
         patch.object(qdrant_client, "query_points", return_value=mock_query_res):
         
        results = await searcher.search("Model Context Protocol", top_k=2)
        assert len(results) > 0
        assert results[0]["id"] == "chunk-1"
        assert "semantic" in results[0]["scores"]

@pytest.mark.asyncio
async def test_hyde_hypothesis_generation():
    """Verify HyDE constructs a hypothetical answer vector."""
    strategy = HyDEStrategy()
    
    mock_empty_res = MagicMock(points=[])
    
    # Mock Gemini client hypothesis text and embedder
    with patch.object(strategy, "generate_hypothesis", return_value="Hypothetical answer") as mock_gen, \
         patch("backend.rag.embeddings.embedder.embedder.embed_query", return_value=[0.1]*768) as mock_embed, \
         patch.object(qdrant_client, "query_points", return_value=mock_empty_res):
         
        results = await strategy.search("What is MCP?", top_k=2)
        assert len(results) == 0
        mock_gen.assert_called_once_with("What is MCP?")
        mock_embed.assert_called_once_with("Hypothetical answer")

@pytest.mark.asyncio
async def test_self_rag_gating():
    """Verify Self-RAG relevance score filtering triggers query expansions."""
    strategy = SelfRAGStrategy()
    
    # Mock scorer returning high score for first chunk, low for second
    async def mock_score(query, text):
        if "first" in text:
            return {"score": 5, "reason": "Relevant"}
        return {"score": 1, "reason": "Irrelevant"}

    strategy.score_passage = mock_score
    strategy.expand_query = AsyncMock(return_value="expanded query")
    
    mock_candidates = [
        {"id": "c-1", "text": "This is the first passage describing platform details"},
        {"id": "c-2", "text": "Unrelated details about cooking recipes"}
    ]
    
    with patch("backend.rag.retrieval.hybrid_search.hybrid_searcher.search", return_value=mock_candidates):
        # We search with retry=1 to prevent recursion loop
        results = await strategy.search("platform", top_k=2, retry_count=1)
        assert len(results) == 1
        assert results[0]["id"] == "c-1"

@pytest.mark.asyncio
async def test_ragas_evaluation_faithfulness():
    """Verify custom RAGAS Faithfulness evaluator matches claim supports."""
    eval_harness = RagasEvaluator()
    
    # Mock Gemini client calls
    mock_client = MagicMock()
    
    # Mock first call to extract claims, second/third to verify them
    mock_res_claims = MagicMock()
    mock_res_claims.text = '{"claims": ["The model is fast", "The context is clean"]}'
    
    mock_res_v1 = MagicMock()
    mock_res_v1.text = '{"supported": true, "reason": "supported"}'
    mock_res_v2 = MagicMock()
    mock_res_v2.text = '{"supported": false, "reason": "unsupported"}'
    
    mock_client.aio.models.generate_content = AsyncMock()
    mock_client.aio.models.generate_content.side_effect = [
        mock_res_claims,
        mock_res_v1,
        mock_res_v2
    ]
    
    eval_harness.client = mock_client
    
    score = await eval_harness.score_faithfulness(
        answer="The model is fast and the context is clean.",
        context_chunks=["Context indicates the model is fast. No text on clean context."]
    )
    
    # Expect 1/2 supported = 0.5 score
    assert score == 0.5
