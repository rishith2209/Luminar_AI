# Enterprise AI Research Intelligence Platform

> A production-grade multi-agent document research and intelligence platform built on Google ADK, A2A agent networking, advanced RAG architectures, and Model Context Protocol (MCP).

---

## System Architecture

```text
       +-------------------------------------------------------------+
       |                     Next.js 15 Frontend                     |
       +------------------------------+------------------------------+
                                      | WebSocket Trace / SSE Stream
                                      v
       +-------------------------------------------------------------+
       |                  FastAPI API Gateway Router                 |
       +------------------------------+------------------------------+
                                      | A2A Protocol (JSON-RPC)
                                      v
       +-------------------------------------------------------------+
       |           Google ADK Orchestrator Multi-Agent System        |
       |  [Orchestrator] -> [Retrieval] -> [Synthesis] -> [Eval]     |
       +-------+----------------------+---------------+--------------+
               |                      |               |
               v                      v               v
       +---------------+      +---------------+      +---------------+
       |   Qdrant DB   |      |  Neo4j Graph  |      |   Redis Cache |
       |   (Vectors)   |      |   (Entities)  |      |   (HyDE TTL)  |
       +---------------+      +---------------+      +---------------+
```

---

## Technical Highlights

- **A2A Agent Networking**: Implements Google ADK agents running over independent HTTP interfaces exchanging structured JSON-RPC Tasks and Artifacts.
- **Advanced 5-tier RAG**: Bypasses naive retrieval limitations using Hybrid RRF Search, HyDE (Hypothetical Document Embeddings), Self-RAG relevance validation, Graph RAG 2-hop traversal, and Contextual Sentence Compression.
- **Standards-Compliant MCP Server**: Exposes 8 operational tools enabling zero-integration-code usage inside Cursor, VS Code, or Claude.
- **RAGAS Evaluations Scored on the Fly**: Every response is evaluated for Faithfulness, Relevancy, and Context Precision using Gemini Flash as an automated judge.

---

## Quick Start (5 Commands)

```bash
# 1. Clone & cd into project folder
cd research-intelligence-platform

# 2. Duplicate environment configurations
copy .env.example .env

# 3. Boot database instances and microservices
docker-compose up --build -d

# 4. Ingest baseline documentation
curl -X POST http://localhost:8000/api/ingest -F "url=https://modelcontextprotocol.io" -F "source_type=url"

# 5. Query the platform
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d "{\"query\": \"Explain MCP architecture\"}"
```

---

## RAG Strategies Reference

1. **Hybrid RRF Search**: Merges keyword-based BM25 relevance and dense vector cosine similarity (using Google's `text-embedding-004`) ranked by Reciprocal Rank Fusion.
2. **HyDE**: Employs Gemini Flash to generate a hypothetical answer matching the query intent, embedding that placeholder to locate source text.
3. **Self-RAG**: Analyzes document relevancy dynamically. Expands the search query using synonyms if retrieved density is poor.
4. **Graph RAG**: Extracts Entity-Relation triplets, loads them to Neo4j, and performs 2-hop traversals to augment vector results.
5. **Contextual Compression**: Trims retrieved chunks down to sentences answering the query, decreasing prompt sizes by ~60%.

---

## MCP Tools Reference

| Tool Name | Input Parameters | Output Description |
| :--- | :--- | :--- |
| `search_knowledge_base` | `{query, top_k, strategy}` | RRF Hybrid / HyDE / Graph vector search hits |
| `ingest_document` | `{source, source_type, title}` | Document registry ID and chunk counts |
| `ask_research_agent` | `{query, mode}` | Grounded answers, citations, evaluations |
| `query_knowledge_graph` | `{entity, hops}` | Traversed entities list and ASCII schema |
| `evaluate_answer` | `{query, answer, context_chunks}` | RAGAS evaluation scores |
| `get_document_stats` | `{}` | Total database sizes and updates |
| `run_benchmark` | `{sample_size}` | Evaluated query benchmarks |
| `explain_retrieval` | `{query, chunk_id}` | BM25, semantic, and RRF score breakdowns |

---

## Benchmark Metrics

| Metric | Target | Description |
| :--- | :--- | :--- |
| **Faithfulness** | `>= 0.85` | Ratio of synthesized claims backed by context |
| **Answer Relevancy**| `>= 0.80` | Measure of how well the output addresses the query |
| **Recall Accuracy** | `>= 0.78` | Coverage of target ground truth facts |
| **P95 Latency** | `< 250ms` | Vector retrieval latency benchmarks |

---

## Contributing

1. Format python code: `ruff format .`
2. Run tests: `pytest backend/`
3. Verify benchmark runner passes gates prior to pushing PRs.
