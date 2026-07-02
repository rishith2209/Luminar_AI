import re
import uuid
import logging
import numpy as np
import datetime
from typing import List, Dict, Any, Tuple
import httpx
from bs4 import BeautifulSoup

# Try importing parsing packages, handle import errors gracefully
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

from backend.rag.db import qdrant_client, get_postgres_pool, COLLECTION_NAME
from backend.rag.embeddings.embedder import embedder

logger = logging.getLogger("rip.ingestion")

class DocumentIngestionPipeline:
    def __init__(self):
        pass

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple rule-based sentence splitter."""
        # Split on sentence terminals followed by space or end of string
        sentence_end = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_end.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    async def semantic_chunking(self, text: str, similarity_threshold: float = 0.85, max_chars: int = 1500) -> List[str]:
        """Group sentences dynamically based on embedding similarities."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
            
        import numpy as np # import inside as it's a utility
        
        # Get embeddings for all sentences
        embeddings = await embedder.embed_documents(sentences)
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_emb = embeddings[i]
            
            # Check similarity with current chunk's average embedding
            # In a simplified version, check similarity with the previous sentence
            # representing continuity. We'll use similarity between current_chunk_embedding and sentence_emb
            dot = np.dot(current_chunk_embedding, sentence_emb)
            norm1 = np.linalg.norm(current_chunk_embedding)
            norm2 = np.linalg.norm(sentence_emb)
            sim = float(dot / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0
            
            current_len = sum(len(s) for s in current_chunk_sentences)
            
            # If similarity is high enough and it doesn't exceed maximum character limit
            if sim >= similarity_threshold and (current_len + len(sentence) < max_chars):
                current_chunk_sentences.append(sentence)
                # Update rolling average embedding
                current_chunk_embedding = np.mean([current_chunk_embedding, sentence_emb], axis=0).tolist()
            else:
                # Save the completed chunk
                chunks.append(" ".join(current_chunk_sentences))
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_chunk_embedding = sentence_emb
                
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return chunks

    async def parse_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF pages using PyMuPDF."""
        pages = []
        if not fitz:
            logger.error("fitz (PyMuPDF) is not installed.")
            raise ImportError("PyMuPDF is required to parse PDFs.")
            
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            if page_text.strip():
                pages.append((page_text, page_num + 1))
        doc.close()
        return pages

    async def parse_docx(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from DOCX using python-docx."""
        if not docx:
            logger.error("python-docx is not installed.")
            raise ImportError("python-docx is required to parse DOCX.")
            
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return [(text, 1)]  # Return as single page context

    async def parse_txt(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from TXT or MD files."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [(content, 1)]

    async def parse_url(self, url: str) -> List[Tuple[str, int]]:
        """Scrape web pages using httpx and BeautifulSoup."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RIP/1.0"}
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        text = soup.get_text(separator="\n")
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        
        return [(clean_text, 1)]

    async def parse_youtube(self, url: str) -> List[Tuple[str, int]]:
        """Extract transcript from YouTube video URL."""
        if not YouTubeTranscriptApi:
            logger.error("youtube-transcript-api is not installed.")
            raise ImportError("youtube-transcript-api is required to parse YouTube.")
            
        # Extract video ID from URL
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        if not video_id_match:
            raise ValueError(f"Invalid YouTube URL: {url}")
        video_id = video_id_match.group(1)
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([item['text'] for item in transcript_list])
            return [(full_text, 1)]
        except Exception as e:
            logger.error(f"Error fetching YouTube transcript for {video_id}: {e}")
            # Fallback to web scraping if API fails or video has no captions
            return await self.parse_url(url)

    async def ingest_document(self, source: str, source_type: str, metadata: Dict[str, Any] = None) -> Tuple[str, int]:
        """Ingests a document, splits semantically, embeds chunks, and indexes them."""
        doc_id = str(uuid.uuid4())
        title = metadata.get("title", source.split("/")[-1] if "/" in source else source)
        source_url = source if source_type in ["url", "youtube"] else ""
        
        # 1. Parse content based on source type
        pages = []
        if source_type == "pdf":
            pages = await self.parse_pdf(source)
        elif source_type == "docx":
            pages = await self.parse_docx(source)
        elif source_type in ["txt", "markdown"]:
            pages = await self.parse_txt(source)
        elif source_type == "url":
            pages = await self.parse_url(source)
        elif source_type == "youtube":
            pages = await self.parse_youtube(source)
        elif source_type == "text":
            pages = [(source, 1)] # Source is direct text passed
            title = metadata.get("title", "Direct Text Snippet")
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
            
        # 2. Chunk semantically page-by-page
        all_chunks = []
        for text, page_num in pages:
            chunks = await self.semantic_chunking(text)
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "page_num": page_num,
                    "chunk_index": chunk_idx
                })
                
        if not all_chunks:
            return doc_id, 0

        # 3. Batch generate embeddings for all chunks
        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = await embedder.embed_documents(chunk_texts)
        
        # 4. Save metadata to Postgres
        postgres_pool = await get_postgres_pool()
        async with postgres_pool.acquire() as conn:
            # Insert document
            await conn.execute(
                "INSERT INTO documents (id, title, source_url, source_type) VALUES ($1, $2, $3, $4)",
                uuid.UUID(doc_id), title, source_url, source_type
            )
            
            # Insert chunks in batch
            for idx, c in enumerate(all_chunks):
                await conn.execute(
                    """INSERT INTO chunks (id, document_id, text, page_num, chunk_index, source_url) 
                       VALUES ($1, $2, $3, $4, $5, $6)""",
                    uuid.UUID(c["id"]), uuid.UUID(doc_id), c["text"], c["page_num"], c["chunk_index"], source_url
                )

        # 5. Index in Neo4j Graph DB
        from backend.rag.strategies.graph_rag import graph_rag_strategy
        for c in all_chunks:
            try:
                await graph_rag_strategy.index_graph(c["text"], doc_id)
            except Exception as e:
                logger.error(f"Failed to index Neo4j graph for chunk: {e}")

        # 6. Save vectors in Qdrant
        points = []
        from qdrant_client.http import models as qmodels
        
        for idx, c in enumerate(all_chunks):
            points.append(qmodels.PointStruct(
                id=c["id"],
                vector=embeddings[idx],
                payload={
                    "text": c["text"],
                    "source_url": source_url,
                    "title": title,
                    "page_num": c["page_num"],
                    "chunk_index": c["chunk_index"],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "document_id": doc_id
                }
            ))
            
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        logger.info(f"Ingested document {title} ({doc_id}) with {len(all_chunks)} chunks.")
        return doc_id, len(all_chunks)

# Singleton pipeline
ingestion_pipeline = DocumentIngestionPipeline()
