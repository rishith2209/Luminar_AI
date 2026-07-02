import os
import json
import logging
from typing import List, Dict, Any
from google import genai

from backend.rag.db import get_neo4j_driver

logger = logging.getLogger("rip.strategies.graph_rag")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
IS_MOCK_ENV = os.getenv("MOCK_DB", "false").lower() == "true" or not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_")

class GraphRAGStrategy:
    def __init__(self):
        if IS_MOCK_ENV:
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client for Graph RAG: {e}")
                self.client = None

    async def extract_entities_and_relationships(self, text: str) -> Dict[str, Any]:
        """Extract entities and relations from text using Gemini Pro."""
        if self.client is None:
            # Mock extraction
            return {
                "entities": [{"name": "Retrieval-Augmented Generation", "type": "Concept"}, {"name": "LLM", "type": "Technology"}],
                "relations": [{"from": "Retrieval-Augmented Generation", "to": "LLM", "type": "AUGMENTS"}]
            }

        prompt = f"""Extract all entities (people, orgs, concepts, dates) and their relationships from this text as JSON: 
{{entities: [{{name, type}}], relations: [{{from, to, type}}]}}

Text: {text}
JSON:"""
        try:
            # Using gemini-2.0-flash (as standard, or we can use gemini-2.0-flash as the fallback)
            # The prompt requested Gemini Pro for extraction. In google-genai, the Gemini Pro model is gemini-2.5-pro or gemini-2.0-pro (or gemini-1.5-pro).
            # We'll use gemini-2.5-pro, falling back to gemini-2.0-flash if there's any model-not-found error.
            model_name = "gemini-2.5-pro"
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
            except Exception:
                model_name = "gemini-2.0-flash"
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
            text_resp = response.text.strip() if response.text else ""
            return json.loads(text_resp)
        except Exception as e:
            logger.error(f"Error extracting entities from text: {e}")
            return {"entities": [], "relations": []}

    async def index_graph(self, text: str, doc_id: str):
        """Extracts entities and stores them in Neo4j."""
        data = await self.extract_entities_and_relationships(text)
        entities = data.get("entities", [])
        relations = data.get("relations", [])
        
        if not entities and not relations:
            return
            
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Merge entities
            for ent in entities:
                name = ent.get("name")
                ent_type = ent.get("type", "Entity")
                if name:
                    session.run(
                        "MERGE (e:Entity {name: $name}) ON CREATE SET e.type = $type",
                        name=name, type=ent_type
                    )
            
            # Merge relationships
            for rel in relations:
                source = rel.get("from")
                target = rel.get("to")
                rel_type = rel.get("type", "RELATED_TO").replace(" ", "_").upper()
                if source and target:
                    # Create nodes just in case they weren't in entities
                    session.run("MERGE (:Entity {name: $name})", name=source)
                    session.run("MERGE (:Entity {name: $name})", name=target)
                    
                    # Create relationship dynamically using Cypher parameter interpolation
                    query = f"""
                    MATCH (source:Entity {{name: $source}})
                    MATCH (target:Entity {{name: $target}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    ON CREATE SET r.document_id = $doc_id
                    """
                    session.run(query, source=source, target=target, doc_id=doc_id)
        logger.info(f"Indexed {len(entities)} entities and {len(relations)} relations to Neo4j.")

    async def extract_query_entities(self, query: str) -> List[str]:
        """Extracts query entities for graph lookup using Gemini Flash."""
        if self.client is None:
            # Simple keyword extraction mock
            return [query.split()[-1]] if query.split() else []

        prompt = f"""Extract a list of named entities or primary subjects from this query as a JSON array of strings: 
{{"entities": ["string"]}}

Query: {query}
JSON:"""
        try:
            response = await self.client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            text_resp = response.text.strip() if response.text else ""
            data = json.loads(text_resp)
            return data.get("entities", [])
        except Exception as e:
            logger.error(f"Error extracting entities from query: {e}")
            return []

    async def search(self, query: str, hops: int = 2) -> List[Dict[str, Any]]:
        """Query Neo4j for 2-hop neighborhood of extracted query entities."""
        entities = await self.extract_query_entities(query)
        if not entities:
            return []

        driver = get_neo4j_driver()
        graph_chunks = []
        
        with driver.session() as session:
            for entity in entities:
                # Query 2-hop neighborhood
                query_cypher = """
                MATCH (n:Entity) WHERE toLower(n.name) = toLower($entity)
                MATCH path = (n)-[r*1..2]-(m:Entity)
                RETURN path LIMIT 10
                """
                results = session.run(query_cypher, entity=entity)
                
                # Format paths to text descriptions
                for record in results:
                    path = record.get("path")
                    if not path:
                        continue
                        
                    # Build readable path string
                    nodes = path.nodes
                    relationships = path.relationships
                    
                    description = []
                    for i, rel in enumerate(relationships):
                        start_node = nodes[i]
                        end_node = nodes[i+1]
                        rel_type = rel.type
                        description.append(
                            f"({start_node.get('name')} [{start_node.get('type')}] "
                            f"is {rel_type} to "
                            f"{end_node.get('name')} [{end_node.get('type')}])"
                        )
                    
                    chunk_text = " Knowledge Graph Context: " + " and ".join(description)
                    graph_chunks.append({
                        "id": f"graph-{entity}-{hash(chunk_text)}",
                        "text": chunk_text,
                        "source_url": "neo4j_knowledge_graph",
                        "page_num": 1,
                        "chunk_index": 0,
                        "title": f"Knowledge Graph - {entity}",
                        "score": 0.95,  # Boosted score
                        "graph_source": True
                    })

        # Remove duplicate graph chunks
        seen = set()
        unique_graph_chunks = []
        for chunk in graph_chunks:
            if chunk["text"] not in seen:
                seen.add(chunk["text"])
                unique_graph_chunks.append(chunk)

        logger.info(f"Graph RAG retrieved {len(unique_graph_chunks)} unique context chunks for entities: {entities}")
        return unique_graph_chunks

# Singleton instance
graph_rag_strategy = GraphRAGStrategy()
