"""
DocuLLaMA Knowledge Graph Manager
AutoLLaMA knowledge graph capabilities for Azure Container Apps
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import Settings, get_settings
from monitoring import get_logger
from document_processor import DocumentChunk

logger = get_logger(__name__)


@dataclass
class KnowledgeNode:
    """Knowledge graph node"""
    node_id: str
    node_type: str  # document, chunk, concept, entity
    title: str
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class KnowledgeEdge:
    """Knowledge graph edge"""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str  # similarity, reference, hierarchy, temporal
    weight: float
    metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class GraphStats:
    """Knowledge graph statistics"""
    total_nodes: int
    total_edges: int
    node_types: Dict[str, int]
    edge_types: Dict[str, int]
    average_degree: float
    clustering_coefficient: float
    connected_components: int


class ConceptExtractor:
    """Extract concepts and entities from text"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        self.concept_threshold = 0.1
    
    async def extract_concepts(self, texts: List[str]) -> Dict[str, List[str]]:
        """Extract key concepts from documents"""
        if not texts:
            return {}
        
        try:
            # Fit TF-IDF on all texts
            tfidf_matrix = self.tfidf.fit_transform(texts)
            feature_names = self.tfidf.get_feature_names_out()
            
            concepts_by_doc = {}
            
            for i, text in enumerate(texts):
                # Get TF-IDF scores for this document
                doc_tfidf = tfidf_matrix[i].toarray()[0]
                
                # Get top concepts
                concept_indices = np.where(doc_tfidf > self.concept_threshold)[0]
                concepts = [feature_names[idx] for idx in concept_indices]
                
                # Sort by TF-IDF score
                scored_concepts = [(concepts[j], doc_tfidf[concept_indices[j]]) 
                                 for j in range(len(concepts))]
                scored_concepts.sort(key=lambda x: x[1], reverse=True)
                
                concepts_by_doc[f"doc_{i}"] = [concept for concept, score in scored_concepts[:20]]
            
            return concepts_by_doc
            
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return {}


class SimilarityAnalyzer:
    """Analyze similarity between documents and concepts"""
    
    def __init__(self):
        self.similarity_threshold = 0.3
    
    async def calculate_semantic_similarity(
        self,
        embeddings1: List[float],
        embeddings2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            vec1 = np.array(embeddings1)
            vec2 = np.array(embeddings2)
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate TF-IDF based content similarity"""
        try:
            tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = tfidf.fit_transform([content1, content2])
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Content similarity calculation failed: {e}")
            return 0.0
    
    async def find_similar_chunks(
        self,
        target_chunk: DocumentChunk,
        all_chunks: List[DocumentChunk],
        similarity_threshold: float = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Find similar chunks to target chunk"""
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        similar_chunks = []
        
        for chunk in all_chunks:
            if chunk.chunk_id == target_chunk.chunk_id:
                continue
            
            # Calculate similarity
            if target_chunk.embeddings and chunk.embeddings:
                similarity = await self.calculate_semantic_similarity(
                    target_chunk.embeddings,
                    chunk.embeddings
                )
            else:
                similarity = await self.calculate_content_similarity(
                    target_chunk.content,
                    chunk.content
                )
            
            if similarity >= similarity_threshold:
                similar_chunks.append((chunk, similarity))
        
        # Sort by similarity
        similar_chunks.sort(key=lambda x: x[1], reverse=True)
        return similar_chunks


class KnowledgeGraphManager:
    """Manage knowledge graph construction and analysis"""
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()
        
        # Graph storage
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        
        # Components
        self.concept_extractor = ConceptExtractor()
        self.similarity_analyzer = SimilarityAnalyzer()
        
        # Configuration
        self.similarity_threshold = 0.3
        self.max_connections_per_node = 10
        self.enable_concept_nodes = True
        self.enable_temporal_edges = True
    
    async def initialize(self):
        """Initialize knowledge graph manager"""
        logger.info("Initializing knowledge graph manager")
        
        # Initialize empty graph
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        
        logger.info("Knowledge graph manager initialized")
    
    async def add_document_chunks(
        self,
        chunks: List[DocumentChunk],
        document_metadata: Dict[str, Any] = None
    ):
        """Add document chunks to knowledge graph"""
        if not chunks:
            return
        
        logger.info(f"Adding {len(chunks)} chunks to knowledge graph")
        
        # Create document node
        document_id = document_metadata.get("document_id", str(uuid.uuid4()))
        document_node = await self._create_document_node(document_id, document_metadata)
        
        # Add chunk nodes
        chunk_nodes = []
        for chunk in chunks:
            chunk_node = await self._create_chunk_node(chunk)
            chunk_nodes.append(chunk_node)
            
            # Connect chunk to document
            await self._create_hierarchy_edge(document_node.node_id, chunk_node.node_id)
        
        # Create similarity edges between chunks
        await self._create_similarity_edges(chunks)
        
        # Extract and connect concepts
        if self.enable_concept_nodes:
            await self._extract_and_connect_concepts(chunks)
        
        # Create temporal edges if enabled
        if self.enable_temporal_edges:
            await self._create_temporal_edges(chunk_nodes)
        
        logger.info(f"Knowledge graph updated: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    async def _create_document_node(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> KnowledgeNode:
        """Create document node"""
        node = KnowledgeNode(
            node_id=f"doc_{document_id}",
            node_type="document",
            title=metadata.get("title", metadata.get("filename", "Unknown Document")),
            content=metadata.get("summary", ""),
            metadata=metadata
        )
        
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **asdict(node))
        
        return node
    
    async def _create_chunk_node(self, chunk: DocumentChunk) -> KnowledgeNode:
        """Create chunk node"""
        node = KnowledgeNode(
            node_id=f"chunk_{chunk.chunk_id}",
            node_type="chunk",
            title=f"Chunk {chunk.position}",
            content=chunk.content,
            metadata=chunk.metadata,
            embeddings=chunk.embeddings
        )
        
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **asdict(node))
        
        return node
    
    async def _create_concept_node(self, concept: str, metadata: Dict[str, Any]) -> KnowledgeNode:
        """Create concept node"""
        node_id = f"concept_{concept.replace(' ', '_').lower()}"
        
        # Check if concept node already exists
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        node = KnowledgeNode(
            node_id=node_id,
            node_type="concept",
            title=concept,
            content=f"Concept: {concept}",
            metadata=metadata
        )
        
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **asdict(node))
        
        return node
    
    async def _create_similarity_edge(
        self,
        source_id: str,
        target_id: str,
        similarity: float,
        edge_type: str = "similarity"
    ) -> KnowledgeEdge:
        """Create similarity edge between nodes"""
        edge_id = f"{edge_type}_{source_id}_{target_id}"
        
        edge = KnowledgeEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=similarity,
            metadata={"similarity_score": similarity}
        )
        
        self.edges[edge.edge_id] = edge
        self.graph.add_edge(
            source_id,
            target_id,
            key=edge_id,
            **asdict(edge)
        )
        
        return edge
    
    async def _create_hierarchy_edge(self, parent_id: str, child_id: str) -> KnowledgeEdge:
        """Create hierarchical edge"""
        return await self._create_similarity_edge(
            parent_id,
            child_id,
            1.0,
            "hierarchy"
        )
    
    async def _create_similarity_edges(self, chunks: List[DocumentChunk]):
        """Create similarity edges between chunks"""
        logger.debug("Creating similarity edges between chunks")
        
        for i, chunk1 in enumerate(chunks):
            similar_chunks = await self.similarity_analyzer.find_similar_chunks(
                chunk1,
                chunks[i+1:],  # Only check remaining chunks to avoid duplicates
                self.similarity_threshold
            )
            
            # Limit connections per node
            similar_chunks = similar_chunks[:self.max_connections_per_node]
            
            for chunk2, similarity in similar_chunks:
                await self._create_similarity_edge(
                    f"chunk_{chunk1.chunk_id}",
                    f"chunk_{chunk2.chunk_id}",
                    similarity
                )
    
    async def _extract_and_connect_concepts(self, chunks: List[DocumentChunk]):
        """Extract concepts and create concept nodes"""
        logger.debug("Extracting concepts from chunks")
        
        # Extract concepts
        chunk_contents = [chunk.content for chunk in chunks]
        concepts_by_doc = await self.concept_extractor.extract_concepts(chunk_contents)
        
        # Create concept nodes and edges
        for doc_key, concepts in concepts_by_doc.items():
            doc_index = int(doc_key.split("_")[1])
            chunk = chunks[doc_index]
            
            for concept in concepts[:5]:  # Top 5 concepts per chunk
                concept_node = await self._create_concept_node(
                    concept,
                    {"extracted_from": chunk.chunk_id}
                )
                
                # Connect chunk to concept
                await self._create_similarity_edge(
                    f"chunk_{chunk.chunk_id}",
                    concept_node.node_id,
                    0.8,  # High weight for extracted concepts
                    "concept_relation"
                )
    
    async def _create_temporal_edges(self, chunk_nodes: List[KnowledgeNode]):
        """Create temporal edges based on chunk positions"""
        # Sort nodes by position
        sorted_nodes = sorted(
            chunk_nodes,
            key=lambda x: x.metadata.get("position", 0)
        )
        
        # Create sequential edges
        for i in range(len(sorted_nodes) - 1):
            current_node = sorted_nodes[i]
            next_node = sorted_nodes[i + 1]
            
            await self._create_similarity_edge(
                current_node.node_id,
                next_node.node_id,
                0.5,  # Moderate weight for temporal connections
                "temporal"
            )
    
    async def find_related_content(
        self,
        node_id: str,
        max_hops: int = 2,
        edge_types: List[str] = None
    ) -> List[KnowledgeNode]:
        """Find related content using graph traversal"""
        if node_id not in self.nodes:
            return []
        
        edge_types = edge_types or ["similarity", "concept_relation", "temporal"]
        related_nodes = []
        visited = set()
        
        # BFS traversal
        queue = [(node_id, 0)]
        visited.add(node_id)
        
        while queue:
            current_id, hops = queue.pop(0)
            
            if hops >= max_hops:
                continue
            
            # Get neighbors
            for neighbor in self.graph.neighbors(current_id):
                if neighbor not in visited:
                    # Check edge types
                    edges = self.graph.get_edge_data(current_id, neighbor)
                    
                    for edge_key, edge_data in edges.items():
                        if edge_data.get("edge_type") in edge_types:
                            visited.add(neighbor)
                            queue.append((neighbor, hops + 1))
                            
                            if neighbor in self.nodes:
                                related_nodes.append(self.nodes[neighbor])
                            break
        
        return related_nodes
    
    async def get_node_centrality(self, centrality_type: str = "betweenness") -> Dict[str, float]:
        """Calculate node centrality measures"""
        try:
            if centrality_type == "betweenness":
                return nx.betweenness_centrality(self.graph)
            elif centrality_type == "closeness":
                return nx.closeness_centrality(self.graph)
            elif centrality_type == "eigenvector":
                return nx.eigenvector_centrality(self.graph, max_iter=1000)
            elif centrality_type == "pagerank":
                return nx.pagerank(self.graph)
            else:
                return nx.degree_centrality(self.graph)
                
        except Exception as e:
            logger.error(f"Centrality calculation failed: {e}")
            return {}
    
    async def detect_communities(self) -> Dict[int, List[str]]:
        """Detect communities in the knowledge graph"""
        try:
            # Convert to undirected graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Use greedy modularity maximization
            communities = nx.community.greedy_modularity_communities(undirected_graph)
            
            community_dict = {}
            for i, community in enumerate(communities):
                community_dict[i] = list(community)
            
            return community_dict
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {}
    
    async def get_graph_stats(self) -> GraphStats:
        """Get knowledge graph statistics"""
        try:
            # Basic stats
            total_nodes = len(self.nodes)
            total_edges = len(self.edges)
            
            # Node types
            node_types = defaultdict(int)
            for node in self.nodes.values():
                node_types[node.node_type] += 1
            
            # Edge types
            edge_types = defaultdict(int)
            for edge in self.edges.values():
                edge_types[edge.edge_type] += 1
            
            # Graph metrics
            if total_nodes > 0:
                average_degree = sum(dict(self.graph.degree()).values()) / total_nodes
                clustering_coefficient = nx.average_clustering(self.graph.to_undirected())
                connected_components = nx.number_connected_components(self.graph.to_undirected())
            else:
                average_degree = 0.0
                clustering_coefficient = 0.0
                connected_components = 0
            
            return GraphStats(
                total_nodes=total_nodes,
                total_edges=total_edges,
                node_types=dict(node_types),
                edge_types=dict(edge_types),
                average_degree=average_degree,
                clustering_coefficient=clustering_coefficient,
                connected_components=connected_components
            )
            
        except Exception as e:
            logger.error(f"Graph stats calculation failed: {e}")
            return GraphStats(0, 0, {}, {}, 0.0, 0.0, 0)
    
    async def get_graph_data(self, collection_name: str = None) -> Dict[str, Any]:
        """Get graph data for visualization"""
        try:
            # Prepare nodes for visualization
            viz_nodes = []
            for node in self.nodes.values():
                viz_nodes.append({
                    "id": node.node_id,
                    "label": node.title,
                    "type": node.node_type,
                    "content": node.content[:200] + "..." if len(node.content) > 200 else node.content,
                    "metadata": node.metadata
                })
            
            # Prepare edges for visualization
            viz_edges = []
            for edge in self.edges.values():
                viz_edges.append({
                    "id": edge.edge_id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type,
                    "weight": edge.weight,
                    "metadata": edge.metadata
                })
            
            # Get graph statistics
            stats = await self.get_graph_stats()
            
            return {
                "nodes": viz_nodes,
                "edges": viz_edges,
                "stats": asdict(stats),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Graph data preparation failed: {e}")
            return {"nodes": [], "edges": [], "stats": {}, "error": str(e)}
    
    async def export_graph(self, format: str = "json") -> str:
        """Export graph in various formats"""
        try:
            if format == "json":
                graph_data = await self.get_graph_data()
                return json.dumps(graph_data, indent=2)
            
            elif format == "gexf":
                # Export as GEXF for Gephi
                return nx.generate_gexf(self.graph)
            
            elif format == "graphml":
                # Export as GraphML
                import io
                buffer = io.StringIO()
                nx.write_graphml(self.graph, buffer)
                return buffer.getvalue()
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Graph export failed: {e}")
            return ""
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up knowledge graph manager")
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()