"""
DocuLLaMA RAG Engine
Advanced RAG system with Qdrant, Azure OpenAI, and AutoLLaMA features
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, asdict
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CollectionInfo, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)
from openai import AzureOpenAI
from rank_bm25 import BM25Okapi
import tiktoken

from config import Settings, get_settings
from monitoring import get_logger, get_cost_tracker
from document_processor import DocumentChunk

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Search result data structure"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_document: str
    chunk_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ChatMessage:
    """Chat message data structure"""
    role: str
    content: str


@dataclass
class ChatCompletionRequest:
    """Chat completion request"""
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.3
    max_tokens: int = 2048
    stream: bool = False
    collection_name: Optional[str] = None


@dataclass
class ChatCompletionResponse:
    """Chat completion response"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class EmbeddingEngine:
    """Azure OpenAI embedding engine"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AzureOpenAI(
            api_key=settings.azure_openai.api_key,
            api_version=settings.azure_openai.api_version,
            azure_endpoint=settings.azure_openai.endpoint
        )
        
        self.embedding_model = settings.azure_openai.embedding_model
        self.embedding_dimensions = settings.azure_openai.embedding_dimensions
        
        # Token counting
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Cost tracking
        self.cost_tracker = get_cost_tracker()
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            # Count tokens for cost tracking
            tokens = len(self.encoding.encode(text))
            
            # Generate embedding
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=self.embedding_model,
                input=text,
                dimensions=self.embedding_dimensions
            )
            
            # Track costs
            self.cost_tracker.track_embedding_usage(tokens)
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Count total tokens
                total_tokens = sum(len(self.encoding.encode(text)) for text in batch)
                
                # Generate embeddings
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    model=self.embedding_model,
                    input=batch,
                    dimensions=self.embedding_dimensions
                )
                
                # Track costs
                self.cost_tracker.track_embedding_usage(total_tokens)
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Batch embedding failed for batch {i}: {e}")
                # Add zero embeddings for failed batch
                embeddings.extend([[0.0] * self.embedding_dimensions] * len(batch))
        
        return embeddings


class HybridSearchEngine:
    """Hybrid search engine combining vector and keyword search"""
    
    def __init__(self, qdrant_client: QdrantClient, settings: Settings):
        self.qdrant_client = qdrant_client
        self.settings = settings
        self.bm25_indices = {}  # Collection name -> BM25Okapi instance
        
    async def index_documents(self, collection_name: str, chunks: List[DocumentChunk]):
        """Index documents for hybrid search"""
        # Prepare documents for BM25
        documents = [chunk.content for chunk in chunks]
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Create BM25 index
        self.bm25_indices[collection_name] = BM25Okapi(tokenized_docs)
        
        logger.info(f"Indexed {len(chunks)} documents for hybrid search in {collection_name}")
    
    async def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        alpha: float = 0.7  # Weight for vector search vs BM25
    ) -> List[SearchResult]:
        """Perform hybrid search"""
        results = []
        
        # Vector search
        vector_results = await self._vector_search(query, collection_name, top_k * 2)
        
        # BM25 search
        bm25_results = await self._bm25_search(query, collection_name, top_k * 2)
        
        # Combine and re-rank results
        combined_results = self._combine_results(vector_results, bm25_results, alpha)
        
        return combined_results[:top_k]
    
    async def _vector_search(self, query: str, collection_name: str, top_k: int) -> List[SearchResult]:
        """Perform vector similarity search"""
        try:
            # This would be implemented with the embedding engine and Qdrant
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _bm25_search(self, query: str, collection_name: str, top_k: int) -> List[SearchResult]:
        """Perform BM25 keyword search"""
        if collection_name not in self.bm25_indices:
            return []
        
        try:
            bm25 = self.bm25_indices[collection_name]
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = bm25.get_scores(query_tokens)
            
            # Get top results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    # This would fetch the actual document from Qdrant
                    # For now, create placeholder result
                    result = SearchResult(
                        chunk_id=f"bm25_{idx}",
                        content=f"BM25 result {idx}",
                        score=float(scores[idx]),
                        metadata={"search_type": "bm25"},
                        source_document="unknown",
                        chunk_type="content"
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _combine_results(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        alpha: float
    ) -> List[SearchResult]:
        """Combine vector and BM25 results"""
        # Normalize scores
        if vector_results:
            max_vector_score = max(r.score for r in vector_results)
            for result in vector_results:
                result.score = result.score / max_vector_score if max_vector_score > 0 else 0
        
        if bm25_results:
            max_bm25_score = max(r.score for r in bm25_results)
            for result in bm25_results:
                result.score = result.score / max_bm25_score if max_bm25_score > 0 else 0
        
        # Combine results with weights
        combined = {}
        
        # Add vector results
        for result in vector_results:
            combined[result.chunk_id] = result
            combined[result.chunk_id].score *= alpha
        
        # Add BM25 results
        for result in bm25_results:
            if result.chunk_id in combined:
                # Combine scores
                combined[result.chunk_id].score += (1 - alpha) * result.score
            else:
                result.score *= (1 - alpha)
                combined[result.chunk_id] = result
        
        # Sort by combined score
        sorted_results = sorted(combined.values(), key=lambda x: x.score, reverse=True)
        
        return sorted_results


class RAGEngine:
    """Main RAG engine with AutoLLaMA features"""
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()
        
        # Initialize components
        self.qdrant_client = None
        self.embedding_engine = None
        self.hybrid_search = None
        self.azure_openai = None
        
        # Chat model settings
        self.chat_model = settings.azure_openai.chat_model
        self.max_context_length = settings.rag.max_context_length
        self.response_temperature = settings.rag.response_temperature
        self.response_max_tokens = settings.rag.response_max_tokens
        
        # Cost tracking
        self.cost_tracker = get_cost_tracker()
        
        # Token encoding for context management
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    async def initialize(self):
        """Initialize RAG engine components"""
        logger.info("Initializing RAG engine")
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=self.settings.qdrant.host,
            port=self.settings.qdrant.port,
            api_key=self.settings.qdrant.api_key,
            https=self.settings.qdrant.use_https,
            timeout=60
        )
        
        # Test connection
        try:
            collections = await asyncio.to_thread(self.qdrant_client.get_collections)
            logger.info(f"Connected to Qdrant with {len(collections.collections)} collections")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        # Initialize embedding engine
        self.embedding_engine = EmbeddingEngine(self.settings)
        
        # Initialize hybrid search
        if self.settings.rag.enable_hybrid_search:
            self.hybrid_search = HybridSearchEngine(self.qdrant_client, self.settings)
            logger.info("Hybrid search enabled")
        
        # Initialize Azure OpenAI client
        self.azure_openai = AzureOpenAI(
            api_key=self.settings.azure_openai.api_key,
            api_version=self.settings.azure_openai.api_version,
            azure_endpoint=self.settings.azure_openai.endpoint
        )
        
        # Ensure default collection exists
        await self._ensure_collection_exists(self.settings.qdrant.collection_name)
        
        logger.info("RAG engine initialized successfully")
    
    async def _ensure_collection_exists(self, collection_name: str):
        """Ensure collection exists in Qdrant"""
        try:
            await asyncio.to_thread(
                self.qdrant_client.get_collection,
                collection_name
            )
            logger.info(f"Collection {collection_name} exists")
        except Exception:
            # Collection doesn't exist, create it
            await asyncio.to_thread(
                self.qdrant_client.create_collection,
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.settings.azure_openai.embedding_dimensions,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection {collection_name}")
    
    async def add_documents(self, chunks: List[DocumentChunk], collection_name: str = None):
        """Add document chunks to the knowledge base"""
        if not collection_name:
            collection_name = self.settings.qdrant.collection_name
        
        if not chunks:
            return
        
        # Ensure collection exists
        await self._ensure_collection_exists(collection_name)
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_engine.embed_batch(texts)
        
        # Prepare points for Qdrant
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type,
                    "position": chunk.position,
                    "metadata": chunk.metadata
                }
            )
            points.append(point)
        
        # Insert into Qdrant
        await asyncio.to_thread(
            self.qdrant_client.upsert,
            collection_name=collection_name,
            points=points
        )
        
        # Update hybrid search index
        if self.hybrid_search:
            await self.hybrid_search.index_documents(collection_name, chunks)
        
        logger.info(f"Added {len(chunks)} chunks to collection {collection_name}")
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        collection_name: str = None,
        enable_reranking: bool = True,
        similarity_threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search the knowledge base"""
        if not collection_name:
            collection_name = self.settings.qdrant.collection_name
        
        try:
            if self.settings.rag.enable_hybrid_search and self.hybrid_search:
                # Use hybrid search
                results = await self.hybrid_search.search(
                    query,
                    collection_name,
                    max_results
                )
            else:
                # Use vector search only
                results = await self._vector_search(
                    query,
                    collection_name,
                    max_results,
                    similarity_threshold
                )
            
            # Apply re-ranking if enabled
            if enable_reranking and self.settings.rag.enable_reranking:
                results = await self._rerank_results(query, results)
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _vector_search(
        self,
        query: str,
        collection_name: str,
        max_results: int,
        similarity_threshold: float
    ) -> List[SearchResult]:
        """Perform vector similarity search"""
        # Generate query embedding
        query_embedding = await self.embedding_engine.embed_text(query)
        
        # Search Qdrant
        search_results = await asyncio.to_thread(
            self.qdrant_client.search,
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=max_results,
            score_threshold=similarity_threshold
        )
        
        # Convert to SearchResult objects
        results = []
        for result in search_results:
            search_result = SearchResult(
                chunk_id=str(result.id),
                content=result.payload.get("content", ""),
                score=result.score,
                metadata=result.payload.get("metadata", {}),
                source_document=result.payload.get("metadata", {}).get("filename", "unknown"),
                chunk_type=result.payload.get("chunk_type", "content")
            )
            results.append(search_result)
        
        return results
    
    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank search results using cross-encoder"""
        # TODO: Implement cross-encoder re-ranking
        # For now, return results as-is
        return results
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate chat completion with RAG"""
        start_time = time.time()
        
        # Extract user query from messages
        user_query = None
        for message in reversed(request.messages):
            if message.role == "user":
                user_query = message.content
                break
        
        if not user_query:
            raise ValueError("No user message found")
        
        # Search for relevant context
        context_results = await self.search(
            query=user_query,
            max_results=self.settings.rag.max_search_results,
            collection_name=request.collection_name
        )
        
        # Build context
        context = self._build_context(context_results)
        
        # Create system message with context
        system_message = self._create_system_message(context)
        
        # Prepare messages for OpenAI
        messages = [system_message] + [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        # Truncate messages to fit context window
        messages = self._truncate_messages(messages, request.max_tokens)
        
        # Generate response
        try:
            response = await asyncio.to_thread(
                self.azure_openai.chat.completions.create,
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Track costs
            if hasattr(response, 'usage') and response.usage:
                self.cost_tracker.track_chat_usage(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            # Convert to our response format
            chat_response = ChatCompletionResponse(
                id=f"docullama-{uuid.uuid4()}",
                object="chat.completion",
                created=int(start_time),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ],
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            )
            
            return chat_response
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion with RAG"""
        # TODO: Implement streaming response
        # For now, fall back to non-streaming
        response = await self.chat_completion(request)
        content = response.choices[0]["message"]["content"]
        
        # Yield content in chunks
        chunk_size = 50
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context from search results"""
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.source_document
            content = result.content[:500]  # Truncate long content
            
            context_parts.append(f"[Source {i}: {source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _create_system_message(self, context: str) -> Dict[str, str]:
        """Create system message with context"""
        system_prompt = f"""You are DocuLLaMA, an AI assistant that answers questions based on the provided context.

Use the following context to answer the user's question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Instructions:
- Answer based primarily on the provided context
- If you need to use information not in the context, clearly indicate that
- Provide specific citations when referencing the context
- Be concise but comprehensive
- If the question cannot be answered from the context, say so clearly"""

        return {"role": "system", "content": system_prompt}
    
    def _truncate_messages(self, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Truncate messages to fit context window"""
        # Calculate available tokens for messages
        available_tokens = self.max_context_length - max_tokens - 100  # Buffer
        
        # Count tokens and truncate if necessary
        total_tokens = 0
        truncated_messages = []
        
        for message in reversed(messages):
            message_tokens = len(self.encoding.encode(message["content"]))
            
            if total_tokens + message_tokens <= available_tokens:
                truncated_messages.insert(0, message)
                total_tokens += message_tokens
            else:
                # Truncate this message if it's not the system message
                if message["role"] != "system":
                    remaining_tokens = available_tokens - total_tokens
                    if remaining_tokens > 100:  # Only truncate if we have reasonable space
                        truncated_content = self._truncate_text(message["content"], remaining_tokens)
                        truncated_message = {**message, "content": truncated_content}
                        truncated_messages.insert(0, truncated_message)
                break
        
        return truncated_messages
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    async def list_collections(self) -> List[str]:
        """List all available collections"""
        try:
            collections = await asyncio.to_thread(self.qdrant_client.get_collections)
            return [collection.name for collection in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    async def create_collection(self, name: str, description: str = None) -> Dict[str, Any]:
        """Create a new collection"""
        try:
            await asyncio.to_thread(
                self.qdrant_client.create_collection,
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.settings.azure_openai.embedding_dimensions,
                    distance=Distance.COSINE
                )
            )
            
            return {
                "name": name,
                "description": description,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            raise
    
    async def delete_collection(self, name: str):
        """Delete a collection"""
        try:
            await asyncio.to_thread(self.qdrant_client.delete_collection, name)
            
            # Remove from hybrid search indices
            if self.hybrid_search and name in self.hybrid_search.bm25_indices:
                del self.hybrid_search.bm25_indices[name]
            
            logger.info(f"Deleted collection {name}")
            
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up RAG engine")
        if self.qdrant_client:
            self.qdrant_client.close()