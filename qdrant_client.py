"""
DocuLLaMA Qdrant Cloud Integration
Production-ready Qdrant client with JWT authentication and advanced features
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import httpx
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CollectionInfo, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest,
    UpdateStatus, CollectionStatus, PayloadSelector,
    ScoredPoint, Record, CountRequest, CountResult
)
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
import backoff

from config import Settings, get_settings
from monitoring import get_logger
from document_processor import DocumentChunk

logger = get_logger(__name__)


@dataclass
class QdrantConnectionInfo:
    """Qdrant connection information"""
    host: str
    port: int
    api_key: str
    use_https: bool
    url: str
    connected: bool = False
    last_health_check: Optional[datetime] = None
    collections: List[str] = None


@dataclass
class CollectionStats:
    """Collection statistics"""
    name: str
    vectors_count: int
    indexed_vectors_count: int
    points_count: int
    segments_count: int
    disk_usage_bytes: int
    ram_usage_bytes: int
    config: Dict[str, Any]


@dataclass
class QdrantMetrics:
    """Qdrant performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_error: Optional[str] = None
    connection_pool_size: int = 0
    active_connections: int = 0


class QdrantConnectionPool:
    """Connection pool for Qdrant clients"""
    
    def __init__(self, connection_info: QdrantConnectionInfo, pool_size: int = 10):
        self.connection_info = connection_info
        self.pool_size = pool_size
        self.clients = []
        self.available_clients = asyncio.Queue(maxsize=pool_size)
        self.metrics = QdrantMetrics()
        
    async def initialize(self):
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            client = AsyncQdrantClient(
                host=self.connection_info.host,
                port=self.connection_info.port,
                api_key=self.connection_info.api_key,
                https=self.connection_info.use_https,
                timeout=60
            )
            self.clients.append(client)
            await self.available_clients.put(client)
        
        self.metrics.connection_pool_size = self.pool_size
        logger.info(f"Initialized Qdrant connection pool with {self.pool_size} clients")
    
    async def get_client(self) -> AsyncQdrantClient:
        """Get client from pool"""
        client = await self.available_clients.get()
        self.metrics.active_connections += 1
        return client
    
    async def return_client(self, client: AsyncQdrantClient):
        """Return client to pool"""
        await self.available_clients.put(client)
        self.metrics.active_connections -= 1
    
    async def close_all(self):
        """Close all connections"""
        for client in self.clients:
            await client.close()
        logger.info("Closed all Qdrant connections")


class QdrantCloudClient:
    """Production-ready Qdrant Cloud client with advanced features"""
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()
        
        # Connection configuration
        self.connection_info = QdrantConnectionInfo(
            host=self.settings.qdrant.host,
            port=self.settings.qdrant.port,
            api_key=self.settings.qdrant.api_key,
            use_https=self.settings.qdrant.use_https,
            url=f"{'https' if self.settings.qdrant.use_https else 'http'}://{self.settings.qdrant.host}:{self.settings.qdrant.port}"
        )
        
        # Initialize components
        self.connection_pool = None
        self.client = None  # Fallback sync client
        self.metrics = QdrantMetrics()
        
        # Configuration
        self.default_collection = self.settings.qdrant.collection_name
        self.vector_dimension = self.settings.azure_openai.embedding_dimensions
        self.batch_size = 100
        self.retry_attempts = 3
        self.timeout = 60
        
        # Environment-based collection naming
        self.environment = self.settings.environment.lower()
        self.collection_prefix = f"{self.environment}_" if self.environment != "production" else ""
        
        # Health check tracking
        self.last_health_check = None
        self.health_check_interval = timedelta(minutes=5)
        
    async def initialize(self):
        """Initialize Qdrant client and connection pool"""
        logger.info("Initializing Qdrant Cloud client")
        
        try:
            # Initialize connection pool
            self.connection_pool = QdrantConnectionPool(
                self.connection_info,
                pool_size=10
            )
            await self.connection_pool.initialize()
            
            # Initialize fallback sync client
            self.client = QdrantClient(
                host=self.connection_info.host,
                port=self.connection_info.port,
                api_key=self.connection_info.api_key,
                https=self.connection_info.use_https,
                timeout=self.timeout
            )
            
            # Test connection
            await self.health_check()
            
            # Ensure default collection exists
            await self.ensure_collection_exists(self.default_collection)
            
            logger.info("Qdrant Cloud client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (ConnectionError, TimeoutError, ResponseHandlingException),
        max_tries=3,
        max_time=60
    )
    async def health_check(self) -> bool:
        """Perform health check with retry logic"""
        try:
            client = await self.connection_pool.get_client()
            start_time = time.time()
            
            try:
                # Test basic connectivity
                collections = await client.get_collections()
                
                # Update metrics
                response_time = time.time() - start_time
                self.metrics.successful_requests += 1
                self.metrics.total_requests += 1
                self.metrics.average_response_time = (
                    (self.metrics.average_response_time * (self.metrics.total_requests - 1) + response_time)
                    / self.metrics.total_requests
                )
                
                # Update connection info
                self.connection_info.connected = True
                self.connection_info.last_health_check = datetime.utcnow()
                self.connection_info.collections = [col.name for col in collections.collections]
                
                logger.debug(f"Qdrant health check successful: {len(collections.collections)} collections")
                return True
                
            finally:
                await self.connection_pool.return_client(client)
                
        except Exception as e:
            self.metrics.failed_requests += 1
            self.metrics.total_requests += 1
            self.metrics.last_error = str(e)
            self.connection_info.connected = False
            
            logger.error(f"Qdrant health check failed: {e}")
            raise
    
    async def get_collection_name(self, base_name: str) -> str:
        """Get environment-prefixed collection name"""
        return f"{self.collection_prefix}{base_name}"
    
    async def ensure_collection_exists(
        self,
        collection_name: str,
        vector_size: int = None,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """Ensure collection exists, create if not"""
        full_collection_name = await self.get_collection_name(collection_name)
        vector_size = vector_size or self.vector_dimension
        
        try:
            client = await self.connection_pool.get_client()
            
            try:
                # Check if collection exists
                await client.get_collection(full_collection_name)
                logger.debug(f"Collection {full_collection_name} already exists")
                return True
                
            except UnexpectedResponse as e:
                if "doesn't exist" in str(e) or "Not found" in str(e):
                    # Collection doesn't exist, create it
                    await client.create_collection(
                        collection_name=full_collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=distance
                        )
                    )
                    logger.info(f"Created collection {full_collection_name}")
                    return True
                else:
                    raise
            finally:
                await self.connection_pool.return_client(client)
                
        except Exception as e:
            logger.error(f"Failed to ensure collection {full_collection_name} exists: {e}")
            raise
    
    async def upsert_documents(
        self,
        chunks: List[DocumentChunk],
        collection_name: str = None
    ) -> bool:
        """Upsert document chunks with batch processing"""
        if not chunks:
            return True
        
        collection_name = collection_name or self.default_collection
        full_collection_name = await self.get_collection_name(collection_name)
        
        # Ensure collection exists
        await self.ensure_collection_exists(collection_name)
        
        try:
            client = await self.connection_pool.get_client()
            
            try:
                # Process in batches
                total_processed = 0
                for i in range(0, len(chunks), self.batch_size):
                    batch = chunks[i:i + self.batch_size]
                    
                    # Prepare points
                    points = []
                    for chunk in batch:
                        if not chunk.embeddings:
                            logger.warning(f"Chunk {chunk.chunk_id} has no embeddings, skipping")
                            continue
                        
                        point = PointStruct(
                            id=chunk.chunk_id,
                            vector=chunk.embeddings,
                            payload={
                                "content": chunk.content,
                                "chunk_type": chunk.chunk_type,
                                "position": chunk.position,
                                "metadata": chunk.metadata,
                                "created_at": datetime.utcnow().isoformat(),
                                "document_id": chunk.metadata.get("document_id"),
                                "filename": chunk.metadata.get("filename"),
                                "source_hash": chunk.metadata.get("source_hash")
                            }
                        )
                        points.append(point)
                    
                    if points:
                        # Upsert batch
                        result = await client.upsert(
                            collection_name=full_collection_name,
                            points=points
                        )
                        
                        if result.status == UpdateStatus.COMPLETED:
                            total_processed += len(points)
                            logger.debug(f"Upserted batch of {len(points)} points to {full_collection_name}")
                        else:
                            logger.warning(f"Batch upsert incomplete: {result.status}")
                
                logger.info(f"Successfully upserted {total_processed} chunks to {full_collection_name}")
                return True
                
            finally:
                await self.connection_pool.return_client(client)
                
        except Exception as e:
            logger.error(f"Failed to upsert documents to {full_collection_name}: {e}")
            raise
    
    async def search_vectors(
        self,
        query_vector: List[float],
        collection_name: str = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Filter] = None,
        payload_selector: Optional[PayloadSelector] = None
    ) -> List[ScoredPoint]:
        """Search vectors with advanced filtering"""
        collection_name = collection_name or self.default_collection
        full_collection_name = await self.get_collection_name(collection_name)
        
        try:
            client = await self.connection_pool.get_client()
            
            try:
                search_request = SearchRequest(
                    vector=query_vector,
                    filter=filter_conditions,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=payload_selector or True,
                    with_vector=False
                )
                
                results = await client.search(
                    collection_name=full_collection_name,
                    **search_request.dict(exclude_none=True)
                )
                
                logger.debug(f"Vector search returned {len(results)} results from {full_collection_name}")
                return results
                
            finally:
                await self.connection_pool.return_client(client)
                
        except Exception as e:
            logger.error(f"Vector search failed in {full_collection_name}: {e}")
            raise
    
    async def delete_points(
        self,
        point_ids: List[str],
        collection_name: str = None
    ) -> bool:
        """Delete points by IDs"""
        collection_name = collection_name or self.default_collection
        full_collection_name = await self.get_collection_name(collection_name)
        
        if not point_ids:
            return True
        
        try:
            client = await self.connection_pool.get_client()
            
            try:
                result = await client.delete(
                    collection_name=full_collection_name,
                    points_selector=point_ids
                )
                
                if result.status == UpdateStatus.COMPLETED:
                    logger.info(f"Deleted {len(point_ids)} points from {full_collection_name}")
                    return True
                else:
                    logger.warning(f"Point deletion incomplete: {result.status}")
                    return False
                    
            finally:
                await self.connection_pool.return_client(client)
                
        except Exception as e:
            logger.error(f"Failed to delete points from {full_collection_name}: {e}")
            raise
    
    async def get_collection_stats(self, collection_name: str = None) -> CollectionStats:
        """Get detailed collection statistics"""
        collection_name = collection_name or self.default_collection
        full_collection_name = await self.get_collection_name(collection_name)
        
        try:
            client = await self.connection_pool.get_client()
            
            try:
                # Get collection info
                collection_info = await client.get_collection(full_collection_name)
                
                return CollectionStats(
                    name=full_collection_name,
                    vectors_count=collection_info.vectors_count or 0,
                    indexed_vectors_count=collection_info.indexed_vectors_count or 0,
                    points_count=collection_info.points_count or 0,
                    segments_count=collection_info.segments_count or 0,
                    disk_usage_bytes=collection_info.disk_data_size or 0,
                    ram_usage_bytes=collection_info.ram_data_size or 0,
                    config=collection_info.config.dict() if collection_info.config else {}
                )
                
            finally:
                await self.connection_pool.return_client(client)
                
        except Exception as e:
            logger.error(f"Failed to get stats for {full_collection_name}: {e}")
            raise
    
    async def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            client = await self.connection_pool.get_client()
            
            try:
                collections = await client.get_collections()
                collection_names = [col.name for col in collections.collections]
                
                # Filter by environment prefix if not production
                if self.collection_prefix:
                    collection_names = [
                        name[len(self.collection_prefix):] 
                        for name in collection_names 
                        if name.startswith(self.collection_prefix)
                    ]
                
                return collection_names
                
            finally:
                await self.connection_pool.return_client(client)
                
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = None,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """Create a new collection"""
        return await self.ensure_collection_exists(collection_name, vector_size, distance)
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        full_collection_name = await self.get_collection_name(collection_name)
        
        try:
            client = await self.connection_pool.get_client()
            
            try:
                result = await client.delete_collection(full_collection_name)
                logger.info(f"Deleted collection {full_collection_name}")
                return True
                
            finally:
                await self.connection_pool.return_client(client)
                
        except Exception as e:
            logger.error(f"Failed to delete collection {full_collection_name}: {e}")
            raise
    
    async def count_points(
        self,
        collection_name: str = None,
        filter_conditions: Optional[Filter] = None
    ) -> int:
        """Count points in collection with optional filter"""
        collection_name = collection_name or self.default_collection
        full_collection_name = await self.get_collection_name(collection_name)
        
        try:
            client = await self.connection_pool.get_client()
            
            try:
                count_request = CountRequest(
                    filter=filter_conditions,
                    exact=True
                )
                
                result = await client.count(
                    collection_name=full_collection_name,
                    **count_request.dict(exclude_none=True)
                )
                
                return result.count
                
            finally:
                await self.connection_pool.return_client(client)
                
        except Exception as e:
            logger.error(f"Failed to count points in {full_collection_name}: {e}")
            raise
    
    async def get_points(
        self,
        point_ids: List[str],
        collection_name: str = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[Record]:
        """Retrieve specific points by ID"""
        collection_name = collection_name or self.default_collection
        full_collection_name = await self.get_collection_name(collection_name)
        
        try:
            client = await self.connection_pool.get_client()
            
            try:
                results = await client.retrieve(
                    collection_name=full_collection_name,
                    ids=point_ids,
                    with_payload=with_payload,
                    with_vectors=with_vectors
                )
                
                return results
                
            finally:
                await self.connection_pool.return_client(client)
                
        except Exception as e:
            logger.error(f"Failed to retrieve points from {full_collection_name}: {e}")
            raise
    
    async def backup_collection(self, collection_name: str, backup_location: str) -> bool:
        """Create collection backup (placeholder for future implementation)"""
        # TODO: Implement collection backup to Azure Blob Storage
        logger.info(f"Backup requested for collection {collection_name} to {backup_location}")
        return True
    
    def get_metrics(self) -> QdrantMetrics:
        """Get client metrics"""
        if self.connection_pool:
            self.metrics.connection_pool_size = self.connection_pool.pool_size
            self.metrics.active_connections = self.connection_pool.metrics.active_connections
        
        return self.metrics
    
    def get_connection_info(self) -> QdrantConnectionInfo:
        """Get connection information"""
        return self.connection_info
    
    async def close(self):
        """Close all connections"""
        if self.connection_pool:
            await self.connection_pool.close_all()
        
        if self.client:
            self.client.close()
        
        logger.info("Qdrant client closed")


# Global client instance
qdrant_client: Optional[QdrantCloudClient] = None


async def get_qdrant_client() -> QdrantCloudClient:
    """Get global Qdrant client instance"""
    global qdrant_client
    
    if qdrant_client is None:
        qdrant_client = QdrantCloudClient()
        await qdrant_client.initialize()
    
    return qdrant_client


async def initialize_qdrant_client(settings: Settings = None) -> QdrantCloudClient:
    """Initialize and return Qdrant client"""
    client = QdrantCloudClient(settings)
    await client.initialize()
    return client