"""
DocuLLaMA - Azure Container Apps Optimized RAG System
Main FastAPI application with AutoLLaMA features integration
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from config import get_settings, Settings
from monitoring import (
    get_metrics_collector, get_health_checker, get_cost_tracker, get_logger,
    OpenTelemetrySetup, MetricsCollector, HealthChecker, CostTracker
)

# Import components (will be created in subsequent steps)
try:
    from document_processor import DocumentProcessor, ProcessingResult
    from rag_engine import RAGEngine, SearchResult, ChatCompletionRequest, ChatCompletionResponse
    from knowledge_graph import KnowledgeGraphManager
    from research_intelligence import ResearchIntelligenceEngine
except ImportError as e:
    # Graceful fallback for missing components
    DocumentProcessor = None
    RAGEngine = None
    KnowledgeGraphManager = None
    ResearchIntelligenceEngine = None

# Initialize settings and logger
settings = get_settings()
logger = get_logger(__name__)

# Security
security = HTTPBearer(auto_error=False)


# Request/Response Models
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    uptime: float
    checks: Dict[str, Any]
    metrics: Dict[str, Any]


class DocumentUploadResponse(BaseModel):
    """Document upload response model"""
    document_id: str
    filename: str
    status: str
    message: str
    processing_time: float
    chunks_created: int
    metadata: Dict[str, Any]


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    collection_name: Optional[str] = Field(default=None, description="Collection to search in")
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float
    search_type: str


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, ge=1, le=4096, description="Maximum tokens to generate")
    stream: bool = Field(default=False, description="Stream the response")
    collection_name: Optional[str] = Field(default=None, description="RAG collection to use")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class MetricsResponse(BaseModel):
    """Metrics response model"""
    request_count: int
    error_count: int
    average_response_time: float
    active_connections: int
    memory_usage: float
    cpu_usage: float
    disk_usage: float
    cost_tracking: Dict[str, float]


# Global components
document_processor: Optional[DocumentProcessor] = None
rag_engine: Optional[RAGEngine] = None
knowledge_graph: Optional[KnowledgeGraphManager] = None
research_intelligence: Optional[ResearchIntelligenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting DocuLLaMA application")
    
    # Initialize components
    global document_processor, rag_engine, knowledge_graph, research_intelligence
    
    try:
        # Initialize document processor
        if DocumentProcessor:
            document_processor = DocumentProcessor(settings)
            await document_processor.initialize()
            logger.info("Document processor initialized")
        
        # Initialize RAG engine
        if RAGEngine:
            rag_engine = RAGEngine(settings)
            await rag_engine.initialize()
            logger.info("RAG engine initialized")
        
        # Initialize knowledge graph (if enabled)
        if settings.advanced_features.enable_knowledge_graph and KnowledgeGraphManager:
            knowledge_graph = KnowledgeGraphManager(settings)
            await knowledge_graph.initialize()
            logger.info("Knowledge graph initialized")
        
        # Initialize research intelligence (if enabled)
        if settings.advanced_features.enable_research_intelligence and ResearchIntelligenceEngine:
            research_intelligence = ResearchIntelligenceEngine(settings)
            await research_intelligence.initialize()
            logger.info("Research intelligence initialized")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down DocuLLaMA application")
    
    if document_processor:
        await document_processor.cleanup()
    if rag_engine:
        await rag_engine.cleanup()
    if knowledge_graph:
        await knowledge_graph.cleanup()
    if research_intelligence:
        await research_intelligence.cleanup()


# Create FastAPI app
app = FastAPI(
    title="DocuLLaMA",
    description="Azure Container Apps optimized RAG system with AutoLLaMA features",
    version="1.0.0",
    docs_url="/docs" if settings.enable_swagger_ui else None,
    redoc_url="/redoc" if settings.enable_redoc else None,
    lifespan=lifespan
)

# Setup middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Setup OpenTelemetry
OpenTelemetrySetup.setup_telemetry(app, settings)

# Metrics and monitoring
metrics_collector = get_metrics_collector()
health_checker = get_health_checker()
cost_tracker = get_cost_tracker()


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests for monitoring"""
    start_time = time.time()
    metrics_collector.increment_active_connections()
    
    try:
        response = await call_next(request)
        
        # Track metrics
        processing_time = time.time() - start_time
        metrics_collector.record_response_time(processing_time)
        metrics_collector.increment_request_count(
            request.url.path,
            request.method
        )
        
        # Add response headers
        response.headers["X-Processing-Time"] = str(processing_time)
        response.headers["X-Request-ID"] = str(uuid.uuid4())
        
        return response
        
    except Exception as e:
        metrics_collector.increment_error_count(type(e).__name__)
        raise
    finally:
        metrics_collector.decrement_active_connections()


# Authentication dependency
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if not settings.security.enable_api_key_auth:
        return True
    
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    # In production, verify against database or Azure Key Vault
    # For now, simple token verification
    if credentials.credentials != "docullama-api-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


# Health check endpoints (Azure Container Apps required)
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for Azure Container Apps"""
    try:
        health_status = await health_checker.check_health()
        return HealthResponse(
            status=health_status.status,
            timestamp=health_status.timestamp.isoformat(),
            version=health_status.version,
            uptime=health_status.uptime,
            checks=health_status.checks,
            metrics=health_status.metrics
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check endpoint for Azure Container Apps"""
    try:
        readiness_status = await health_checker.check_readiness()
        
        if readiness_status["status"] == "ready":
            return readiness_status
        else:
            return JSONResponse(
                status_code=503,
                content=readiness_status
            )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)}
        )


@app.get("/live", tags=["Health"])
async def liveness_check():
    """Liveness check endpoint for Azure Container Apps"""
    return await health_checker.check_liveness()


# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get application metrics"""
    metrics_data = metrics_collector.get_metrics()
    return MetricsResponse(
        request_count=metrics_data.request_count,
        error_count=metrics_data.error_count,
        average_response_time=metrics_data.average_response_time,
        active_connections=metrics_data.active_connections,
        memory_usage=metrics_data.memory_usage,
        cpu_usage=metrics_data.cpu_usage,
        disk_usage=metrics_data.disk_usage,
        cost_tracking=metrics_data.cost_tracking
    )


# Document processing endpoints
@app.post("/v1/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = None,
    authenticated: bool = Depends(verify_api_key)
):
    """Upload and process a document"""
    if not document_processor:
        raise HTTPException(status_code=501, detail="Document processor not available")
    
    start_time = time.time()
    
    try:
        # Process document
        result = await document_processor.process_document(
            file=file,
            collection_name=collection_name or settings.qdrant.collection_name
        )
        
        processing_time = time.time() - start_time
        
        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            status=result.status,
            message=result.message,
            processing_time=processing_time,
            chunks_created=result.chunks_created,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


# Search endpoints
@app.post("/v1/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(
    request: SearchRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """Search documents in the knowledge base"""
    if not rag_engine:
        raise HTTPException(status_code=501, detail="RAG engine not available")
    
    start_time = time.time()
    
    try:
        results = await rag_engine.search(
            query=request.query,
            max_results=request.max_results,
            collection_name=request.collection_name,
            enable_reranking=request.enable_reranking,
            similarity_threshold=request.similarity_threshold
        )
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            query=request.query,
            results=[result.to_dict() for result in results],
            total_results=len(results),
            processing_time=processing_time,
            search_type="hybrid" if settings.rag.enable_hybrid_search else "vector"
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# OpenAI-compatible chat completion endpoint
@app.post("/v1/chat/completions", tags=["Chat"])
async def chat_completions(
    request: ChatCompletionRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """OpenAI-compatible chat completion endpoint with RAG"""
    if not rag_engine:
        raise HTTPException(status_code=501, detail="RAG engine not available")
    
    try:
        if request.stream:
            return StreamingResponse(
                rag_engine.chat_completion_stream(request),
                media_type="application/json"
            )
        else:
            response = await rag_engine.chat_completion(request)
            
            # Track costs
            if hasattr(response, 'usage'):
                cost_tracker.track_chat_usage(
                    response.usage.get('prompt_tokens', 0),
                    response.usage.get('completion_tokens', 0)
                )
            
            return response
            
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


# Knowledge graph endpoints (if enabled)
if settings.advanced_features.enable_knowledge_graph:
    @app.get("/v1/knowledge-graph", tags=["Knowledge Graph"])
    async def get_knowledge_graph(
        collection_name: Optional[str] = None,
        authenticated: bool = Depends(verify_api_key)
    ):
        """Get knowledge graph representation"""
        if not knowledge_graph:
            raise HTTPException(status_code=501, detail="Knowledge graph not available")
        
        try:
            graph_data = await knowledge_graph.get_graph_data(collection_name)
            return {"graph": graph_data}
        except Exception as e:
            logger.error(f"Knowledge graph retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Research intelligence endpoints (if enabled)
if settings.advanced_features.enable_research_intelligence:
    @app.post("/v1/research/analyze", tags=["Research Intelligence"])
    async def analyze_research_content(
        content: str,
        analysis_type: str = "comprehensive",
        authenticated: bool = Depends(verify_api_key)
    ):
        """Analyze content for research insights"""
        if not research_intelligence:
            raise HTTPException(status_code=501, detail="Research intelligence not available")
        
        try:
            analysis = await research_intelligence.analyze_content(content, analysis_type)
            return {"analysis": analysis}
        except Exception as e:
            logger.error(f"Research analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Collection management endpoints
@app.get("/v1/collections", tags=["Collections"])
async def list_collections(authenticated: bool = Depends(verify_api_key)):
    """List all available collections"""
    if not rag_engine:
        raise HTTPException(status_code=501, detail="RAG engine not available")
    
    try:
        collections = await rag_engine.list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/collections", tags=["Collections"])
async def create_collection(
    name: str,
    description: Optional[str] = None,
    authenticated: bool = Depends(verify_api_key)
):
    """Create a new collection"""
    if not rag_engine:
        raise HTTPException(status_code=501, detail="RAG engine not available")
    
    try:
        result = await rag_engine.create_collection(name, description)
        return {"status": "created", "collection": result}
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/collections/{collection_name}", tags=["Collections"])
async def delete_collection(
    collection_name: str,
    authenticated: bool = Depends(verify_api_key)
):
    """Delete a collection"""
    if not rag_engine:
        raise HTTPException(status_code=501, detail="RAG engine not available")
    
    try:
        await rag_engine.delete_collection(collection_name)
        return {"status": "deleted", "collection_name": collection_name}
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with system information"""
    return {
        "name": "DocuLLaMA",
        "version": "1.0.0",
        "description": "Azure Container Apps optimized RAG system with AutoLLaMA features",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "docs_url": "/docs" if settings.enable_swagger_ui else None,
        "health_url": "/health",
        "metrics_url": "/metrics"
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    metrics_collector.increment_error_count("HTTPException")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    metrics_collector.increment_error_count("GeneralException")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    # For development only
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.reload,
        log_level=settings.monitoring.log_level.lower(),
        workers=1 if settings.reload else settings.workers
    )