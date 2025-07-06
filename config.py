"""
DocuLLaMA Configuration Module
Azure Container Apps optimized configuration management
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration"""
    endpoint: str = Field(..., env='AZURE_OPENAI_ENDPOINT')
    api_key: str = Field(..., env='AZURE_OPENAI_API_KEY')
    api_version: str = Field(default='2024-02-01', env='AZURE_OPENAI_API_VERSION')
    chat_model: str = Field(default='gpt-4o-mini', env='AZURE_OPENAI_CHAT_MODEL')
    embedding_model: str = Field(default='text-embedding-3-small', env='AZURE_OPENAI_EMBEDDING_MODEL')
    embedding_dimensions: int = Field(default=768, env='AZURE_OPENAI_EMBEDDING_DIMENSIONS')


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration"""
    host: str = Field(default='localhost', env='QDRANT_HOST')
    port: int = Field(default=6333, env='QDRANT_PORT')
    api_key: Optional[str] = Field(default=None, env='QDRANT_API_KEY')
    collection_name: str = Field(default='docullama_documents', env='QDRANT_COLLECTION_NAME')
    use_https: bool = Field(default=False, env='QDRANT_USE_HTTPS')


class AzureConfig(BaseModel):
    """Azure services configuration"""
    tenant_id: Optional[str] = Field(default=None, env='AZURE_TENANT_ID')
    client_id: Optional[str] = Field(default=None, env='AZURE_CLIENT_ID')
    client_secret: Optional[str] = Field(default=None, env='AZURE_CLIENT_SECRET')
    keyvault_url: Optional[str] = Field(default=None, env='AZURE_KEYVAULT_URL')
    storage_account_name: Optional[str] = Field(default=None, env='AZURE_STORAGE_ACCOUNT_NAME')
    storage_container_name: str = Field(default='docullama-documents', env='AZURE_STORAGE_CONTAINER_NAME')


class DocumentProcessingConfig(BaseModel):
    """Document processing configuration"""
    tika_server_url: str = Field(default='http://localhost:9998', env='TIKA_SERVER_URL')
    max_file_size_mb: int = Field(default=100, env='MAX_FILE_SIZE_MB')
    chunk_size: int = Field(default=512, env='CHUNK_SIZE')
    chunk_overlap: int = Field(default=50, env='CHUNK_OVERLAP')
    min_chunk_size: int = Field(default=50, env='MIN_CHUNK_SIZE')
    quality_threshold: float = Field(default=0.6, env='QUALITY_THRESHOLD')
    enable_ocr: bool = Field(default=True, env='ENABLE_OCR')
    supported_formats: List[str] = Field(
        default=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt', 'txt', 'md', 'py', 'js', 'html', 'xml', 'json', 'csv'],
        env='SUPPORTED_FORMATS'
    )


class RAGConfig(BaseModel):
    """RAG (Retrieval-Augmented Generation) configuration"""
    max_context_length: int = Field(default=8192, env='MAX_CONTEXT_LENGTH')
    max_search_results: int = Field(default=10, env='MAX_SEARCH_RESULTS')
    rerank_top_k: int = Field(default=5, env='RERANK_TOP_K')
    enable_hybrid_search: bool = Field(default=True, env='ENABLE_HYBRID_SEARCH')
    enable_reranking: bool = Field(default=True, env='ENABLE_RERANKING')
    similarity_threshold: float = Field(default=0.7, env='SIMILARITY_THRESHOLD')
    response_temperature: float = Field(default=0.3, env='RESPONSE_TEMPERATURE')
    response_max_tokens: int = Field(default=2048, env='RESPONSE_MAX_TOKENS')


class CacheConfig(BaseModel):
    """Caching and performance configuration"""
    redis_url: str = Field(default='redis://localhost:6379/0', env='REDIS_URL')
    cache_ttl_seconds: int = Field(default=3600, env='CACHE_TTL_SECONDS')
    enable_response_caching: bool = Field(default=True, env='ENABLE_RESPONSE_CACHING')
    enable_embedding_caching: bool = Field(default=True, env='ENABLE_EMBEDDING_CACHING')
    max_concurrent_requests: int = Field(default=100, env='MAX_CONCURRENT_REQUESTS')
    rate_limit_per_minute: int = Field(default=60, env='RATE_LIMIT_PER_MINUTE')


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration"""
    applicationinsights_connection_string: Optional[str] = Field(default=None, env='APPLICATIONINSIGHTS_CONNECTION_STRING')
    log_level: str = Field(default='INFO', env='LOG_LEVEL')
    enable_structured_logging: bool = Field(default=True, env='ENABLE_STRUCTURED_LOGGING')
    enable_opentelemetry: bool = Field(default=True, env='ENABLE_OPENTELEMETRY')
    metrics_export_interval: int = Field(default=30, env='METRICS_EXPORT_INTERVAL')
    enable_cost_tracking: bool = Field(default=True, env='ENABLE_COST_TRACKING')


class SecurityConfig(BaseModel):
    """Security and authentication configuration"""
    jwt_secret_key: str = Field(..., env='JWT_SECRET_KEY')
    jwt_algorithm: str = Field(default='HS256', env='JWT_ALGORITHM')
    jwt_expiration_hours: int = Field(default=24, env='JWT_EXPIRATION_HOURS')
    enable_api_key_auth: bool = Field(default=True, env='ENABLE_API_KEY_AUTH')
    api_key_header_name: str = Field(default='X-API-Key', env='API_KEY_HEADER_NAME')
    cors_origins: List[str] = Field(default=['*'], env='CORS_ORIGINS')


class HealthCheckConfig(BaseModel):
    """Health check and monitoring configuration"""
    health_check_timeout: int = Field(default=10, env='HEALTH_CHECK_TIMEOUT')
    readiness_check_dependencies: List[str] = Field(
        default=['qdrant', 'azure_openai', 'tika'],
        env='READINESS_CHECK_DEPENDENCIES'
    )
    enable_metrics_endpoint: bool = Field(default=True, env='ENABLE_METRICS_ENDPOINT')
    metrics_port: int = Field(default=8001, env='METRICS_PORT')


class AdvancedFeaturesConfig(BaseModel):
    """Advanced features configuration (AutoLLaMA integration)"""
    enable_research_intelligence: bool = Field(default=True, env='ENABLE_RESEARCH_INTELLIGENCE')
    enable_knowledge_graph: bool = Field(default=True, env='ENABLE_KNOWLEDGE_GRAPH')
    enable_multi_source_rag: bool = Field(default=True, env='ENABLE_MULTI_SOURCE_RAG')
    enable_auto_discovery: bool = Field(default=False, env='ENABLE_AUTO_DISCOVERY')
    research_temperature: float = Field(default=0.6, env='RESEARCH_TEMPERATURE')
    max_research_depth: int = Field(default=10, env='MAX_RESEARCH_DEPTH')
    enable_citation_formatting: bool = Field(default=True, env='ENABLE_CITATION_FORMATTING')
    citation_styles: List[str] = Field(default=['mla', 'apa', 'chicago'], env='CITATION_STYLES')


class Settings(BaseSettings):
    """Main application settings"""
    
    # Azure Container Apps required
    port: int = Field(default=8000, env='PORT')
    workers: int = Field(default=4, env='WORKERS')
    timeout: int = Field(default=300, env='TIMEOUT')
    environment: str = Field(default='production', env='ENVIRONMENT')
    
    # Debug and development
    debug: bool = Field(default=False, env='DEBUG')
    reload: bool = Field(default=False, env='RELOAD')
    enable_swagger_ui: bool = Field(default=True, env='ENABLE_SWAGGER_UI')
    enable_redoc: bool = Field(default=True, env='ENABLE_REDOC')
    profiling_enabled: bool = Field(default=False, env='PROFILING_ENABLED')
    
    # Configuration sections
    azure_openai: AzureOpenAIConfig = Field(default_factory=AzureOpenAIConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    azure: AzureConfig = Field(default_factory=AzureConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    advanced_features: AdvancedFeaturesConfig = Field(default_factory=AdvancedFeaturesConfig)
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = 'ignore'


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def is_production() -> bool:
    """Check if running in production environment"""
    return settings.environment.lower() == 'production'


def is_development() -> bool:
    """Check if running in development environment"""
    return settings.environment.lower() in ['development', 'dev']


def get_version() -> str:
    """Get application version"""
    return os.getenv('APP_VERSION', '1.0.0')