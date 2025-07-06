"""
DocuLLaMA Monitoring and Health Check Module
Azure Container Apps optimized monitoring and observability
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from fastapi import HTTPException
import structlog
import httpx
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from config import get_settings

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


@dataclass
class HealthStatus:
    """Health status data class"""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    checks: Dict[str, Any]
    metrics: Dict[str, Any]


@dataclass
class MetricsData:
    """Metrics data class"""
    request_count: int
    error_count: int
    average_response_time: float
    active_connections: int
    memory_usage: float
    cpu_usage: float
    disk_usage: float
    cost_tracking: Dict[str, float]


class CostTracker:
    """Track Azure OpenAI API costs"""
    
    def __init__(self):
        self.costs = {
            'embedding_tokens': 0,
            'chat_tokens': 0,
            'total_cost': 0.0
        }
        self.pricing = {
            'text-embedding-3-small': 0.00002,  # per 1K tokens
            'gpt-4o-mini': 0.00015,  # per 1K tokens (input)
            'gpt-4o-mini-output': 0.0006,  # per 1K tokens (output)
        }
    
    def track_embedding_usage(self, tokens: int):
        """Track embedding token usage"""
        self.costs['embedding_tokens'] += tokens
        cost = (tokens / 1000) * self.pricing['text-embedding-3-small']
        self.costs['total_cost'] += cost
    
    def track_chat_usage(self, input_tokens: int, output_tokens: int):
        """Track chat completion token usage"""
        self.costs['chat_tokens'] += input_tokens + output_tokens
        input_cost = (input_tokens / 1000) * self.pricing['gpt-4o-mini']
        output_cost = (output_tokens / 1000) * self.pricing['gpt-4o-mini-output']
        self.costs['total_cost'] += input_cost + output_cost
    
    def get_costs(self) -> Dict[str, float]:
        """Get current cost tracking data"""
        return self.costs.copy()
    
    def reset_costs(self):
        """Reset cost tracking"""
        self.costs = {
            'embedding_tokens': 0,
            'chat_tokens': 0,
            'total_cost': 0.0
        }


class MetricsCollector:
    """Collect and track application metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.active_connections = 0
        self.start_time = time.time()
        self.cost_tracker = CostTracker()
        
        # OpenTelemetry setup
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Metrics instruments
        self.request_counter = self.meter.create_counter(
            "requests_total",
            description="Total number of requests"
        )
        self.error_counter = self.meter.create_counter(
            "errors_total",
            description="Total number of errors"
        )
        self.response_time_histogram = self.meter.create_histogram(
            "response_time_seconds",
            description="Response time in seconds"
        )
        self.active_connections_gauge = self.meter.create_up_down_counter(
            "active_connections",
            description="Number of active connections"
        )
    
    def increment_request_count(self, endpoint: str, method: str):
        """Increment request count"""
        self.request_count += 1
        self.request_counter.add(1, {"endpoint": endpoint, "method": method})
    
    def increment_error_count(self, error_type: str):
        """Increment error count"""
        self.error_count += 1
        self.error_counter.add(1, {"error_type": error_type})
    
    def record_response_time(self, duration: float):
        """Record response time"""
        self.response_times.append(duration)
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        self.response_time_histogram.record(duration)
    
    def increment_active_connections(self):
        """Increment active connections"""
        self.active_connections += 1
        self.active_connections_gauge.add(1)
    
    def decrement_active_connections(self):
        """Decrement active connections"""
        self.active_connections -= 1
        self.active_connections_gauge.add(-1)
    
    def get_metrics(self) -> MetricsData:
        """Get current metrics"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return MetricsData(
            request_count=self.request_count,
            error_count=self.error_count,
            average_response_time=avg_response_time,
            active_connections=self.active_connections,
            memory_usage=self._get_memory_usage(),
            cpu_usage=self._get_cpu_usage(),
            disk_usage=self._get_disk_usage(),
            cost_tracking=self.cost_tracker.get_costs()
        )
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
    
    def _get_disk_usage(self) -> float:
        """Get disk usage percentage"""
        try:
            import psutil
            return psutil.disk_usage('/').percent
        except ImportError:
            return 0.0


class HealthChecker:
    """Health check manager for Azure Container Apps"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.logger = structlog.get_logger(__name__)
        self.start_time = time.time()
        self.version = "1.0.0"
        
    async def check_health(self) -> HealthStatus:
        """Perform comprehensive health check"""
        checks = {}
        overall_status = "healthy"
        
        # Check dependencies
        for dependency in self.settings.health_check.readiness_check_dependencies:
            try:
                if dependency == "qdrant":
                    checks["qdrant"] = await self._check_qdrant()
                elif dependency == "azure_openai":
                    checks["azure_openai"] = await self._check_azure_openai()
                elif dependency == "tika":
                    checks["tika"] = await self._check_tika()
                else:
                    checks[dependency] = {"status": "unknown", "message": "Unknown dependency"}
            except Exception as e:
                checks[dependency] = {"status": "unhealthy", "error": str(e)}
                overall_status = "unhealthy"
        
        # Get system metrics
        metrics = self._get_system_metrics()
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=self.version,
            uptime=time.time() - self.start_time,
            checks=checks,
            metrics=metrics
        )
    
    async def check_readiness(self) -> Dict[str, Any]:
        """Check if application is ready to serve requests"""
        health_status = await self.check_health()
        
        if health_status.status == "healthy":
            return {
                "status": "ready",
                "timestamp": health_status.timestamp.isoformat(),
                "uptime": health_status.uptime
            }
        else:
            return {
                "status": "not_ready",
                "timestamp": health_status.timestamp.isoformat(),
                "issues": [
                    f"{check}: {details.get('error', 'unhealthy')}"
                    for check, details in health_status.checks.items()
                    if details.get('status') != 'healthy'
                ]
            }
    
    async def check_liveness(self) -> Dict[str, Any]:
        """Check if application is alive (basic health check)"""
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": time.time() - self.start_time,
            "version": self.version
        }
    
    async def _check_qdrant(self) -> Dict[str, Any]:
        """Check Qdrant vector database connection"""
        try:
            from qdrant_client import QdrantClient
            
            client = QdrantClient(
                host=self.settings.qdrant.host,
                port=self.settings.qdrant.port,
                api_key=self.settings.qdrant.api_key,
                https=self.settings.qdrant.use_https,
                timeout=self.settings.health_check.health_check_timeout
            )
            
            # Test connection
            collections = await asyncio.to_thread(client.get_collections)
            
            return {
                "status": "healthy",
                "collections_count": len(collections.collections),
                "response_time": "< 1s"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_azure_openai(self) -> Dict[str, Any]:
        """Check Azure OpenAI API connection"""
        try:
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                api_key=self.settings.azure_openai.api_key,
                api_version=self.settings.azure_openai.api_version,
                azure_endpoint=self.settings.azure_openai.endpoint
            )
            
            # Test with minimal request
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.settings.azure_openai.chat_model,
                messages=[{"role": "user", "content": "health check"}],
                max_tokens=1
            )
            
            return {
                "status": "healthy",
                "model": self.settings.azure_openai.chat_model,
                "response_time": "< 1s"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_tika(self) -> Dict[str, Any]:
        """Check Apache Tika server connection"""
        try:
            async with httpx.AsyncClient(timeout=self.settings.health_check.health_check_timeout) as client:
                response = await client.get(f"{self.settings.document_processing.tika_server_url}/version")
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "version": response.text.strip(),
                        "response_time": "< 1s"
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}"
                    }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            import psutil
            
            return {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "disk_percent": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids()),
                "boot_time": psutil.boot_time()
            }
        except ImportError:
            return {
                "memory_percent": 0,
                "cpu_percent": 0,
                "disk_percent": 0,
                "process_count": 0,
                "boot_time": 0
            }


class OpenTelemetrySetup:
    """Setup OpenTelemetry for Azure Container Apps"""
    
    @staticmethod
    def setup_telemetry(app, settings):
        """Setup OpenTelemetry tracing and metrics"""
        if not settings.monitoring.enable_opentelemetry:
            return
        
        # Setup trace provider
        trace_provider = TracerProvider()
        trace.set_tracer_provider(trace_provider)
        
        # Setup OTLP exporter for Application Insights
        if settings.monitoring.applicationinsights_connection_string:
            otlp_exporter = OTLPSpanExporter(
                endpoint="https://dc.services.visualstudio.com/v2/track",
                headers={"Authorization": f"Bearer {settings.monitoring.applicationinsights_connection_string}"}
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace_provider.add_span_processor(span_processor)
        
        # Setup metric provider
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(),
            export_interval_millis=settings.monitoring.metrics_export_interval * 1000
        )
        metric_provider = MeterProvider(metric_readers=[metric_reader])
        metrics.set_meter_provider(metric_provider)
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        RequestsInstrumentor().instrument()


# Global instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker()
cost_tracker = CostTracker()


def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance"""
    return metrics_collector


def get_health_checker() -> HealthChecker:
    """Get health checker instance"""
    return health_checker


def get_cost_tracker() -> CostTracker:
    """Get cost tracker instance"""
    return cost_tracker


def get_logger(name: str = __name__):
    """Get structured logger"""
    return structlog.get_logger(name)