"""
DocuLLaMA Document Processing Pipeline
Universal document processing with Apache Tika and AutoLLaMA features
"""

import asyncio
import hashlib
import json
import mimetypes
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import aiofiles
import httpx
from fastapi import UploadFile
from pydantic import BaseModel

from config import Settings, get_settings
from monitoring import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """Document chunk data structure"""
    chunk_id: str
    content: str
    chunk_type: str  # header, content, summary, conclusion
    position: int
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None


@dataclass
class ProcessingResult:
    """Document processing result"""
    document_id: str
    filename: str
    status: str
    message: str
    chunks_created: int
    metadata: Dict[str, Any]
    chunks: List[DocumentChunk]


class DocumentMetadata(BaseModel):
    """Document metadata model"""
    filename: str
    file_size: int
    content_type: str
    document_type: str
    created_at: datetime
    processed_at: datetime
    source_hash: str
    page_count: Optional[int] = None
    word_count: int
    language: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    extraction_method: str
    quality_score: float


class TikaClient:
    """Apache Tika client for document processing"""
    
    def __init__(self, tika_url: str):
        self.tika_url = tika_url
        self.timeout = 60
    
    async def extract_text(self, file_content: bytes, content_type: str) -> Dict[str, Any]:
        """Extract text from document using Tika"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Extract text
                text_response = await client.put(
                    f"{self.tika_url}/tika/text",
                    content=file_content,
                    headers={'Content-Type': content_type}
                )
                text_content = text_response.text
                
                # Extract metadata
                metadata_response = await client.put(
                    f"{self.tika_url}/meta",
                    content=file_content,
                    headers={'Content-Type': content_type, 'Accept': 'application/json'}
                )
                metadata = metadata_response.json()
                
                return {
                    'text': text_content,
                    'metadata': metadata,
                    'extraction_method': 'apache_tika'
                }
                
        except Exception as e:
            logger.error(f"Tika extraction failed: {e}")
            raise


class DocumentProcessor:
    """Universal document processor with AutoLLaMA features"""
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()
        self.tika_client = TikaClient(self.settings.document_processing.tika_server_url)
        
        # Quality thresholds
        self.min_content_length = self.settings.document_processing.min_chunk_size
        self.quality_threshold = self.settings.document_processing.quality_threshold
        
        # Supported formats
        self.supported_formats = set(self.settings.document_processing.supported_formats)
        
        # Initialize processing stats
        self.processing_stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'duplicates_detected': 0,
            'quality_rejected': 0
        }
    
    async def initialize(self):
        """Initialize the document processor"""
        logger.info("Initializing document processor")
        
        # Test Tika connection
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.settings.document_processing.tika_server_url}/version")
                if response.status_code == 200:
                    logger.info(f"Tika server connected: {response.text.strip()}")
                else:
                    raise Exception(f"Tika server unhealthy: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Tika server: {e}")
            raise
        
        logger.info("Document processor initialized successfully")
    
    async def process_document(
        self,
        file: UploadFile,
        collection_name: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """Process uploaded document with AutoLLaMA features"""
        document_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Read file content
            file_content = await file.read()
            file_size = len(file_content)
            
            # Validate file
            self._validate_file(file.filename, file_size, file.content_type)
            
            # Generate content hash for duplicate detection
            content_hash = hashlib.sha256(file_content).hexdigest()
            
            # Check for duplicates (implement duplicate detection logic)
            if await self._is_duplicate(content_hash):
                return ProcessingResult(
                    document_id=document_id,
                    filename=file.filename,
                    status="duplicate",
                    message="Document already exists in the system",
                    chunks_created=0,
                    metadata={"content_hash": content_hash},
                    chunks=[]
                )
            
            # Extract content using Tika
            extraction_result = await self.tika_client.extract_text(
                file_content,
                file.content_type or mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
            )
            
            text_content = extraction_result['text']
            tika_metadata = extraction_result['metadata']
            
            # Validate content quality
            if not self._validate_content_quality(text_content):
                self.processing_stats['quality_rejected'] += 1
                return ProcessingResult(
                    document_id=document_id,
                    filename=file.filename,
                    status="rejected",
                    message="Content quality below threshold",
                    chunks_created=0,
                    metadata={"quality_score": 0.0},
                    chunks=[]
                )
            
            # Create document metadata
            metadata = DocumentMetadata(
                filename=file.filename,
                file_size=file_size,
                content_type=file.content_type or 'unknown',
                document_type=self._detect_document_type(file.filename, tika_metadata),
                created_at=start_time,
                processed_at=datetime.utcnow(),
                source_hash=content_hash,
                page_count=self._extract_page_count(tika_metadata),
                word_count=len(text_content.split()),
                language=tika_metadata.get('language'),
                author=tika_metadata.get('Author'),
                title=tika_metadata.get('title') or file.filename,
                extraction_method='apache_tika',
                quality_score=self._calculate_quality_score(text_content, tika_metadata)
            )
            
            # Enhanced hierarchical chunking (AutoLLaMA feature)
            chunks = await self._create_hierarchical_chunks(
                text_content,
                document_id,
                metadata.dict(),
                tika_metadata
            )
            
            # Filter chunks by quality
            quality_chunks = [chunk for chunk in chunks if self._is_chunk_quality(chunk)]
            
            if not quality_chunks:
                return ProcessingResult(
                    document_id=document_id,
                    filename=file.filename,
                    status="rejected",
                    message="No quality chunks created",
                    chunks_created=0,
                    metadata=metadata.dict(),
                    chunks=[]
                )
            
            # Update stats
            self.processing_stats['documents_processed'] += 1
            self.processing_stats['chunks_created'] += len(quality_chunks)
            
            return ProcessingResult(
                document_id=document_id,
                filename=file.filename,
                status="success",
                message="Document processed successfully",
                chunks_created=len(quality_chunks),
                metadata=metadata.dict(),
                chunks=quality_chunks
            )
            
        except Exception as e:
            logger.error(f"Document processing failed for {file.filename}: {e}")
            return ProcessingResult(
                document_id=document_id,
                filename=file.filename,
                status="error",
                message=f"Processing failed: {str(e)}",
                chunks_created=0,
                metadata={"error": str(e)},
                chunks=[]
            )
    
    def _validate_file(self, filename: str, file_size: int, content_type: str):
        """Validate uploaded file"""
        if not filename:
            raise ValueError("Filename is required")
        
        # Check file size
        max_size = self.settings.document_processing.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            raise ValueError(f"File size {file_size} exceeds maximum {max_size} bytes")
        
        # Check file extension
        file_ext = Path(filename).suffix.lower().lstrip('.')
        if self.supported_formats and file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    async def _is_duplicate(self, content_hash: str) -> bool:
        """Check if document is duplicate based on content hash"""
        # TODO: Implement duplicate detection with Qdrant or Redis
        # For now, return False (no duplicates)
        return False
    
    def _validate_content_quality(self, content: str) -> bool:
        """Validate content quality (AutoLLaMA feature)"""
        if not content or len(content.strip()) < self.min_content_length:
            return False
        
        # Check for meaningful content (not just whitespace or junk)
        meaningful_chars = sum(1 for c in content if c.isalnum())
        if meaningful_chars / len(content) < 0.3:  # At least 30% alphanumeric
            return False
        
        return True
    
    def _detect_document_type(self, filename: str, metadata: Dict[str, Any]) -> str:
        """Detect document type from filename and metadata"""
        file_ext = Path(filename).suffix.lower()
        
        # Map extensions to document types
        type_mapping = {
            '.pdf': 'pdf',
            '.doc': 'word',
            '.docx': 'word',
            '.xls': 'excel',
            '.xlsx': 'excel',
            '.ppt': 'powerpoint',
            '.pptx': 'powerpoint',
            '.txt': 'text',
            '.md': 'markdown',
            '.py': 'code',
            '.js': 'code',
            '.html': 'web',
            '.xml': 'structured_data',
            '.json': 'structured_data',
            '.csv': 'data'
        }
        
        return type_mapping.get(file_ext, 'unknown')
    
    def _extract_page_count(self, metadata: Dict[str, Any]) -> Optional[int]:
        """Extract page count from Tika metadata"""
        # Try various page count fields
        page_fields = ['xmpTPg:NPages', 'meta:page-count', 'Page-Count', 'pdf:docinfo:pages']
        
        for field in page_fields:
            if field in metadata:
                try:
                    return int(metadata[field])
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _calculate_quality_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate content quality score (AutoLLaMA feature)"""
        score = 0.0
        
        # Content length factor (0-0.3)
        content_length = len(content)
        if content_length > 1000:
            score += 0.3
        elif content_length > 500:
            score += 0.2
        elif content_length > 100:
            score += 0.1
        
        # Meaningful content factor (0-0.3)
        if content:
            meaningful_ratio = sum(1 for c in content if c.isalnum()) / len(content)
            score += meaningful_ratio * 0.3
        
        # Structure factor (0-0.2)
        if any(header in content.lower() for header in ['introduction', 'conclusion', 'summary', 'abstract']):
            score += 0.1
        
        if '\n' in content:  # Has line breaks (structured)
            score += 0.1
        
        # Metadata factor (0-0.2)
        if metadata.get('title'):
            score += 0.05
        if metadata.get('Author'):
            score += 0.05
        if metadata.get('language'):
            score += 0.05
        if metadata.get('Created'):
            score += 0.05
        
        return min(score, 1.0)
    
    async def _create_hierarchical_chunks(
        self,
        content: str,
        document_id: str,
        metadata: Dict[str, Any],
        tika_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create hierarchical chunks (AutoLLaMA enhanced feature)"""
        chunks = []
        
        # Split content into sections
        sections = self._split_into_sections(content)
        
        chunk_size = self.settings.document_processing.chunk_size
        chunk_overlap = self.settings.document_processing.chunk_overlap
        
        for section_idx, section in enumerate(sections):
            section_type = self._classify_section_type(section)
            
            # Create chunks from section
            section_chunks = self._create_chunks_from_text(
                section,
                chunk_size,
                chunk_overlap,
                document_id,
                section_idx,
                section_type,
                metadata
            )
            
            chunks.extend(section_chunks)
        
        return chunks
    
    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections"""
        # Simple section splitting based on multiple newlines
        sections = []
        current_section = []
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if this line starts a new section
            if self._is_section_boundary(line, current_section):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line] if line else []
            else:
                if line:  # Skip empty lines between sections
                    current_section.append(line)
        
        # Add final section
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _is_section_boundary(self, line: str, current_section: List[str]) -> bool:
        """Check if line indicates a section boundary"""
        if not current_section:
            return False
        
        # Headers/titles (all caps, short lines)
        if line.isupper() and len(line) < 100:
            return True
        
        # Numbered sections
        if line.startswith(('1.', '2.', '3.', '4.', '5.')) and len(line) < 100:
            return True
        
        # Common section headers
        section_headers = [
            'introduction', 'abstract', 'summary', 'conclusion',
            'background', 'methodology', 'results', 'discussion',
            'references', 'bibliography', 'appendix'
        ]
        
        if any(header in line.lower() for header in section_headers):
            return True
        
        return False
    
    def _classify_section_type(self, section: str) -> str:
        """Classify section type for enhanced chunking"""
        section_lower = section.lower()
        
        # Classification keywords
        if any(keyword in section_lower for keyword in ['abstract', 'summary']):
            return 'summary'
        elif any(keyword in section_lower for keyword in ['introduction', 'background']):
            return 'introduction'
        elif any(keyword in section_lower for keyword in ['conclusion', 'summary', 'final']):
            return 'conclusion'
        elif any(keyword in section_lower for keyword in ['method', 'approach', 'procedure']):
            return 'methodology'
        elif any(keyword in section_lower for keyword in ['result', 'finding', 'outcome']):
            return 'results'
        elif any(keyword in section_lower for keyword in ['reference', 'bibliography', 'citation']):
            return 'references'
        else:
            return 'content'
    
    def _create_chunks_from_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        document_id: str,
        section_idx: int,
        section_type: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create chunks from text with overlap"""
        chunks = []
        words = text.split()
        
        if not words:
            return chunks
        
        start_idx = 0
        chunk_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_content = ' '.join(chunk_words)
            
            # Create chunk metadata
            chunk_metadata = {
                **metadata,
                'section_idx': section_idx,
                'section_type': section_type,
                'chunk_idx': chunk_idx,
                'start_word': start_idx,
                'end_word': end_idx,
                'word_count': len(chunk_words),
                'position_in_document': start_idx / len(words)
            }
            
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_{section_idx}_{chunk_idx}",
                content=chunk_content,
                chunk_type=section_type,
                position=chunk_idx,
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            if end_idx >= len(words):
                break
            
            start_idx = end_idx - overlap
            chunk_idx += 1
        
        return chunks
    
    def _is_chunk_quality(self, chunk: DocumentChunk) -> bool:
        """Check if chunk meets quality standards"""
        # Content length check
        if len(chunk.content) < self.min_content_length:
            return False
        
        # Meaningful content check
        meaningful_chars = sum(1 for c in chunk.content if c.isalnum())
        if meaningful_chars / len(chunk.content) < 0.3:
            return False
        
        # Word count check
        word_count = len(chunk.content.split())
        if word_count < 10:  # At least 10 words
            return False
        
        return True
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up document processor")
        # Add any cleanup logic here
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()