"""
DocuLLaMA Research Intelligence Engine
AutoLLaMA research intelligence capabilities for Azure Container Apps
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from openai import AzureOpenAI

from config import Settings, get_settings
from monitoring import get_logger, get_cost_tracker
from document_processor import DocumentChunk

logger = get_logger(__name__)


class ResearchTopicCategory(Enum):
    """Research topic categories"""
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    HISTORICAL = "historical"
    SOCIAL = "social"
    MEDICAL = "medical"
    LEGAL = "legal"
    EDUCATIONAL = "educational"
    CREATIVE = "creative"


class ResearchStatus(Enum):
    """Research suggestion status"""
    PENDING = "pending"
    REVIEWED = "reviewed"
    IMPLEMENTED = "implemented"
    ARCHIVED = "archived"
    REJECTED = "rejected"


@dataclass
class ResearchInsight:
    """Research insight data structure"""
    insight_id: str
    content: str
    confidence: float
    category: ResearchTopicCategory
    keywords: List[str]
    supporting_evidence: List[str]
    follow_up_questions: List[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            "category": self.category.value,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ResearchSuggestion:
    """Research suggestion data structure"""
    suggestion_id: str
    title: str
    description: str
    confidence: float
    category: ResearchTopicCategory
    priority: str  # high, medium, low
    estimated_effort: str  # hours, days, weeks
    prerequisites: List[str]
    potential_sources: List[str]
    expected_outcomes: List[str]
    status: ResearchStatus
    insights: List[ResearchInsight]
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            "category": self.category.value,
            "status": self.status.value,
            "insights": [insight.to_dict() for insight in self.insights],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class ContentGap:
    """Content gap analysis result"""
    gap_id: str
    topic: str
    description: str
    importance: float
    coverage_percentage: float
    missing_aspects: List[str]
    suggested_queries: List[str]
    related_content: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ResearchPattern:
    """Detected research pattern"""
    pattern_id: str
    pattern_type: str  # trending, seasonal, emerging, declining
    description: str
    frequency: int
    strength: float
    time_span: str
    related_topics: List[str]
    examples: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ContentAnalyzer:
    """Analyze content for research insights"""
    
    def __init__(self, azure_openai: AzureOpenAI, settings: Settings):
        self.azure_openai = azure_openai
        self.settings = settings
        self.cost_tracker = get_cost_tracker()
        
        # Analysis prompts
        self.insight_prompt = """
        Analyze the following content and provide research insights:
        
        Content: {content}
        
        Please provide:
        1. Key insights and findings (max 5)
        2. Confidence level for each insight (0.0-1.0)
        3. Research categories (academic, technical, business, etc.)
        4. Important keywords and concepts
        5. Follow-up research questions
        6. Supporting evidence or examples
        
        Respond in JSON format with the following structure:
        {{
            "insights": [
                {{
                    "content": "insight description",
                    "confidence": 0.8,
                    "category": "academic",
                    "keywords": ["keyword1", "keyword2"],
                    "supporting_evidence": ["evidence1", "evidence2"],
                    "follow_up_questions": ["question1", "question2"]
                }}
            ]
        }}
        """
        
        self.gap_analysis_prompt = """
        Analyze the following content collection for knowledge gaps:
        
        Content Topics: {topics}
        Content Summary: {summary}
        
        Identify:
        1. Important topics that are missing or underrepresented
        2. Gaps in coverage of existing topics
        3. Suggested research directions
        4. Priority levels for addressing gaps
        
        Respond in JSON format:
        {{
            "gaps": [
                {{
                    "topic": "missing topic",
                    "description": "why this is important",
                    "importance": 0.8,
                    "coverage_percentage": 0.2,
                    "missing_aspects": ["aspect1", "aspect2"],
                    "suggested_queries": ["query1", "query2"]
                }}
            ]
        }}
        """
    
    async def analyze_content_insights(self, content: str) -> List[ResearchInsight]:
        """Analyze content for research insights"""
        try:
            prompt = self.insight_prompt.format(content=content[:4000])  # Limit content length
            
            response = await asyncio.to_thread(
                self.azure_openai.chat.completions.create,
                model=self.settings.azure_openai.chat_model,
                messages=[
                    {"role": "system", "content": "You are a research analyst expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Track costs
            if hasattr(response, 'usage') and response.usage:
                self.cost_tracker.track_chat_usage(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            try:
                result = json.loads(response_text)
                insights = []
                
                for insight_data in result.get("insights", []):
                    insight = ResearchInsight(
                        insight_id=str(uuid.uuid4()),
                        content=insight_data.get("content", ""),
                        confidence=insight_data.get("confidence", 0.0),
                        category=ResearchTopicCategory(insight_data.get("category", "technical")),
                        keywords=insight_data.get("keywords", []),
                        supporting_evidence=insight_data.get("supporting_evidence", []),
                        follow_up_questions=insight_data.get("follow_up_questions", []),
                        created_at=datetime.utcnow()
                    )
                    insights.append(insight)
                
                return insights
                
            except json.JSONDecodeError:
                logger.error("Failed to parse AI response as JSON")
                return []
                
        except Exception as e:
            logger.error(f"Content insight analysis failed: {e}")
            return []
    
    async def analyze_content_gaps(
        self,
        chunks: List[DocumentChunk],
        existing_topics: List[str] = None
    ) -> List[ContentGap]:
        """Analyze content for knowledge gaps"""
        try:
            # Prepare content summary
            topics = existing_topics or []
            if not topics:
                # Extract topics from chunks
                topics = self._extract_topics_from_chunks(chunks)
            
            content_summary = self._create_content_summary(chunks)
            
            prompt = self.gap_analysis_prompt.format(
                topics=", ".join(topics[:20]),  # Limit topics
                summary=content_summary[:2000]  # Limit summary
            )
            
            response = await asyncio.to_thread(
                self.azure_openai.chat.completions.create,
                model=self.settings.azure_openai.chat_model,
                messages=[
                    {"role": "system", "content": "You are a research strategy expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2000
            )
            
            # Track costs
            if hasattr(response, 'usage') and response.usage:
                self.cost_tracker.track_chat_usage(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            try:
                result = json.loads(response_text)
                gaps = []
                
                for gap_data in result.get("gaps", []):
                    gap = ContentGap(
                        gap_id=str(uuid.uuid4()),
                        topic=gap_data.get("topic", ""),
                        description=gap_data.get("description", ""),
                        importance=gap_data.get("importance", 0.0),
                        coverage_percentage=gap_data.get("coverage_percentage", 0.0),
                        missing_aspects=gap_data.get("missing_aspects", []),
                        suggested_queries=gap_data.get("suggested_queries", []),
                        related_content=[]  # Will be populated later
                    )
                    gaps.append(gap)
                
                return gaps
                
            except json.JSONDecodeError:
                logger.error("Failed to parse gap analysis response as JSON")
                return []
                
        except Exception as e:
            logger.error(f"Content gap analysis failed: {e}")
            return []
    
    def _extract_topics_from_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Extract topics from document chunks"""
        topics = set()
        
        for chunk in chunks:
            # Use chunk metadata if available
            if "topics" in chunk.metadata:
                topics.update(chunk.metadata["topics"])
            
            # Extract from chunk type
            if chunk.chunk_type in ["introduction", "summary", "conclusion"]:
                # These chunks likely contain topic information
                words = chunk.content.lower().split()
                # Simple topic extraction (could be enhanced with NLP)
                important_words = [w for w in words if len(w) > 5 and w.isalpha()]
                topics.update(important_words[:5])
        
        return list(topics)[:20]  # Limit to 20 topics
    
    def _create_content_summary(self, chunks: List[DocumentChunk]) -> str:
        """Create content summary from chunks"""
        summaries = []
        
        for chunk in chunks:
            if chunk.chunk_type in ["summary", "abstract", "introduction", "conclusion"]:
                summaries.append(chunk.content[:200])  # First 200 chars
        
        if not summaries:
            # Fallback to first few chunks
            summaries = [chunk.content[:200] for chunk in chunks[:3]]
        
        return " ".join(summaries)


class PatternDetector:
    """Detect patterns in research data"""
    
    def __init__(self):
        self.min_pattern_frequency = 3
        self.pattern_strength_threshold = 0.6
    
    async def detect_research_patterns(
        self,
        suggestions: List[ResearchSuggestion],
        time_window: timedelta = timedelta(days=30)
    ) -> List[ResearchPattern]:
        """Detect patterns in research suggestions"""
        patterns = []
        
        # Filter recent suggestions
        cutoff_date = datetime.utcnow() - time_window
        recent_suggestions = [
            s for s in suggestions 
            if s.created_at >= cutoff_date
        ]
        
        if len(recent_suggestions) < self.min_pattern_frequency:
            return patterns
        
        # Detect category patterns
        category_patterns = await self._detect_category_patterns(recent_suggestions)
        patterns.extend(category_patterns)
        
        # Detect keyword patterns
        keyword_patterns = await self._detect_keyword_patterns(recent_suggestions)
        patterns.extend(keyword_patterns)
        
        # Detect temporal patterns
        temporal_patterns = await self._detect_temporal_patterns(recent_suggestions)
        patterns.extend(temporal_patterns)
        
        return patterns
    
    async def _detect_category_patterns(
        self,
        suggestions: List[ResearchSuggestion]
    ) -> List[ResearchPattern]:
        """Detect patterns in research categories"""
        patterns = []
        
        # Count categories
        category_counts = {}
        for suggestion in suggestions:
            category = suggestion.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Identify trending categories
        total_suggestions = len(suggestions)
        for category, count in category_counts.items():
            frequency = count / total_suggestions
            
            if count >= self.min_pattern_frequency and frequency >= 0.3:
                pattern = ResearchPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type="trending",
                    description=f"High frequency of {category} research suggestions",
                    frequency=count,
                    strength=frequency,
                    time_span="recent",
                    related_topics=[category],
                    examples=[
                        s.title for s in suggestions 
                        if s.category.value == category
                    ][:3]
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_keyword_patterns(
        self,
        suggestions: List[ResearchSuggestion]
    ) -> List[ResearchPattern]:
        """Detect patterns in keywords and topics"""
        patterns = []
        
        # Collect all keywords from insights
        keyword_counts = {}
        for suggestion in suggestions:
            for insight in suggestion.insights:
                for keyword in insight.keywords:
                    keyword_counts[keyword.lower()] = keyword_counts.get(keyword.lower(), 0) + 1
        
        # Identify trending keywords
        total_insights = sum(len(s.insights) for s in suggestions)
        if total_insights > 0:
            for keyword, count in keyword_counts.items():
                frequency = count / total_insights
                
                if count >= self.min_pattern_frequency and frequency >= 0.2:
                    pattern = ResearchPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="emerging",
                        description=f"Emerging keyword: {keyword}",
                        frequency=count,
                        strength=frequency,
                        time_span="recent",
                        related_topics=[keyword],
                        examples=[
                            s.title for s in suggestions
                            if any(keyword in i.keywords for i in s.insights)
                        ][:3]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_temporal_patterns(
        self,
        suggestions: List[ResearchSuggestion]
    ) -> List[ResearchPattern]:
        """Detect temporal patterns in research suggestions"""
        patterns = []
        
        # Group by time periods (daily)
        daily_counts = {}
        for suggestion in suggestions:
            day_key = suggestion.created_at.date()
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
        
        # Analyze trends
        if len(daily_counts) >= 3:
            counts = list(daily_counts.values())
            dates = sorted(daily_counts.keys())
            
            # Simple trend detection
            if len(counts) >= 3:
                recent_avg = np.mean(counts[-3:])
                earlier_avg = np.mean(counts[:-3]) if len(counts) > 3 else 0
                
                if recent_avg > earlier_avg * 1.5 and recent_avg >= 2:
                    pattern = ResearchPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="trending",
                        description="Increasing research activity",
                        frequency=int(recent_avg),
                        strength=recent_avg / (earlier_avg + 1),
                        time_span=f"{dates[0]} to {dates[-1]}",
                        related_topics=["general_activity"],
                        examples=[]
                    )
                    patterns.append(pattern)
        
        return patterns


class ResearchIntelligenceEngine:
    """Main research intelligence engine"""
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()
        
        # Initialize AI client
        self.azure_openai = AzureOpenAI(
            api_key=self.settings.azure_openai.api_key,
            api_version=self.settings.azure_openai.api_version,
            azure_endpoint=self.settings.azure_openai.endpoint
        )
        
        # Initialize components
        self.content_analyzer = ContentAnalyzer(self.azure_openai, self.settings)
        self.pattern_detector = PatternDetector()
        
        # Storage (in production, use database)
        self.suggestions: Dict[str, ResearchSuggestion] = {}
        self.insights: Dict[str, ResearchInsight] = {}
        self.patterns: Dict[str, ResearchPattern] = {}
        self.gaps: Dict[str, ContentGap] = {}
        
        # Configuration
        self.max_suggestions_per_analysis = 10
        self.suggestion_confidence_threshold = 0.5
        self.auto_archive_days = 30
    
    async def initialize(self):
        """Initialize research intelligence engine"""
        logger.info("Initializing research intelligence engine")
        
        # Test AI connection
        try:
            test_response = await asyncio.to_thread(
                self.azure_openai.chat.completions.create,
                model=self.settings.azure_openai.chat_model,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            logger.info("Azure OpenAI connection successful")
        except Exception as e:
            logger.error(f"Failed to connect to Azure OpenAI: {e}")
            raise
        
        logger.info("Research intelligence engine initialized")
    
    async def analyze_content(
        self,
        content: str,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze content and generate research intelligence"""
        try:
            # Generate insights
            insights = await self.content_analyzer.analyze_content_insights(content)
            
            # Store insights
            for insight in insights:
                self.insights[insight.insight_id] = insight
            
            # Generate research suggestions based on insights
            suggestions = await self._generate_research_suggestions(insights)
            
            # Store suggestions
            for suggestion in suggestions:
                self.suggestions[suggestion.suggestion_id] = suggestion
            
            # Detect patterns if we have enough data
            patterns = []
            if len(self.suggestions) >= 5:
                all_suggestions = list(self.suggestions.values())
                patterns = await self.pattern_detector.detect_research_patterns(all_suggestions)
                
                # Store patterns
                for pattern in patterns:
                    self.patterns[pattern.pattern_id] = pattern
            
            return {
                "insights": [insight.to_dict() for insight in insights],
                "suggestions": [suggestion.to_dict() for suggestion in suggestions],
                "patterns": [pattern.to_dict() for pattern in patterns],
                "analysis_type": analysis_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {
                "insights": [],
                "suggestions": [],
                "patterns": [],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _generate_research_suggestions(
        self,
        insights: List[ResearchInsight]
    ) -> List[ResearchSuggestion]:
        """Generate research suggestions from insights"""
        suggestions = []
        
        for insight in insights:
            if insight.confidence >= self.suggestion_confidence_threshold:
                # Create research suggestion based on insight
                suggestion = ResearchSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    title=f"Research: {insight.content[:50]}...",
                    description=self._create_suggestion_description(insight),
                    confidence=insight.confidence,
                    category=insight.category,
                    priority=self._determine_priority(insight.confidence),
                    estimated_effort=self._estimate_effort(insight),
                    prerequisites=insight.supporting_evidence,
                    potential_sources=self._suggest_sources(insight),
                    expected_outcomes=insight.follow_up_questions,
                    status=ResearchStatus.PENDING,
                    insights=[insight],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                suggestions.append(suggestion)
        
        return suggestions[:self.max_suggestions_per_analysis]
    
    def _create_suggestion_description(self, insight: ResearchInsight) -> str:
        """Create description for research suggestion"""
        description = f"Based on the insight: {insight.content}\n\n"
        
        if insight.keywords:
            description += f"Key topics: {', '.join(insight.keywords)}\n"
        
        if insight.follow_up_questions:
            description += f"Research questions:\n"
            for question in insight.follow_up_questions[:3]:
                description += f"- {question}\n"
        
        return description
    
    def _determine_priority(self, confidence: float) -> str:
        """Determine priority based on confidence"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _estimate_effort(self, insight: ResearchInsight) -> str:
        """Estimate research effort"""
        if len(insight.follow_up_questions) > 3:
            return "weeks"
        elif len(insight.follow_up_questions) > 1:
            return "days"
        else:
            return "hours"
    
    def _suggest_sources(self, insight: ResearchInsight) -> List[str]:
        """Suggest research sources based on category"""
        category_sources = {
            ResearchTopicCategory.ACADEMIC: [
                "Google Scholar", "PubMed", "arXiv", "Research databases"
            ],
            ResearchTopicCategory.TECHNICAL: [
                "Technical documentation", "GitHub", "Stack Overflow", "Industry blogs"
            ],
            ResearchTopicCategory.BUSINESS: [
                "Industry reports", "Company websites", "Business databases", "News sources"
            ],
            ResearchTopicCategory.SCIENTIFIC: [
                "Scientific journals", "Research institutions", "Government databases"
            ]
        }
        
        return category_sources.get(insight.category, ["Web search", "Expert interviews"])
    
    async def get_research_suggestions(
        self,
        limit: int = 50,
        status_filter: Optional[ResearchStatus] = None,
        category_filter: Optional[ResearchTopicCategory] = None
    ) -> List[Dict[str, Any]]:
        """Get research suggestions with filtering"""
        suggestions = list(self.suggestions.values())
        
        # Apply filters
        if status_filter:
            suggestions = [s for s in suggestions if s.status == status_filter]
        
        if category_filter:
            suggestions = [s for s in suggestions if s.category == category_filter]
        
        # Sort by confidence and creation date
        suggestions.sort(key=lambda x: (x.confidence, x.created_at), reverse=True)
        
        return [suggestion.to_dict() for suggestion in suggestions[:limit]]
    
    async def update_suggestion_status(
        self,
        suggestion_id: str,
        new_status: ResearchStatus
    ) -> bool:
        """Update research suggestion status"""
        if suggestion_id in self.suggestions:
            self.suggestions[suggestion_id].status = new_status
            self.suggestions[suggestion_id].updated_at = datetime.utcnow()
            logger.info(f"Updated suggestion {suggestion_id} status to {new_status.value}")
            return True
        return False
    
    async def get_research_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get research summary statistics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Recent suggestions
        recent_suggestions = [
            s for s in self.suggestions.values()
            if s.created_at >= cutoff_date
        ]
        
        # Status breakdown
        status_counts = {}
        for suggestion in recent_suggestions:
            status = suggestion.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Category breakdown
        category_counts = {}
        for suggestion in recent_suggestions:
            category = suggestion.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Average confidence
        avg_confidence = (
            sum(s.confidence for s in recent_suggestions) / len(recent_suggestions)
            if recent_suggestions else 0
        )
        
        return {
            "total_suggestions": len(recent_suggestions),
            "status_breakdown": status_counts,
            "category_breakdown": category_counts,
            "average_confidence": avg_confidence,
            "time_period_days": days,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def cleanup(self):
        """Cleanup resources and auto-archive old suggestions"""
        logger.info("Cleaning up research intelligence engine")
        
        # Auto-archive old suggestions
        cutoff_date = datetime.utcnow() - timedelta(days=self.auto_archive_days)
        archived_count = 0
        
        for suggestion in self.suggestions.values():
            if (suggestion.created_at < cutoff_date and 
                suggestion.status == ResearchStatus.PENDING):
                suggestion.status = ResearchStatus.ARCHIVED
                suggestion.updated_at = datetime.utcnow()
                archived_count += 1
        
        if archived_count > 0:
            logger.info(f"Auto-archived {archived_count} old suggestions")
        
        logger.info("Research intelligence cleanup complete")