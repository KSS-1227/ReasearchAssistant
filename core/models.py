"""
Data models for the Research Assistant System
CSYE 7374 Final Project - Summer 2025
"""

from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field
from datetime import datetime

# ============================================================================
# Enums and Constants
# ============================================================================

class ResearchDomain(str, Enum):
    """Supported research domains"""
    MACHINE_LEARNING = "machine_learning"
    COMPUTER_VISION = "computer_vision"
    NATURAL_LANGUAGE = "natural_language"
    ROBOTICS = "robotics"
    CYBERSECURITY = "cybersecurity"
    SOFTWARE_ENGINEERING = "software_engineering"
    OTHER = "other"

class QuoteType(str, Enum):
    """Types of extracted quotes"""
    FINDING = "finding"
    METHODOLOGY = "methodology"
    SUMMARY = "summary"
    CONCLUSION = "conclusion"

# ============================================================================
# Core Data Models
# ============================================================================

@dataclass
class Paper:
    """Represents a research paper with all metadata"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    year: int
    venue: str
    citations: List[str]
    key_quotes: List[Dict[str, Any]]
    relevance_score: float = 0.0
    
    def __post_init__(self):
        """Initialize empty key_quotes if not provided"""
        if not self.key_quotes:
            self.key_quotes = []
        # Initialize optional attributes to prevent AttributeError
        if not hasattr(self, 'full_text'):
            self.full_text = ""
        if not hasattr(self, 'metadata'):
            self.metadata = {}

@dataclass
class Citation:
    """Represents an extracted citation"""
    authors: str
    year: str
    source_paper_id: str
    citation_text: str
    confidence: float = 1.0

@dataclass
class KeyQuote:
    """Represents an extracted key quote"""
    text: str
    source_paper_id: str
    quote_type: QuoteType
    confidence: float
    position_in_abstract: int = 0

# ============================================================================
# Pydantic Models for LLM I/O
# ============================================================================

class ResearchSynthesis(BaseModel):
    """Structured output from LLM synthesis - ensures consistent format"""
    research_question: str
    key_findings: List[str] = Field(min_length=1, max_length=15, description="Comprehensive technical discoveries")
    methodology_insights: List[str] = Field(min_length=0, max_length=10, description="Research method observations")
    research_gaps: List[str] = Field(min_length=0, max_length=8, description="Areas needing investigation")
    recommended_papers: List[str] = Field(min_length=0, max_length=8, description="Relevant paper titles")
    confidence_score: float = Field(ge=0, le=1, description="Synthesis quality confidence")
    
    # Add new fields for richer output
    technical_contributions: List[str] = Field(default=[], max_items=10, description="Specific technical innovations and algorithmic contributions")
    comparative_analysis: List[str] = Field(default=[], max_items=8, description="Detailed comparisons between different approaches")
    practical_implications: List[str] = Field(default=[], max_items=6, description="Real-world applications and implementation considerations")
    limitations: List[str] = Field(default=[], max_items=10, description="Project limitations and constraints mentioned in paper")
    performance_metrics: List[str] = Field(default=[], max_items=10, description="Quantitative results like accuracy, F1-score, baseline metrics")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ============================================================================
# Session Management Models
# ============================================================================

@dataclass
class ResearchSession:
    """Tracks state for each research query session"""
    session_id: str
    query: str
    domain: ResearchDomain
    created_at: str
    papers_found: List[Paper]
    extracted_citations: List[Citation] = None
    extracted_quotes: List[KeyQuote] = None
    synthesis: Optional[ResearchSynthesis] = None
    total_llm_calls: int = 0
    total_cost: float = 0.0
    processing_time: float = 0.0
    
    def __post_init__(self):
        """Initialize optional fields"""
        if self.extracted_citations is None:
            self.extracted_citations = []
        if self.extracted_quotes is None:
            self.extracted_quotes = []

# ============================================================================
# Agent Performance Models
# ============================================================================

@dataclass
class AgentMetrics:
    """Performance metrics for individual agents"""
    agent_name: str
    uses_llm: bool
    calls_made: int
    llm_calls_made: int
    processing_time: float
    success_rate: float
    last_execution: Optional[str] = None

@dataclass
class SystemMetrics:
    """Overall system performance metrics"""
    total_sessions: int
    total_llm_calls: int
    total_cost: float
    average_llm_calls_per_session: float
    agent_metrics: Dict[str, AgentMetrics]
    efficiency_rating: str
    
    @classmethod
    def calculate_from_sessions(cls, sessions: Dict[str, ResearchSession], agent_metrics: Dict[str, AgentMetrics]):
        """Calculate system metrics from session data"""
        total_sessions = len(sessions)
        total_llm_calls = sum(session.total_llm_calls for session in sessions.values())
        total_cost = sum(session.total_cost for session in sessions.values())
        
        avg_calls = total_llm_calls / total_sessions if total_sessions > 0 else 0
        
        # Determine efficiency rating
        if avg_calls <= 1:
            efficiency_rating = "EXCELLENT"
        elif avg_calls <= 1.5:
            efficiency_rating = "VERY_GOOD"
        elif avg_calls <= 2:
            efficiency_rating = "GOOD"
        else:
            efficiency_rating = "NEEDS_IMPROVEMENT"
            
        return cls(
            total_sessions=total_sessions,
            total_llm_calls=total_llm_calls,
            total_cost=total_cost,
            average_llm_calls_per_session=avg_calls,
            agent_metrics=agent_metrics,
            efficiency_rating=efficiency_rating
        )

# ============================================================================
# Result Models
# ============================================================================

class ResearchResult(BaseModel):
    """Complete result from a research query"""
    session_id: str
    success: bool
    query: str
    domain: str
    papers_found: Dict[str, Any]
    extracted_insights: Dict[str, Any]
    research_synthesis: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

class ErrorResult(BaseModel):
    """Standardized error response"""
    session_id: str
    success: bool = False
    error: str
    performance_metrics: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# ============================================================================
# Utility Functions
# ============================================================================

def create_paper_summary(paper: Paper) -> Dict[str, Any]:
    """Create a summary representation of a paper for API responses - with defensive null checks"""
    paper_id = getattr(paper, 'id', 'unknown')
    paper_title = getattr(paper, 'title', 'Untitled')
    paper_authors = getattr(paper, 'authors', [])
    paper_year = getattr(paper, 'year', 0)
    paper_venue = getattr(paper, 'venue', 'Unknown')
    paper_relevance = getattr(paper, 'relevance_score', 0.0)
    paper_citations = getattr(paper, 'citations', [])
    paper_abstract = getattr(paper, 'abstract', '')
    paper_quotes = getattr(paper, 'key_quotes', [])
    
    return {
        "id": str(paper_id),
        "title": str(paper_title),
        "authors": [str(a) for a in (paper_authors[:3] if paper_authors else [])],
        "year": int(paper_year),
        "venue": str(paper_venue),
        "relevance_score": round(float(paper_relevance), 3),
        "key_quotes": [
            {
                "text": str(quote.get("text", ""))[:150] + "..." if len(str(quote.get("text", ""))) > 150 else str(quote.get("text", "")),
                "type": str(quote.get("quote_type", "unknown")),
                "confidence": float(quote.get("confidence", 0.5))
            }
            for quote in (paper_quotes[:2] if paper_quotes else []) if isinstance(quote, dict)
        ],
        "citation_count": len(paper_citations) if paper_citations else 0,
        "abstract_preview": str(paper_abstract)[:200] + "..." if len(str(paper_abstract)) > 200 else str(paper_abstract)
    }

def validate_research_query(query: str) -> Dict[str, Any]:
    """Validate research query input"""
    if not query or len(query.strip()) < 5:
        return {"valid": False, "error": "Query too short - minimum 5 characters"}
    
    if len(query) > 500:
        return {"valid": False, "error": "Query too long - maximum 500 characters"}
    
    # Check for potentially problematic queries
    problematic_terms = ["hack", "exploit", "illegal", "harmful"]
    if any(term in query.lower() for term in problematic_terms):
        return {"valid": False, "error": "Query contains potentially problematic terms"}
    
    return {"valid": True, "cleaned_query": query.strip()}