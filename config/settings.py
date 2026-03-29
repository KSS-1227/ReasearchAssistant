"""
Configuration settings for the Research Assistant System
CSYE 7374 Final Project - Summer 2025
"""

import os
import logging
import logging.config
from typing import Dict, Any

# ============================================================================
# System Configuration
# ============================================================================

class SystemConfig:
    """Central configuration for the research assistant system"""
    
    # API Configuration - Now using Google Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
    # Fallback to OPENAI_API_KEY for backward compatibility if needed
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    DEFAULT_MODEL = "gemini-2.5-flash"  # Stable, widely-available model
    
    # Real token-based pricing (per 1M tokens) — sourced from Google AI pricing page
    COST_PER_TOKEN = {
        "gemini-2.0-flash":  {"input": 0.00000010,  "output": 0.0000004},
        "gemini-2.5-flash":  {"input": 0.00000015,  "output": 0.0000006},
        "gemini-2.5-pro":    {"input": 0.00000125,  "output": 0.000010},
        "gemini-1.5-pro":    {"input": 0.00000125,  "output": 0.000005},
        "gemini-1.5-flash":  {"input": 0.000000075, "output": 0.0000003},
        "gemini-pro":        {"input": 0.0000005,   "output": 0.0000015},
    }
    
    # Performance Targets
    MAX_LLM_CALLS_PER_QUERY = 2
    TARGET_LLM_CALLS_PER_QUERY = 1
    MAX_PROCESSING_TIME = 60  # seconds
    
    # Literature Scanner Configuration
    LITERATURE_CONFIG = {
        "max_papers_per_domain": 20,
        "min_relevance_threshold": 0.1,
        "recency_boost_factor": 0.05,
        "citation_boost_factor": 0.02,
        "max_recency_boost": 0.2,
        "max_citation_boost": 0.3
    }
    
    # Citation Extractor Configuration
    CITATION_CONFIG = {
        "min_quote_length": 20,
        "max_quote_length": 200,
        "max_quotes_per_paper": 3,
        "confidence_threshold": 0.5
    }
    
    # Synthesis Agent Configuration
    SYNTHESIS_CONFIG = {
        "max_input_papers": 8,
        "max_tokens": 6000,
        "temperature": 0.3,
        "min_key_findings": 5,
        "max_key_findings": 15,
        "min_research_gaps": 2,
        "max_research_gaps": 8,
        "min_methodology_insights": 3,
        "max_methodology_insights": 10
    }
    
    # Document Processing Configuration
    DOCUMENT_CONFIG = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_documents": 100,
        "supported_formats": [".pdf", ".txt", ".md"],
        "vector_store_type": "FAISS",
        "embedding_model": "models/text-embedding-004"  # Google Gemini embedding model
    }

    # RAG pipeline constants (replaces magic numbers scattered across files)
    RAG_CONFIG = {
        "max_full_text_chars": 30_000,    # max chars sent per doc to LLM
        "embedding_dimension": 768,        # text-embedding-004 output size
        "max_references_parsed": 20,       # citation extractor reference limit
        "max_quotes_per_paper": 5,         # quotes included in LLM context
        "heading_max_length": 80,          # chars before heading is truncated
        "heading_scan_lines": 10,          # lines checked for section heading
        "api_key_preview_chars": 10,       # chars shown in key confirmation log
    }
    
    # UI Configuration
    UI_CONFIG = {
        "streamlit_port": 8501,
        "page_title": "🔬 Research Assistant AI",
        "page_icon": "🔬",
        "layout": "wide",
        "sidebar_state": "expanded"
    }
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """Check if valid API key is configured for Gemini"""
        return cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != "your-gemini-api-key-here"

    @classmethod
    def calculate_real_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate actual cost from real token counts returned by the API"""
        rates = cls.COST_PER_TOKEN.get(model, {"input": 0.000002, "output": 0.000002})
        return (input_tokens * rates["input"]) + (output_tokens * rates["output"])

    @classmethod
    def get_cost_estimate(cls, model: str = None) -> float:
        """Estimate cost for a typical call (used only for pre-call budget display)"""
        model = model or cls.DEFAULT_MODEL
        # Assume ~2000 input tokens and ~500 output tokens as a typical call
        return cls.calculate_real_cost(model, input_tokens=2000, output_tokens=500)
    
    @classmethod
    def get_efficiency_status(cls, llm_calls: int) -> Dict[str, Any]:
        """Evaluate system efficiency"""
        if llm_calls <= 1:
            return {"status": "EXCELLENT", "emoji": "🏆", "color": "green"}
        elif llm_calls <= cls.TARGET_LLM_CALLS_PER_QUERY:
            return {"status": "EXCELLENT", "emoji": "✅", "color": "green"}
        elif llm_calls <= cls.MAX_LLM_CALLS_PER_QUERY:
            return {"status": "GOOD", "emoji": "✅", "color": "blue"}
        else:
            return {"status": "NEEDS_IMPROVEMENT", "emoji": "⚠️", "color": "orange"}

# ============================================================================
# Domain-Specific Configuration
# ============================================================================

DOMAIN_KEYWORDS = {
    "machine_learning": [
        "learning", "neural", "model", "training", "algorithm", "ai",
        "deep learning", "supervised", "unsupervised", "reinforcement",
        "classification", "regression", "clustering", "optimization"
    ],
    "computer_vision": [
        "vision", "image", "visual", "detection", "recognition", "cv",
        "object detection", "segmentation", "feature extraction", 
        "convolutional", "imaging", "video", "perception"
    ],
    "natural_language": [
        "language", "text", "nlp", "linguistic", "semantic", "parsing",
        "natural language processing", "tokenization", "embedding",
        "transformer", "attention", "sentiment", "translation"
    ],
    "robotics": [
        "robot", "autonomous", "control", "navigation", "manipulation",
        "robotics", "automation", "sensor", "actuator", "planning",
        "motion planning", "localization", "mapping", "kinematics"
    ],
    "cybersecurity": [
        "security", "attack", "vulnerability", "encryption", "malware",
        "cybersecurity", "threat", "intrusion", "firewall", "authentication",
        "cryptography", "network security", "penetration testing"
    ],
    "software_engineering": [
        "software", "engineering", "development", "testing", "architecture",
        "programming", "code", "design patterns", "agile", "devops",
        "software architecture", "system design", "quality assurance"
    ]
}

# ============================================================================
# Logging Configuration
# ============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# Apply logging config immediately when settings module is imported
logging.config.dictConfig(LOGGING_CONFIG)