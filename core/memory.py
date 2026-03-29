"""
Session Memory Management for Research Assistant System

Handles session state, research history, and performance tracking.
"""

import time
from typing import Dict, Optional, List, Any
from datetime import datetime
import logging
from core.models import ResearchSession, ResearchDomain, AgentMetrics, SystemMetrics
from config.settings import SystemConfig

logger = logging.getLogger(__name__)

class ResearchMemory:
    """
    Deterministic session state management
    
    Tracks all research sessions, agent performance, and system metrics
    without requiring any LLM calls.
    """
    
    def __init__(self):
        self.sessions: Dict[str, ResearchSession] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.session_counter = 0
        
        # Initialize agent metrics
        self._initialize_agent_metrics()
    
    def create_session(self, query: str, domain: ResearchDomain) -> str:
        """Create new research session"""
        self.session_counter += 1
        session_id = f"RESEARCH-{int(time.time())}-{self.session_counter:03d}"
        
        session = ResearchSession(
            session_id=session_id,
            query=query,
            domain=domain,
            created_at=datetime.now().isoformat(),
            papers_found=[]
        )
        
        self.sessions[session_id] = session
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Retrieve session by ID"""
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, **kwargs):
        """Update session with new data"""
        if session_id in self.sessions:
            for key, value in kwargs.items():
                if hasattr(self.sessions[session_id], key):
                    setattr(self.sessions[session_id], key, value)
    
    def get_recent_sessions(self, limit: int = 10) -> List[ResearchSession]:
        """Get most recent research sessions"""
        sessions = list(self.sessions.values())
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[:limit]
    
    def get_sessions_by_domain(self, domain: ResearchDomain) -> List[ResearchSession]:
        """Get all sessions for a specific research domain"""
        return [session for session in self.sessions.values() if session.domain == domain]
    
    def _initialize_agent_metrics(self):
        """Initialize metrics tracking for each agent"""
        agents = [
            ("LiteratureScanner", False),
            ("CitationExtractor", False), 
            ("SynthesisAgent", True),
            ("Coordinator", False)
        ]
        
        for agent_name, uses_llm in agents:
            self.agent_metrics[agent_name] = AgentMetrics(
                agent_name=agent_name,
                uses_llm=uses_llm,
                calls_made=0,
                llm_calls_made=0,
                processing_time=0.0,
                success_rate=1.0
            )
    
    def update_agent_metrics(self, agent_name: str, 
                           processing_time: float, 
                           llm_calls: int = 0, 
                           success: bool = True):
        """Update performance metrics for an agent"""
        
        if agent_name in self.agent_metrics:
            metrics = self.agent_metrics[agent_name]
            metrics.calls_made += 1
            metrics.llm_calls_made += llm_calls
            metrics.processing_time += processing_time
            metrics.last_execution = datetime.now().isoformat()
            
            # Update success rate (simple moving average)
            current_success_rate = metrics.success_rate
            metrics.success_rate = (current_success_rate * (metrics.calls_made - 1) + int(success)) / metrics.calls_made
    
    def get_system_metrics(self) -> SystemMetrics:
        """Calculate comprehensive system performance metrics"""
        return SystemMetrics.calculate_from_sessions(self.sessions, self.agent_metrics)
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about research domains processed"""
        
        domain_counts = {}
        total_papers_by_domain = {}
        
        for session in self.sessions.values():
            domain = session.domain.value
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            total_papers_by_domain[domain] = total_papers_by_domain.get(domain, 0) + len(session.papers_found)
        
        return {
            "domains_queried": len(domain_counts),
            "domain_distribution": domain_counts,
            "papers_per_domain": total_papers_by_domain,
            "most_popular_domain": max(domain_counts.items(), key=lambda x: x[1])[0] if domain_counts else None
        }
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """Generate detailed efficiency analysis"""
        
        if not self.sessions:
            return {"status": "no_data", "message": "No sessions to analyze"}
        
        # Calculate efficiency metrics
        total_sessions = len(self.sessions)
        total_llm_calls = sum(session.total_llm_calls for session in self.sessions.values())
        total_cost = sum(session.total_cost for session in self.sessions.values())
        
        avg_llm_calls = total_llm_calls / total_sessions
        avg_cost = total_cost / total_sessions
        
        # Efficiency benchmarks
        efficiency_status = SystemConfig.get_efficiency_status(int(avg_llm_calls))
        
        # Cost comparison with traditional systems
        traditional_calls_per_query = 8  # Typical research system
        traditional_cost = traditional_calls_per_query * SystemConfig.get_cost_estimate()
        cost_savings = ((traditional_cost - avg_cost) / traditional_cost) * 100
        
        return {
            "status": "success",
            "total_sessions": total_sessions,
            "average_llm_calls_per_query": round(avg_llm_calls, 2),
            "average_cost_per_query": round(avg_cost, 4),
            "efficiency_rating": efficiency_status["status"],
            "meets_target": avg_llm_calls <= SystemConfig.MAX_LLM_CALLS_PER_QUERY,
            "cost_savings_vs_traditional": round(cost_savings, 1),
            "total_cost": round(total_cost, 4),
            "cost_efficiency": "EXCELLENT" if avg_cost < 0.005 else "GOOD"
        }
    
    def clear_history(self):
        """Clear all session history - useful for testing"""
        self.sessions.clear()
        self.session_counter = 0
        self._initialize_agent_metrics()
        logger.info("Session history cleared")
    
    def export_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session data for external analysis"""
        
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_metadata": {
                "id": session.session_id,
                "query": session.query,
                "domain": session.domain.value,
                "created_at": session.created_at,
                "processing_time": session.processing_time
            },
            "papers_data": [
                {
                    "id": paper.id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "relevance_score": paper.relevance_score,
                    "key_quotes": paper.key_quotes
                }
                for paper in session.papers_found
            ],
            "synthesis_results": session.synthesis.model_dump() if session.synthesis else None,
            "performance_metrics": {
                "llm_calls": session.total_llm_calls,
                "cost": session.total_cost,
                "efficiency_rating": SystemConfig.get_efficiency_status(session.total_llm_calls)["status"]
            }
        }