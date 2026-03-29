"""
Research Coordinator - Research Assistant System

Main coordinator that orchestrates all agents using deterministic routing.
NO LLM calls - uses pure logic for agent coordination.
"""

import os
import time
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from core.llm_interface import LLMInterface
from core.memory import ResearchMemory
from core.models import ResearchDomain, ResearchResult, ErrorResult, validate_research_query
from core.document_processor import DocumentProcessor
from agents.literature_scanner import LiteratureScanner
from agents.citation_extractor import CitationExtractor
from agents.synthesis_agent import SynthesisAgent
from config.settings import SystemConfig
import logging

logger = logging.getLogger(__name__)

class ResearchCoordinator:
    """
    Main system coordinator - orchestrates all agents deterministically
    
    Responsibilities:
    - Route research queries to appropriate agents
    - Manage session state and performance tracking
    - Coordinate agent execution order
    - Provide unified interface for the system
    """
    
    def __init__(self, api_key: str):
        """Initialize coordinator with all agents and systems"""
        
        # Load environment variables to get both API keys
        load_dotenv()
        gemini_key = api_key  # The passed API key
        openai_key = os.getenv("OPENAI_API_KEY", "")
        
        # Initialize core systems
        # Use Gemini key for LLM calls
        self.llm = LLMInterface(gemini_key)
        self.memory = ResearchMemory()
        
        # Initialize document processor with Google key for embeddings
        # Use the same Gemini API key for embeddings
        self.document_processor = DocumentProcessor(gemini_key)
        
        # Initialize all agents
        self.literature_scanner = LiteratureScanner()
        self.citation_extractor = CitationExtractor()
        self.synthesis_agent = SynthesisAgent(self.llm)
        
        # System metadata
        self.created_at = time.time()
        self.total_queries_processed = 0
        
        logger.info("Research Coordinator initialized")
        logger.info("  LiteratureScanner uses_llm=%s", self.literature_scanner.uses_llm)
        logger.info("  CitationExtractor  uses_llm=%s", self.citation_extractor.uses_llm)
        logger.info("  SynthesisAgent     uses_llm=%s", self.synthesis_agent.uses_llm)
        logger.info("  Target: <= %d LLM calls per query", SystemConfig.MAX_LLM_CALLS_PER_QUERY)
    
    def research_query(self, 
                      query: str, 
                      domain: str = "machine_learning", 
                      max_papers: int = 8,
                      pdf_papers: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point - processes research query end-to-end
        
        Args:
            query: Research question or topic
            domain: Research domain (machine_learning, computer_vision, etc.)
            max_papers: Maximum papers to analyze
            
        Returns:
            Complete research results with performance metrics
        """
        
        start_time = time.time()
        self.total_queries_processed += 1
        
        # Validate input
        validation = validate_research_query(query)
        if not validation["valid"]:
            return self._create_error_response("INVALID", validation["error"])
        
        query = validation["cleaned_query"]
        
        # Classify domain deterministically
        research_domain = self._classify_domain(query, domain)
        
        # Create research session
        session_id = self.memory.create_session(query, research_domain)
        
        logger.info("Research query #%d | domain=%s | session=%s",
                    self.total_queries_processed, research_domain.value, session_id)
        
        try:
            # Execute the three-agent pipeline
            result = self._execute_research_pipeline(session_id, query, research_domain, max_papers)
            
            # Update session with final metrics
            processing_time = time.time() - start_time
            self.memory.update_session(
                session_id, 
                total_llm_calls=self.llm.call_count,
                total_cost=self.llm.total_cost,
                processing_time=processing_time
            )
            
            # Add final performance metrics
            result["performance_metrics"]["processing_time"] = round(processing_time, 2)
            result["performance_metrics"]["efficiency_rating"] = SystemConfig.get_efficiency_status(
                result["performance_metrics"]["total_llm_calls"]
            )
            
            self._print_research_summary(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Research pipeline failed: {str(e)}"
            logger.error("Research pipeline failed: %s", error_msg)
            
            return self._create_error_response(session_id, error_msg, processing_time)
    
    def search_uploaded_documents(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search through uploaded documents using FAISS vector store
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Search results with document chunks and similarity scores
        """
        try:
            results = self.document_processor.search_documents(query, k=max_results)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "llm_calls_made": self.document_processor.llm_call_count
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": [],
                "total_results": 0,
                "llm_calls_made": self.document_processor.llm_call_count
            }
    
    def get_document_processing_stats(self) -> Dict[str, Any]:
        """Get current document processing statistics"""
        return self.document_processor.get_processing_stats()
    
    def process_document(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """
        Process a document and add to FAISS vector store
        
        Args:
            file_path: Path to the document file
            file_type: Type of document (pdf, txt, etc.)
            
        Returns:
            Processing result with metadata
        """
        try:
            result = self.document_processor.process_document(file_path, file_type)
            
            # Update LLM call count in coordinator
            if result.get('success'):
                self.llm.call_count += result.get('llm_calls_made', 0)
            
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "file_type": file_type
            }
    
    def _execute_research_pipeline(self, 
                                  session_id: str, 
                                  query: str, 
                                  domain: ResearchDomain, 
                                  max_papers: int) -> Dict[str, Any]:
        """
        Execute the three-agent research pipeline deterministically
        
        Pipeline: Literature Scanner → Citation Extractor → Synthesis Agent
        """
        
        # Step 1: Literature Scanner (0 LLM calls)
        logger.info("LiteratureScanner: searching for relevant papers...")
        
        scanner_input = {
            "query": query,
            "domain": domain.value,
            "max_results": max_papers,
            "coordinator": self,  # Pass coordinator reference for vector store access
            "use_vector_store": True  # Enable vector store usage
        }
        
        scanner_result = self.literature_scanner.process(scanner_input)
        
        if not scanner_result.get("success", False):
            raise Exception(f"Literature scan failed: {scanner_result.get('error', 'Unknown error')}")
        
        papers_found = scanner_result["papers"]
        if not papers_found:
            raise Exception("No relevant papers found for the query")
        
        # Update session with papers
        self.memory.update_session(session_id, papers_found=papers_found)
        
        logger.info("LiteratureScanner: found %d papers", len(papers_found))
        
        # Step 2: Citation Extractor (0 LLM calls)
        logger.info("CitationExtractor: processing citations and quotes...")
        
        extraction_result = self.citation_extractor.process(papers_found)
        
        if not extraction_result.get("success", False):
            raise Exception(f"Citation extraction failed: {extraction_result.get('error', 'Unknown error')}")
        
        logger.info("CitationExtractor: %d citations, %d quotes",
                    extraction_result['citations_extracted'],
                    extraction_result['quotes_extracted'])
        logger.debug("Using enhanced papers with metadata extraction")
        
        # Step 3: Synthesis Agent (1 LLM call)
        logger.info("SynthesisAgent: creating research synthesis...")
        
        synthesis_input = {
            "query": query,
            "papers": extraction_result["enhanced_papers"],  # Use enhanced papers from extraction result
            "extracted_data": extraction_result
        }
        
        synthesis_result = self.synthesis_agent.process(synthesis_input)
        
        if not synthesis_result.get("success", False):
            raise Exception(f"Research synthesis failed: {synthesis_result.get('error', 'Unknown error')}")
        
        synthesis = synthesis_result["synthesis"]
        logger.info("SynthesisAgent: %d findings generated", len(synthesis.key_findings))
        
        # Update session with synthesis
        self.memory.update_session(session_id, synthesis=synthesis)
        
        # Compile comprehensive result
        return self._compile_research_result(
            session_id, query, domain, papers_found, 
            extraction_result, synthesis_result
        )
    
    def _compile_research_result(self, 
                                session_id: str,
                                query: str, 
                                domain: ResearchDomain,
                                papers: List,
                                extraction_result: Dict[str, Any],
                                synthesis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive research result"""
        
        from core.models import create_paper_summary
        
        synthesis = synthesis_result["synthesis"]
        
        return {
            "session_id": session_id,
            "success": True,
            "query": query,
            "domain": domain.value,
            "papers_found": {
                "count": len(papers),
                "papers": [create_paper_summary(paper) for paper in papers]
            },
            "extracted_insights": {
                "total_citations": extraction_result["citations_extracted"],
                "total_quotes": extraction_result["quotes_extracted"],
                "key_quotes": extraction_result["key_quotes"][:10],  # Top 10 quotes
                "top_authors": extraction_result["metadata_analysis"]["top_authors"],
                "venues": list(extraction_result["metadata_analysis"]["venue_distribution"].keys()),
                "year_span": f"{extraction_result['metadata_analysis']['year_range']['min']}-{extraction_result['metadata_analysis']['year_range']['max']}",
                "citation_network": extraction_result["citation_network"],
                "research_insights": extraction_result["research_insights"]
            },
            "research_synthesis": {
                "key_findings": synthesis.key_findings,
                "methodology_insights": synthesis.methodology_insights,
                "research_gaps": synthesis.research_gaps,
                "recommended_papers": synthesis.recommended_papers,
                "confidence": synthesis.confidence_score,
                "completeness": synthesis_result["synthesis_completeness"]
            },
            "performance_metrics": {
                "total_llm_calls": self.llm.call_count,
                "estimated_cost": self.llm.total_cost,
                "papers_analyzed": len(papers),
                "agents_used": 3,
                "llm_agent_calls": synthesis_result["llm_calls_made"],
                "deterministic_agent_calls": 2  # Scanner + Extractor
            }
        }
    
    def _classify_domain(self, query: str, domain_hint: str) -> ResearchDomain:
        """
        Deterministic domain classification using keyword matching
        
        Uses if/else logic instead of LLM for cost efficiency.
        """
        
        query_lower = query.lower()
        
        # Import domain keywords
        from config.settings import DOMAIN_KEYWORDS
        
        # Score each domain based on keyword matches
        domain_scores = {}
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Weight longer keywords more heavily
                    score += len(keyword.split())
            domain_scores[domain] = score
        
        # Find best matching domain
        if domain_scores and max(domain_scores.values()) > 0:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            try:
                return ResearchDomain(best_domain)
            except ValueError:
                pass
        
        # Fall back to provided hint
        try:
            return ResearchDomain(domain_hint)
        except ValueError:
            return ResearchDomain.OTHER
    
    def _create_error_response(self, session_id: str, error_msg: str, processing_time: float = 0.0) -> Dict[str, Any]:
        """Create standardized error response"""
        
        return {
            "session_id": session_id,
            "success": False,
            "error": error_msg,
            "performance_metrics": {
                "total_llm_calls": self.llm.call_count,
                "estimated_cost": self.llm.total_cost,
                "processing_time": processing_time,
                "agents_used": 0
            },
            "timestamp": time.time()
        }
    
    def _print_research_summary(self, result: Dict[str, Any]):
        """Log comprehensive research summary at INFO level."""
        if not result["success"]:
            logger.error("Research failed: %s", result.get('error', 'Unknown error'))
            return

        synthesis = result["research_synthesis"]
        metrics   = result["performance_metrics"]

        logger.info("Research complete | papers=%d domain=%s",
                    result['papers_found']['count'], result['domain'])
        for i, finding in enumerate(synthesis["key_findings"][:3], 1):
            logger.info("  Finding %d: %.80s", i, finding)
        logger.info("LLM calls=%d cost=$%.4f time=%ss efficiency=%s",
                    metrics['total_llm_calls'],
                    metrics['estimated_cost'],
                    metrics['processing_time'],
                    metrics['efficiency_rating']['status'])
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        system_metrics = self.memory.get_system_metrics()
        efficiency_report = self.memory.get_efficiency_report()
        domain_stats = self.memory.get_domain_statistics()
        
        # Get document processing stats
        doc_stats = self.get_document_processing_stats()
        
        return {
            "total_research_sessions": system_metrics.total_sessions,
            "total_llm_calls": system_metrics.total_llm_calls,
            "total_cost": system_metrics.total_cost,
            "average_llm_calls_per_session": system_metrics.average_llm_calls_per_session,
            "efficiency_score": system_metrics.efficiency_rating,
            "agent_performance": {
                agent_name: {
                    "calls_made": metrics.calls_made,
                    "llm_calls": metrics.llm_calls_made,
                    "success_rate": metrics.success_rate,
                    "avg_processing_time": round(metrics.processing_time / max(1, metrics.calls_made), 3)
                }
                for agent_name, metrics in system_metrics.agent_metrics.items()
            },
            "domain_statistics": domain_stats,
            "efficiency_analysis": efficiency_report,
            "document_processing": {
                "total_documents": doc_stats["total_documents"],
                "total_chunks": doc_stats["total_chunks"],
                "vector_store_size": doc_stats["vector_store_size"],
                "llm_calls_for_embeddings": doc_stats["llm_calls_made"]
            }
        }
    
    def reset_system(self):
        """Reset all system counters and history - useful for testing"""
        self.llm.reset_counters()
        self.memory.clear_history()
        self.literature_scanner.reset_metrics()
        self.citation_extractor.reset_metrics()
        self.synthesis_agent.reset_metrics()
        # Reset document processor
        if hasattr(self, 'document_processor'):
            self.document_processor.reset_processor()
        self.total_queries_processed = 0
        logger.info("System reset complete")
    
    def validate_system_architecture(self) -> Dict[str, Any]:
        """Validate that system meets project requirements"""
        
        agents = {
            "LiteratureScanner": self.literature_scanner,
            "CitationExtractor": self.citation_extractor,
            "SynthesisAgent": self.synthesis_agent
        }
        
        total_agents = len(agents)
        llm_agents = sum(1 for agent in agents.values() if agent.uses_llm)
        deterministic_agents = total_agents - llm_agents
        
        # Check requirements
        requirements = {
            "min_3_agents": total_agents >= 3,
            "max_2_llm_agents": llm_agents <= 2,
            "min_50_percent_deterministic": (deterministic_agents / total_agents) >= 0.5,
            "deterministic_routing": True  # Coordinator uses no LLMs
        }
        
        return {
            "total_agents": total_agents,
            "llm_agents": llm_agents,
            "deterministic_agents": deterministic_agents,
            "deterministic_percentage": round((deterministic_agents / total_agents) * 100, 1),
            "requirements": requirements,
            "overall_compliance": all(requirements.values()),
            "architecture_summary": f"{deterministic_agents}/{total_agents} agents are deterministic ({(deterministic_agents/total_agents)*100:.1f}%)"
        }
    
    def get_agent_execution_flow(self) -> Dict[str, Any]:
        """Get detailed information about agent execution flow"""
        
        return {
            "execution_order": [
                {
                    "step": 1,
                    "agent": "LiteratureScanner",
                    "function": "Search and rank relevant papers",
                    "llm_calls": 0,
                    "input": "Research query + domain",
                    "output": "Ranked list of relevant papers"
                },
                {
                    "step": 2,
                    "agent": "CitationExtractor", 
                    "function": "Extract citations and key quotes",
                    "llm_calls": 0,
                    "input": "List of papers",
                    "output": "Citations, quotes, and metadata"
                },
                {
                    "step": 3,
                    "agent": "SynthesisAgent",
                    "function": "Synthesize research insights",
                    "llm_calls": 1,
                    "input": "Papers + extracted data",
                    "output": "Research synthesis and insights"
                }
            ],
            "total_pipeline_llm_calls": 1,
            "deterministic_steps": 2,
            "llm_powered_steps": 1,
            "data_flow": "Query → Papers → Citations/Quotes → Synthesis → Results"
        }
    
    def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        
        diagnostics = {
            "timestamp": time.time(),
            "system_health": {},
            "agent_health": {},
            "configuration_check": {},
            "performance_analysis": {}
        }
        
        # System health checks
        diagnostics["system_health"] = {
            "llm_interface_ready": self.llm.client is not None,
            "api_key_configured": SystemConfig.validate_api_key(),
            "memory_initialized": self.memory is not None,
            "agents_initialized": all([
                self.literature_scanner is not None,
                self.citation_extractor is not None,
                self.synthesis_agent is not None
            ])
        }
        
        # Agent health checks
        for agent_name, agent in [
            ("LiteratureScanner", self.literature_scanner),
            ("CitationExtractor", self.citation_extractor),
            ("SynthesisAgent", self.synthesis_agent)
        ]:
            diagnostics["agent_health"][agent_name] = {
                "initialized": agent is not None,
                "uses_llm_correctly": agent.uses_llm == (agent_name == "SynthesisAgent"),
                "performance_data_available": hasattr(agent, 'calls_made')
            }
        
        # Configuration validation
        diagnostics["configuration_check"] = {
            "max_llm_calls_target": SystemConfig.MAX_LLM_CALLS_PER_QUERY,
            "cost_per_call": SystemConfig.get_cost_estimate(),
            "literature_config_valid": bool(SystemConfig.LITERATURE_CONFIG),
            "synthesis_config_valid": bool(SystemConfig.SYNTHESIS_CONFIG)
        }
        
        # Performance analysis
        diagnostics["performance_analysis"] = self.get_system_stats()
        
        # Overall system status
        all_health_checks = []
        for category in ["system_health", "agent_health"]:
            for check_dict in diagnostics[category].values():
                if isinstance(check_dict, dict):
                    all_health_checks.extend(check_dict.values())
                else:
                    all_health_checks.append(check_dict)
        
        diagnostics["overall_status"] = {
            "healthy": all(all_health_checks),
            "health_percentage": (sum(all_health_checks) / len(all_health_checks)) * 100 if all_health_checks else 0,
            "ready_for_production": all(all_health_checks) and SystemConfig.validate_api_key()
        }
        
        return diagnostics
    
    def research_query_with_pdfs(self, 
                                query: str, 
                                pdf_papers: List[Dict[str, Any]], 
                                domain: str = "machine_learning") -> Dict[str, Any]:
        """
        Process research query using uploaded PDF papers instead of database
        
        Args:
            query: Research question or topic
            pdf_papers: List of PDF paper data dictionaries
            domain: Research domain (default: machine_learning)
            
        Returns:
            Complete research results using PDF data
        """
        
        if not pdf_papers:
            return {
                "success": False,
                "error": "No PDF papers provided for analysis"
            }
        
        logger.info("Processing query with %d PDF papers | domain=%s", len(pdf_papers), domain)
        
        # Use the existing research pipeline but with PDF data
        return self.research_query(query, domain, len(pdf_papers), pdf_papers)