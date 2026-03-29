"""
Synthesis Agent - Research Assistant System

Agent 3: Synthesizes research findings using LLM - EXACTLY 1 LLM call per query.
This is the ONLY agent that uses LLMs in the entire system.
"""

import json
import re
from typing import List, Dict, Any, Optional
from agents.base_agent import BaseAgent
from core.models import Paper, ResearchSynthesis
from core.llm_interface import LLMInterface, validate_llm_response
from core.prompts import create_synthesis_prompt
from config.settings import SystemConfig
import logging

logger = logging.getLogger(__name__)

class SynthesisAgent(BaseAgent):
    """
    Agent 3: Research synthesis using LLM - the only LLM-powered agent
    
    Takes processed papers and extracted data to create:
    - Cross-paper analysis and insights
    - Research gap identification  
    - Methodology comparisons
    - Coherent academic synthesis
    """
    
    def __init__(self, llm: LLMInterface):
        super().__init__("SynthesisAgent", uses_llm=True)
        self.llm = llm
        self.config = SystemConfig.SYNTHESIS_CONFIG
        self.synthesis_calls = 0
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for research synthesis
        
        Args:
            input_data: Dict containing 'query', 'papers', 'extracted_data'
            
        Returns:
            Dict with synthesis results and metadata
        """
        return self._execute_with_tracking(self._synthesize_research, input_data)
    
    def _synthesize_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research synthesis using LLM"""
        
        # Extract input components
        query = input_data.get("query", "")
        papers = input_data.get("papers", [])
        extracted_data = input_data.get("extracted_data", {})
        
        logger.info("Synthesizing research from %d papers...", len(papers))

        # Prepare papers for LLM analysis
        papers_summary = self._prepare_papers_for_synthesis(papers, extracted_data)

        logger.debug("Papers summary length: %d chars", len(papers_summary))
        for i, paper in enumerate(papers):
            logger.debug(
                "Paper %d: title=%r authors=%s year=%s text_len=%d",
                i + 1, paper.title, paper.authors, paper.year,
                len(getattr(paper, 'full_text', '') or ''),
            )
        
        # Create synthesis prompt
        messages = create_synthesis_prompt(query, papers_summary)
        
        # Make the LLM call (ONLY LLM call in entire system!)
        self.synthesis_calls += 1
        logger.info("LLM call #%d | chars=%d | papers=%d | query=%r",
                    self.synthesis_calls, len(papers_summary), len(papers), query)
        
        response = self.llm.make_call(messages, {"type": "json_object"})
        
        logger.debug("LLM response received: %s chars",
                     len(response.content) if response and response.content else 0)

        if response and response.content:
            logger.info("LLM response: %d chars", len(response.content))

            # Try to parse JSON
            try:
                parsed = json.loads(response.content)
                logger.debug("JSON parsed OK — keys: %s", list(parsed.keys()))
            except Exception as e:
                logger.error("JSON parse error: %s | raw: %.500s", e, response.content)
            
            synthesis = validate_llm_response(response.content, ResearchSynthesis)
            
            if synthesis:
                logger.info("Synthesis OK — findings=%d insights=%d gaps=%d confidence=%.2f",
                            len(synthesis.key_findings),
                            len(synthesis.methodology_insights),
                            len(synthesis.research_gaps),
                            synthesis.confidence_score)
                
                return {
                    "success": True,
                    "synthesis": synthesis,
                    "llm_calls_made": 1,
                    "papers_analyzed": len(papers),
                    "input_tokens_approx": len(papers_summary.split()),
                    "synthesis_confidence": synthesis.confidence_score,
                    "synthesis_completeness": self._assess_synthesis_completeness(synthesis),
                    "used_llm": True  # Flag to indicate LLM was used
                }
            else:
                logger.warning("LLM response validation failed — using deterministic fallback")
                logger.debug("Raw response (first 1000 chars): %.1000s",
                             response.content if response.content else 'No content')
                return self._create_fallback_synthesis(query, papers, extracted_data)
        else:
            logger.warning("LLM call returned None — using deterministic fallback")
            return self._create_fallback_synthesis(query, papers, extracted_data)
    
    def _prepare_papers_for_synthesis(self, papers: List[Paper], extracted_data: Dict[str, Any]) -> str:
        """
        Prepare comprehensive paper summary for LLM analysis
        
        Creates detailed summary with maximum content for thorough synthesis.
        """
        
        summary_parts = []
        
        # Research landscape overview
        metadata = extracted_data.get("metadata_analysis", {})
        summary_parts.append("COMPREHENSIVE RESEARCH LANDSCAPE OVERVIEW:")
        summary_parts.append(f"• {len(papers)} papers analyzed with detailed content extraction")
        summary_parts.append(f"• Publication years: {metadata.get('year_range', {}).get('min', 'N/A')}-{metadata.get('year_range', {}).get('max', 'N/A')}")
        summary_parts.append(f"• Top venues: {', '.join(list(metadata.get('venue_distribution', {}).keys())[:5])}")
        summary_parts.append(f"• Leading authors: {', '.join([author for author, _ in metadata.get('top_authors', [])[:5]])}")
        summary_parts.append(f"• Total citations analyzed: {extracted_data.get('citations_extracted', 0)}")
        summary_parts.append(f"• Key quotes extracted: {extracted_data.get('quotes_extracted', 0)}")
        summary_parts.append("")
        
        # Individual paper summaries - include MAXIMUM content for comprehensive analysis
        summary_parts.append("DETAILED PAPER ANALYSIS WITH FULL CONTENT:")
        top_papers = sorted(papers, key=lambda p: p.relevance_score, reverse=True)[:self.config["max_input_papers"]]
        
        for i, paper in enumerate(top_papers, 1):
            # Format paper summary
            authors_str = ", ".join(paper.authors[:3])  # Include more authors
            if len(paper.authors) > 3:
                authors_str += " et al."
            
            summary_parts.append(f"\n{'='*60}")
            summary_parts.append(f"PAPER {i}: {paper.title}")
            summary_parts.append(f"Authors: {authors_str} ({paper.year})")
            summary_parts.append(f"Venue: {paper.venue}")
            summary_parts.append(f"Relevance Score: {paper.relevance_score:.3f}")
            
            # Add comprehensive metadata
            if hasattr(paper, 'metadata'):
                meta = paper.metadata
                if 'page_range' in meta:
                    summary_parts.append(f"Pages Covered: {meta['page_range']}")
                if 'total_pages' in meta:
                    summary_parts.append(f"Total Pages: {meta['total_pages']}")
                if 'headings' in meta and meta['headings']:
                    summary_parts.append(f"Sections: {', '.join(list(meta['headings'])[:8])}")
            
            summary_parts.append(f"{'='*60}")
            
            # Include FULL content with ALL pages
            if hasattr(paper, 'full_text') and paper.full_text:
                summary_parts.append(f"\nCOMPLETE DOCUMENT CONTENT (ALL PAGES):")
                # Send up to 30k chars to include more pages
                summary_parts.append(paper.full_text[:30000])
                if len(paper.full_text) > 30000:
                    summary_parts.append(f"\n[Content continues... Total length: {len(paper.full_text)} chars]")
            else:
                summary_parts.append(f"\nABSTRACT:")
                summary_parts.append(paper.abstract)
            
            # Include ALL key quotes with full context
            if paper.key_quotes:
                summary_parts.append(f"\nKEY TECHNICAL INSIGHTS AND QUOTES:")
                for j, quote in enumerate(paper.key_quotes[:5], 1):  # Include up to 5 quotes
                    quote_text = quote.get("text", "") if isinstance(quote, dict) else quote.text
                    quote_type = quote.get("quote_type", "general") if isinstance(quote, dict) else getattr(quote, 'quote_type', 'general')
                    confidence = quote.get("confidence", 0.5) if isinstance(quote, dict) else getattr(quote, 'confidence', 0.5)
                    
                    summary_parts.append(f"   Quote {j} ({quote_type}, confidence: {confidence:.2f}):")
                    summary_parts.append(f"   \"{quote_text}\"")
                    summary_parts.append("")
            
            summary_parts.append(f"\n{'-'*60}\n")
        
        # Enhanced research insights from extraction
        insights = extracted_data.get("research_insights", {})
        if insights:
            summary_parts.append("COMPREHENSIVE EXTRACTED INSIGHTS:")
            
            quote_analysis = insights.get("quote_analysis", {})
            if quote_analysis:
                summary_parts.append(f"• Total quotes analyzed: {quote_analysis.get('total_quotes', 0)}")
                summary_parts.append(f"• Average quote confidence: {quote_analysis.get('average_confidence', 0):.3f}")
                summary_parts.append(f"• High-confidence quotes: {quote_analysis.get('high_confidence_quotes', 0)}")
                summary_parts.append(f"• Quote type distribution: {quote_analysis.get('quote_type_distribution', {})}")
            
            citation_analysis = insights.get("citation_analysis", {})
            if citation_analysis:
                summary_parts.append(f"• Total citations processed: {citation_analysis.get('total_citations', 0)}")
                summary_parts.append(f"• Citation time span: {citation_analysis.get('citation_time_span', 0)} years")
                summary_parts.append(f"• Average citation year: {citation_analysis.get('average_citation_year', 'N/A')}")
            
            temporal_analysis = insights.get("temporal_analysis", {})
            if temporal_analysis:
                summary_parts.append(f"• Paper year range: {temporal_analysis.get('paper_year_range', 'N/A')}")
                summary_parts.append(f"• Recency score: {temporal_analysis.get('recency_score', 0):.2f}")
        
        # Add citation network information if available
        citation_network = extracted_data.get("citation_network", {})
        if citation_network:
            summary_parts.append("\nCITATION NETWORK ANALYSIS:")
            summary_parts.append(f"• Total connections: {citation_network.get('total_connections', 0)}")
            summary_parts.append(f"• Unique authors in network: {citation_network.get('unique_authors', 0)}")
            summary_parts.append(f"• Research clusters identified: {len(citation_network.get('clusters', {}))}")
        
        final_summary = "\n".join(summary_parts)
        
        logger.debug("Summary: %d chars, ~%d tokens, %d papers",
                     len(final_summary), len(final_summary.split()), len(top_papers))
        return final_summary
    
    def _create_fallback_synthesis(self, query: str, papers: List[Paper], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic fallback synthesis when LLM fails.
        Extracts real sentences from document content to answer the query.
        """

        print("🔧 Creating deterministic fallback synthesis from document content...")

        top_papers = sorted(papers, key=lambda p: p.relevance_score, reverse=True)[:8]
        query_keywords = set(query.lower().split())

        key_findings = []
        methodology_insights = []
        technical_contributions = []

        for paper in top_papers:
            paper_title_clean = paper.title.replace("Document: ", "")
            source_text = (
                (getattr(paper, 'full_text', '') or '') or paper.abstract or ''
            )
            sentences = [s.strip() for s in source_text.replace('\n', ' ').split('.') if len(s.strip()) > 40]

            # Score sentences by keyword overlap with the query
            scored = []
            for s in sentences:
                overlap = len(query_keywords & set(s.lower().split()))
                if overlap > 0:
                    scored.append((overlap, s))
            scored.sort(key=lambda x: x[0], reverse=True)

            for _, sentence in scored[:3]:
                key_findings.append(f"[{paper_title_clean}] {sentence.strip()}")

            for _, sentence in scored[3:5]:
                methodology_insights.append(f"[{paper_title_clean}] {sentence.strip()}")

            for quote in paper.key_quotes[:2]:
                qt = quote.get("text", "") if isinstance(quote, dict) else getattr(quote, 'text', '')
                if len(qt) > 30:
                    technical_contributions.append(f"[{paper_title_clean}] {qt}")

        # If no keyword-matching sentences found, use top sentences from best paper
        if not key_findings and top_papers:
            best = top_papers[0]
            source = (getattr(best, 'full_text', '') or best.abstract or '')
            sentences = [s.strip() for s in source.replace('\n', ' ').split('.') if len(s.strip()) > 40]
            for s in sentences[:5]:
                key_findings.append(f"[{best.title.replace('Document: ', '')}] {s}")

        # If still nothing, be honest about it
        if not key_findings:
            key_findings = [
                f"The uploaded documents do not appear to contain direct information about: \"{query}\"",
                "Try uploading documents that are specifically related to your question.",
            ]
            methodology_insights = ["No relevant methodology found in the uploaded documents for this query."]
            research_gaps = [f"The current document set does not cover the topic: \"{query}\""]
        else:
            research_gaps = [
                "The documents may not fully cover all aspects of this question.",
                "Consider uploading additional sources for a more complete answer.",
            ]

        if not methodology_insights:
            methodology_insights = ["No specific methodology details found matching this query in the uploaded documents."]

        recommended_papers = [
            f"{p.title.replace('Document: ', '')} (relevance: {p.relevance_score:.2f})"
            for p in top_papers[:5]
        ]

        fallback_synthesis = ResearchSynthesis(
            research_question=query,
            key_findings=key_findings[:12],
            methodology_insights=methodology_insights[:8],
            research_gaps=research_gaps[:5],
            recommended_papers=recommended_papers,
            confidence_score=0.5,
            technical_contributions=technical_contributions[:6],
            comparative_analysis=[],
            practical_implications=[]
        )

        logger.info("Fallback synthesis — findings=%d insights=%d gaps=%d",
                    len(fallback_synthesis.key_findings),
                    len(fallback_synthesis.methodology_insights),
                    len(fallback_synthesis.research_gaps))

        return {
            "success": True,
            "synthesis": fallback_synthesis,
            "llm_calls_made": 0,
            "papers_analyzed": len(papers),
            "fallback_used": True,
            "synthesis_confidence": 0.5,
            "synthesis_completeness": self._assess_synthesis_completeness(fallback_synthesis),
            "used_llm": False
        }
    
    def _assess_synthesis_completeness(self, synthesis: ResearchSynthesis) -> Dict[str, Any]:
        """Assess the quality and completeness of enhanced synthesis"""
        
        completeness_scores = {
            "key_findings": len(synthesis.key_findings) >= 5,  # Increased minimum
            "methodology_insights": len(synthesis.methodology_insights) >= 3,  # Increased minimum
            "research_gaps": len(synthesis.research_gaps) >= 2,  # Increased minimum
            "recommended_papers": len(synthesis.recommended_papers) >= 2,
            "sufficient_confidence": synthesis.confidence_score >= 0.7,
            "technical_contributions": len(getattr(synthesis, 'technical_contributions', [])) >= 2,
            "comparative_analysis": len(getattr(synthesis, 'comparative_analysis', [])) >= 1,
            "practical_implications": len(getattr(synthesis, 'practical_implications', [])) >= 1,
            "comprehensive_findings": len(synthesis.key_findings) >= 8,  # Bonus for comprehensive output
            "detailed_insights": len(synthesis.methodology_insights) >= 5   # Bonus for detailed insights
        }
        
        total_criteria = len(completeness_scores)
        met_criteria = sum(completeness_scores.values())
        completeness_percentage = (met_criteria / total_criteria) * 100
        
        # Calculate content richness score
        total_content_items = (
            len(synthesis.key_findings) +
            len(synthesis.methodology_insights) +
            len(synthesis.research_gaps) +
            len(getattr(synthesis, 'technical_contributions', [])) +
            len(getattr(synthesis, 'comparative_analysis', [])) +
            len(getattr(synthesis, 'practical_implications', []))
        )
        
        return {
            "criteria_met": met_criteria,
            "total_criteria": total_criteria,
            "completeness_percentage": round(completeness_percentage, 1),
            "quality_rating": self._get_quality_rating(completeness_percentage),
            "detailed_scores": completeness_scores,
            "content_richness_score": total_content_items,
            "is_comprehensive": total_content_items >= 20  # Flag for comprehensive output
        }
    
    def _get_quality_rating(self, completeness_percentage: float) -> str:
        """Convert completeness percentage to quality rating"""
        if completeness_percentage >= 90:
            return "EXCELLENT"
        elif completeness_percentage >= 75:
            return "GOOD"
        elif completeness_percentage >= 60:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"
    
    def validate_input(self, input_data: Any) -> Dict[str, Any]:
        """Validate synthesis agent input"""
        
        if not isinstance(input_data, dict):
            return {"valid": False, "error": "Input must be a dictionary"}
        
        required_keys = ["query", "papers", "extracted_data"]
        for key in required_keys:
            if key not in input_data:
                return {"valid": False, "error": f"Missing required key: {key}"}
        
        query = input_data.get("query", "")
        if not query or len(query.strip()) < 3:
            return {"valid": False, "error": "Query must be at least 3 characters"}
        
        papers = input_data.get("papers", [])
        if not isinstance(papers, list) or len(papers) == 0:
            return {"valid": False, "error": "Papers list cannot be empty"}
        
        return {"valid": True}
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about synthesis performance"""
        
        metrics = self.get_performance_metrics()
        
        return {
            **metrics,
            "synthesis_specific_metrics": {
                "total_synthesis_calls": self.synthesis_calls,
                "llm_calls_per_synthesis": 1,  # Always exactly 1
                "max_input_papers": self.config["max_input_papers"],
                "max_tokens_per_call": self.config["max_tokens"],
                "temperature_setting": self.config["temperature"],
                "cost_per_synthesis": SystemConfig.get_cost_estimate(),
                "enhanced_output_enabled": True,  # Flag for enhanced output
                "comprehensive_fallback_enabled": True  # Flag for comprehensive fallback
            }
        }
    
    def diagnose_synthesis_issues(self, papers: List[Paper], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diagnostic function to identify potential issues with synthesis output
        """
        
        diagnostics = {
            "input_analysis": {
                "papers_count": len(papers),
                "papers_with_full_text": sum(1 for p in papers if hasattr(p, 'full_text') and p.full_text),
                "papers_with_quotes": sum(1 for p in papers if p.key_quotes),
                "total_content_length": sum(len(getattr(p, 'full_text', '')) for p in papers),
                "average_relevance_score": sum(p.relevance_score for p in papers) / len(papers) if papers else 0
            },
            "configuration_check": {
                "max_tokens_configured": self.config["max_tokens"],
                "temperature_setting": self.config["temperature"],
                "max_input_papers": self.config["max_input_papers"],
                "llm_interface_available": self.llm is not None
            },
            "potential_issues": [],
            "recommendations": []
        }
        
        # Identify potential issues
        if diagnostics["input_analysis"]["papers_count"] == 0:
            diagnostics["potential_issues"].append("No papers provided for synthesis")
            diagnostics["recommendations"].append("Ensure document processing completed successfully")
        
        if diagnostics["input_analysis"]["papers_with_full_text"] == 0:
            diagnostics["potential_issues"].append("No papers have full text content")
            diagnostics["recommendations"].append("Check document processing and text extraction")
        
        if diagnostics["input_analysis"]["total_content_length"] < 1000:
            diagnostics["potential_issues"].append("Very limited content available for synthesis")
            diagnostics["recommendations"].append("Upload more comprehensive documents or check text extraction quality")
        
        if diagnostics["configuration_check"]["max_tokens_configured"] < 4000:
            diagnostics["potential_issues"].append("Token limit may be too low for comprehensive synthesis")
            diagnostics["recommendations"].append("Consider increasing max_tokens in SYNTHESIS_CONFIG")
        
        return diagnostics