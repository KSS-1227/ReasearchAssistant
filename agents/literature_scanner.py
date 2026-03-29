```
"""
Literature Scanner Agent - Research Assistant System

Agent 1: Searches and ranks papers using deterministic vector similarity.
NO LLM calls - pure algorithmic approach for cost efficiency.
"""
```

import re
import math
from typing import List, Dict, Any, Set
from agents.base_agent import BaseAgent
from core.models import Paper, ResearchDomain
# Removed database dependency - using PDF data instead
from config.settings import SystemConfig, DOMAIN_KEYWORDS

class LiteratureScanner(BaseAgent):
    """
    Agent 1: Literature discovery and ranking using deterministic methods
    
    Uses sophisticated relevance scoring without LLM calls:
    - Term overlap analysis
    - Recency weighting  
    - Citation count boost
    - Domain-specific keyword expansion
    """
    
    def __init__(self):
        super().__init__("LiteratureScanner", uses_llm=False)
        # No database dependency - will use PDF data passed from coordinator
        self.config = SystemConfig.LITERATURE_CONFIG
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for literature scanning
        
        Args:
            input_data: Dict with 'query', 'domain', 'max_results' keys
            
        Returns:
            Dict with ranked papers and metadata
        """
        return self._execute_with_tracking(self._search_papers, input_data)
    
    def _search_papers(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute paper search with relevance ranking using PDF data or FAISS vector store"""
        
        # Extract parameters
        query = input_data.get("query", "")
        domain = ResearchDomain(input_data.get("domain", "other"))
        max_results = input_data.get("max_results", 10)
        pdf_papers = input_data.get("pdf_papers", [])  # Get PDF data from coordinator
        use_vector_store = input_data.get("use_vector_store", False)  # New parameter
        
        # Validate input
        validation = self.validate_input(input_data)
        if not validation["valid"]:
            return {"success": False, "error": validation["error"], "papers": []}
        
        print(f"🔍 Scanning literature for: '{query}' in {domain.value}")
        
        # If vector store is available and requested, use it
        if use_vector_store and hasattr(input_data.get('coordinator'), 'document_processor'):
            coordinator = input_data['coordinator']
            if coordinator.document_processor.vector_store:
                print("   🔍 Using FAISS vector store for semantic search...")
                vector_results = coordinator.document_processor.search_documents(query, k=max_results)
                
                # Group vector store results by source document
                document_groups = {}
                for result in vector_results:
                    metadata = result.get('metadata', {})
                    source_file = metadata.get('source_file', result.get('source', 'unknown'))
                    page = metadata.get('page', 0)
                    heading = metadata.get('heading', 'N/A')
                    
                    if source_file not in document_groups:
                        document_groups[source_file] = {
                            'chunks': [],
                            'best_relevance': 0.0,
                            'metadata': metadata,
                            'pages': set(),
                            'headings': set()
                        }
                    
                    relevance = 1.0 - result.get('similarity_score', 0.0)
                    document_groups[source_file]['chunks'].append({
                        'content': result.get('content', ''),
                        'relevance': relevance,
                        'page': page,
                        'heading': heading
                    })
                    document_groups[source_file]['best_relevance'] = max(
                        document_groups[source_file]['best_relevance'], 
                        relevance
                    )
                    document_groups[source_file]['pages'].add(page)
                    document_groups[source_file]['headings'].add(heading)
                
                # Convert grouped documents to Paper objects
                papers = []
                for doc_id, (source_file, doc_data) in enumerate(document_groups.items()):
                    # Sort chunks by page number
                    sorted_chunks = sorted(doc_data['chunks'], key=lambda x: x.get('page', 0))
                    
                    # Combine chunks with page/heading info preserved
                    combined_content = '\n\n'.join([
                        f"[Page {chunk['page']}, Section: {chunk['heading']}]\n{chunk['content']}" 
                        for chunk in sorted_chunks
                    ])
                    
                    original_filename = doc_data['metadata'].get('original_filename', source_file)
                    display_name = original_filename.split('/')[-1].split('\\')[-1] if '/' in original_filename or '\\' in original_filename else original_filename
                    
                    # Get page range from all chunks
                    pages = sorted(list(doc_data['pages']))
                    page_range = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0]) if pages else "N/A"
                    headings_list = list(doc_data['headings'])
                    
                    print(f"   📊 Document spans pages: {page_range}, Sections: {len(headings_list)}")
                    
                    paper = Paper(
                        id=f"doc_{doc_id}_{original_filename}",
                        title=f"Document: {display_name}",
                        authors=['Research Paper'],
                        abstract=combined_content[:500] + "..." if len(combined_content) > 500 else combined_content,
                        year=doc_data['metadata'].get('year', 2024),
                        venue=f"Research Document ({display_name})",
                        citations=[],
                        key_quotes=[]
                    )
                    paper.relevance_score = doc_data['best_relevance']
                    paper.full_text = combined_content  # Full content with all pages
                    paper.metadata = {
                        **doc_data['metadata'],
                        'source_file': source_file,
                        'chunk_count': len(doc_data['chunks']),
                        'document_type': 'uploaded',
                        'page_range': page_range,
                        'pages': pages,
                        'headings': headings_list,
                        'total_pages': len(pages)
                    }
                    papers.append(paper)
                
                # Sort by relevance score
                papers.sort(key=lambda x: x.relevance_score, reverse=True)
                
                return {
                    "success": True,
                    "papers": papers[:max_results],
                    "search_method": "vector_store",
                    "total_papers": len(papers),
                    "vector_store_used": True,
                    "query_expansion": {
                        "original_terms": len(query.split()),
                        "expanded_terms": len(query.split()),
                        "domain_keywords_added": 0
                    }
                }
        
        # Fallback to PDF papers if available
        if pdf_papers:
            print("   📄 Using uploaded PDF papers for analysis...")
            
            # Expand query with domain-specific terms
            expanded_query = self._expand_query(query, domain)
            
            # Calculate relevance scores for PDF papers
            scored_papers = []
            query_terms = self._extract_query_terms(expanded_query)
            
            for paper_data in pdf_papers:
                # Create Paper object from PDF data
                paper = Paper(
                    id=paper_data.get('id', 'unknown'),
                    title=paper_data.get('title', 'Unknown Title'),
                    authors=paper_data.get('authors', ['Unknown Author']),
                    abstract=paper_data.get('abstract', ''),
                    year=paper_data.get('year', 2024),
                    venue=paper_data.get('venue', 'Uploaded PDF'),
                    citations=paper_data.get('citations', []),  # Use citations list
                    key_quotes=[]  # Initialize empty key quotes
                )
                
                relevance = self._calculate_relevance_score(paper, query_terms)
                
                if relevance > self.config["min_relevance_threshold"]:
                    paper.relevance_score = relevance
                    scored_papers.append(paper)
            
            # Sort by relevance and return top results
            scored_papers.sort(key=lambda p: p.relevance_score, reverse=True)
            top_papers = scored_papers[:max_results]
            
            return {
                "success": True,
                "papers": top_papers,
                "search_method": "pdf_analysis",
                "total_papers": len(top_papers)
            }
        
        # No data available
        return {
            "success": False,
            "error": "No documents available for analysis. Please upload documents first.",
            "papers": [],
            "search_method": "none"
        }
    
    def _expand_query(self, query: str, domain: ResearchDomain) -> str:
        """
        Expand query with domain-specific keywords for better recall
        
        This deterministic expansion improves search without LLM calls.
        """
        query_lower = query.lower()
        domain_keywords = DOMAIN_KEYWORDS.get(domain.value, [])
        
        # Find domain keywords that might be relevant
        relevant_keywords = []
        for keyword in domain_keywords:
            # Add keyword if it's related to query terms
            if any(term in keyword for term in query_lower.split()):
                relevant_keywords.append(keyword)
        
        # Combine original query with relevant domain keywords
        expanded = query
        if relevant_keywords:
            expanded += " " + " ".join(relevant_keywords[:3])  # Limit expansion
        
        return expanded
    
    def _extract_query_terms(self, query: str) -> Set[str]:
        """Extract and normalize search terms from query"""
        
        # Convert to lowercase and split
        terms = set(query.lower().split())
        
        # Remove stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should"
        }
        terms = terms - stop_words
        
        # Remove very short terms
        terms = {term for term in terms if len(term) > 2}
        
        # Add important n-grams
        words = query.lower().split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 8:  # Only meaningful bigrams
                terms.add(bigram)
        
        return terms
    
    def _calculate_relevance_score(self, paper: Paper, query_terms: Set[str]) -> float:
        """
        Calculate paper relevance using multi-factor scoring
        
        Combines term overlap, recency, and citation metrics without LLM calls.
        """
        
        # Combine title, abstract for scoring (weight title more heavily)
        title_text = paper.title.lower()
        abstract_text = paper.abstract.lower()
        
        # Extract paper terms
        paper_terms = set((title_text + " " + abstract_text).split())
        
        # Calculate base relevance (term overlap)
        title_matches = len(query_terms.intersection(set(title_text.split())))
        abstract_matches = len(query_terms.intersection(set(abstract_text.split())))
        
        # Weight title matches more heavily
        total_matches = (title_matches * 2) + abstract_matches
        base_relevance = total_matches / len(query_terms) if query_terms else 0
        
        # Recency boost (newer papers get slight boost)
        current_year = 2024
        recency_years = max(0, paper.year - 2020)
        recency_boost = min(
            self.config["max_recency_boost"], 
            recency_years * self.config["recency_boost_factor"]
        )
        
        # Citation boost (well-cited papers get boost)
        citation_boost = min(
            self.config["max_citation_boost"],
            len(paper.citations) * self.config["citation_boost_factor"]
        )
        
        # Venue quality boost (simple heuristic)
        venue_boost = 0.1 if any(term in paper.venue.lower() for term in ["nature", "science", "acm", "ieee"]) else 0
        
        # Calculate total relevance
        total_relevance = base_relevance + recency_boost + citation_boost + venue_boost
        
        return min(1.0, total_relevance)  # Cap at 1.0
    
    def validate_input(self, input_data: Any) -> Dict[str, Any]:
        """Validate literature scanner input"""
        
        if not isinstance(input_data, dict):
            return {"valid": False, "error": "Input must be a dictionary"}
        
        required_keys = ["query", "domain"]
        for key in required_keys:
            if key not in input_data:
                return {"valid": False, "error": f"Missing required key: {key}"}
        
        query = input_data.get("query", "")
        if not query or len(query.strip()) < 3:
            return {"valid": False, "error": "Query must be at least 3 characters"}
        
        # Validate domain
        try:
            ResearchDomain(input_data["domain"])
        except ValueError:
            return {"valid": False, "error": f"Invalid domain: {input_data['domain']}"}
        
        # Validate max_results
        max_results = input_data.get("max_results", 10)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 50:
            return {"valid": False, "error": "max_results must be integer between 1-50"}
        
        return {"valid": True}
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about search performance"""
        
        metrics = self.get_performance_metrics()
        
        return {
            **metrics,
            "search_specific_metrics": {
                "pdf_papers_processed": 0,  # Will be updated when PDFs are loaded
                "domains_supported": len([d for d in ResearchDomain]),
                "relevance_threshold": self.config["min_relevance_threshold"],
                "max_results_per_search": self.config.get("max_papers_per_domain", 20)
            }
        }
    
    def search_papers_by_domain(self, domain: ResearchDomain, limit: int = None) -> List[Paper]:
        """
        Get papers from a specific domain (for analysis/debugging)
        
        Args:
            domain: Research domain to search
            limit: Maximum number of papers to return
            
        Returns:
            List of papers from the domain (empty since using PDFs)
        """
        # No database - return empty list since we're using PDF data
        return []