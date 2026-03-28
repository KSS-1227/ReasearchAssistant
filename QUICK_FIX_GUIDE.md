# Quick Fix Guide - Critical Issues

## 🚨 CRITICAL: Fix These First (Next 1-2 hours)

### Issue #1: Undefined Variable `enhanced_papers`
**Location**: [core/coordinator.py](core/coordinator.py#L249)

**Quick Fix**:
```python
# Line 249 - CHANGE THIS:
synthesis_input = {
    "query": query,
    "papers": enhanced_papers,  # ❌ WRONG - undefined
    "extracted_data": extraction_result
}

# TO THIS:
synthesis_input = {
    "query": query,
    "papers": extraction_result["enhanced_papers"],  # ✅ CORRECT
    "extracted_data": extraction_result
}
```

**Test**: Run any research query - it should complete synthesis instead of crashing.

---

### Issue #2: Missing None Check After LLM Call
**Location**: [core/synthesis_agent.py](core/synthesis_agent.py#L90-L95)

**Quick Fix**:
```python
# Line 90-95 - CHANGE THIS:
response = self.llm.make_call(messages, {"type": "json_object"})

if response and response.content:
    logger.info("LLM response: %d chars", len(response.content))
    # ... code that uses response.content

# TO THIS:
response = self.llm.make_call(messages, {"type": "json_object"})

if response is None:
    logger.warning("LLM call returned None — using deterministic fallback")
    return self._create_fallback_synthesis(query, papers, extracted_data)

if not response.content:
    logger.warning("LLM response empty — using deterministic fallback")
    return self._create_fallback_synthesis(query, papers, extracted_data)

# Now safe to use response
logger.info("LLM response: %d chars", len(response.content))
# ... rest of code
```

**Test**: Simulate LLM failure by stopping Gemini API - app should use fallback synthesis.

---

### Issue #3: Inconsistent Dict vs Object Access
**Location**: [core/synthesis_agent.py](core/synthesis_agent.py#L248-L260)

**Quick Fix** - Standardize on dicts everywhere:

```python
# In citation_extractor.py - Line 50, CHANGE THIS:
paper.key_quotes = [quote.__dict__ for quote in paper_quotes]  # ✅ Already correct

# In synthesis_agent.py - Line 248-260, CHANGE THIS:
for j, quote in enumerate(paper.key_quotes[:5], 1):
    # This defensive check shouldn't be needed if we standardize
    quote_text = quote.get("text", "") if isinstance(quote, dict) else quote.text
    quote_type = quote.get("quote_type", "general") if isinstance(quote, dict) else getattr(quote, 'quote_type', 'general')

# TO THIS - Safe dict access everywhere:
for j, quote in enumerate(paper.key_quotes[:5], 1):
    # Assume all quotes are dicts (standardized format)
    if not isinstance(quote, dict):
        logger.warning("Quote is not a dict: %s", type(quote))
        continue
    
    quote_text = quote.get("text", "")
    quote_type = quote.get("quote_type", "general")
    confidence = quote.get("confidence", 0.5)
```

**Test**: Process papers through full pipeline and verify quotes display correctly.

---

## ⚠️ HIGH PRIORITY: After Critical Fixes (Next 4 hours)

### Issue #4: Vector Store Error Handling
**Location**: [core/document_processor.py](core/document_processor.py#L170-180)

**Quick Fix**:
```python
# Add try-catch around vector store operations
try:
    if self.vector_store is None:
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
    else:
        self.vector_store.add_documents(chunks)
    self.llm_call_count += len(chunks)
except Exception as e:
    logger.error("Vector store operation failed: %s", str(e))
    raise RuntimeError(f"Failed to update vector store: {str(e)}")
```

---

### Issue #5: Vector Store Null Check
**Location**: [core/document_processor.py](core/document_processor.py#L425)

**Quick Fix**:
```python
def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not self.vector_store:
        logger.debug("No vector store available - returning empty results")
        return []
    
    try:
        results = self.vector_store.similarity_search_with_scores(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata, "score": score} 
                for doc, score in results]
    except Exception as e:
        logger.error("Vector store search failed: %s", str(e))
        return []  # Graceful degradation
```

---

### Issue #6: Input Validation for PDF Data
**Location**: [core/literature_scanner.py](core/literature_scanner.py#L220-238)

**Quick Fix**:
```python
for paper_data in pdf_papers:
    if not isinstance(paper_data, dict):
        logger.warning("Invalid paper data format")
        continue
    
    # Ensure authors is a list
    authors = paper_data.get('authors', ['Unknown Author'])
    if isinstance(authors, str):
        authors = [authors]
    elif not isinstance(authors, list):
        authors = ['Unknown Author']
    
    # Ensure year is int
    try:
        year = int(paper_data.get('year', 2024))
    except (ValueError, TypeError):
        year = 2024
    
    paper = Paper(
        id=str(paper_data.get('id', 'unknown')),
        title=str(paper_data.get('title', 'Unknown Title')),
        authors=authors,
        abstract=str(paper_data.get('abstract', '')),
        year=year,
        venue=str(paper_data.get('venue', 'Uploaded PDF')),
        citations=paper_data.get('citations', []),
        key_quotes=[]
    )
```

---

### Issue #7: Paper Dataclass Optional Fields
**Location**: [core/models.py](core/models.py#L43-55)

**Quick Fix**:
```python
from typing import Optional

@dataclass
class Paper:
    id: str
    title: str
    authors: List[str]
    abstract: str
    year: int
    venue: str
    citations: List[str]
    key_quotes: List[Dict[str, Any]]
    relevance_score: float = 0.0
    full_text: Optional[str] = None  # ✅ Add this
    metadata: Optional[Dict[str, Any]] = None  # ✅ Add this
    
    def __post_init__(self):
        if not self.key_quotes:
            self.key_quotes = []
        if self.metadata is None:
            self.metadata = {}
        if self.full_text is None:
            self.full_text = ""
```

---

## Quick Verification

Run these commands after each fix:

```bash
# Check for syntax errors
python -m py_compile core/coordinator.py
python -m py_compile core/synthesis_agent.py
python -m py_compile core/citation_extractor.py

# Run basic tests
python -m pytest tests/ -v

# Start the app
streamlit run streamlit_app.py
```

---

## Deployment Checklist

- [ ] Fix Issue #1: `enhanced_papers` undefined
- [ ] Fix Issue #2: LLM response None check
- [ ] Fix Issue #3: Dict/object access standardization
- [ ] Fix Issue #4: Vector store error handling
- [ ] Fix Issue #5: Vector store null checks
- [ ] Fix Issue #6: PDF input validation
- [ ] Fix Issue #7: Paper dataclass options
- [ ] Run full pipeline test
- [ ] Verify all tests pass
- [ ] Deploy to staging
- [ ] Run integration tests
- [ ] Deploy to production

---

## Emergency Rollback

If deployment fails:
1. Revert coordinator.py, synthesis_agent.py, models.py
2. Verify with previous version of CODE_REVIEW_SENIOR_ANALYSIS.md
3. Create incident ticket
4. Do not attempt fixes in production
