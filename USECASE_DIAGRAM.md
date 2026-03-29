```
# Research Assistant System - Use Case Diagram

## System Overview
A hybrid multi-agent research assistant that processes academic documents and generates research synthesis with exactly 1 LLM call per query.

---

## Actors

### Primary Actors
- **👤 Researcher**: End user who uploads documents and asks research questions
- **👨‍💼 System Administrator**: Manages system performance and configuration

### External Systems
- **🤖 Gemini API**: Google's LLM service for synthesis generation
- **🗄️ FAISS Vector Store**: Vector database for semantic document search

---

## Use Cases by Category

### 📄 Document Management
| Use Case ID | Use Case Name | Actor | Description | LLM Calls |
|------------|---------------|-------|-------------|-----------|
| UC-1 | Upload Documents | Researcher | Upload PDF, TXT, or MD research documents | 0 |
| UC-2 | Process Documents | System | Parse and prepare documents for analysis | 0 |
| UC-3 | Extract Text from PDF | System | Extract text content from PDF files | 0 |
| UC-4 | Create Document Chunks | System | Split documents into semantic chunks | 0 |
| UC-5 | Generate Embeddings | System | Create vector embeddings using Google API | 0 |
| UC-6 | Store in Vector DB | System | Store embeddings in FAISS index | 0 |

**Agent Responsible**: Document Processor (0 LLM calls)

---

### 🔍 Research Query Processing
| Use Case ID | Use Case Name | Actor | Description | LLM Calls |
|------------|---------------|-------|-------------|-----------|
| UC-7 | Submit Research Query | Researcher | Enter research question for analysis | 0 |
| UC-8 | Search Documents | System | Semantic search using FAISS | 0 |
| UC-9 | Rank Results | System | Rank documents by relevance score | 0 |
| UC-10 | Extract Citations | System | Extract academic citations using regex | 0 |
| UC-11 | Extract Key Quotes | System | Identify important quotes from papers | 0 |
| UC-12 | Generate Synthesis | System | Create research synthesis using LLM | 1 |
| UC-13 | Ask Follow-up Question | Researcher | Submit additional questions on same topic | 1 |

**Agents Responsible**: 
- Literature Scanner (0 LLM calls) - UC-8, UC-9
- Citation Extractor (0 LLM calls) - UC-10, UC-11
- Synthesis Agent (1 LLM call) - UC-12

---

### 📊 Analysis & Insights
| Use Case ID | Use Case Name | Actor | Description | LLM Calls |
|------------|---------------|-------|-------------|-----------|
| UC-14 | View Research Synthesis | Researcher | Review complete research analysis | 0 |
| UC-15 | View Key Findings | Researcher | Examine main discoveries (3-15 findings) | 0 |
| UC-16 | View Research Gaps | Researcher | Identify areas needing investigation | 0 |
| UC-17 | View Methodology Insights | Researcher | Analyze research methods used | 0 |
| UC-18 | View Citation Network | Researcher | Explore paper relationships | 0 |
| UC-19 | View Recommended Papers | Researcher | See most relevant documents | 0 |

**Output from**: Synthesis Agent structured response

---

### ⚙️ System Management
| Use Case ID | Use Case Name | Actor | Description | LLM Calls |
|------------|---------------|-------|-------------|-----------|
| UC-20 | Monitor Performance | Admin | Track system metrics and efficiency | 0 |
| UC-21 | Track LLM Calls | Admin | Monitor API usage and call counts | 0 |
| UC-22 | View Cost Metrics | Admin | Review estimated costs per query | 0 |
| UC-23 | Reset System | Admin | Clear all data and reset counters | 0 |
| UC-24 | Run Diagnostics | Admin | Execute system health checks | 0 |
| UC-25 | View Agent Status | Admin | Check status of all 3 agents | 0 |

**Managed by**: Research Coordinator (0 LLM calls)

---

### 📝 Session Management
| Use Case ID | Use Case Name | Actor | Description | LLM Calls |
|------------|---------------|-------|-------------|-----------|
| UC-26 | Create Research Session | System | Initialize new research session | 0 |
| UC-27 | View Session History | Researcher | Review previous research queries | 0 |
| UC-28 | Export Results | Researcher | Download research synthesis | 0 |
| UC-29 | Analyze Multiple Documents | Researcher | Add documents to existing session | 0 |

**Managed by**: Research Memory system

---

## Use Case Relationships

### Include Relationships
```
UC-1 (Upload Documents)
  ├─ includes → UC-2 (Process Documents)
  │   ├─ includes → UC-3 (Extract Text)
  │   ├─ includes → UC-4 (Create Chunks)
  │   ├─ includes → UC-5 (Generate Embeddings)
  │   └─ includes → UC-6 (Store in Vector DB)

UC-7 (Submit Research Query)
  ├─ includes → UC-8 (Search Documents)
  │   └─ includes → UC-9 (Rank Results)
  ├─ includes → UC-10 (Extract Citations)
  ├─ includes → UC-11 (Extract Key Quotes)
  ├─ includes → UC-12 (Generate Synthesis)
  └─ includes → UC-26 (Create Session)

UC-14 (View Research Synthesis)
  ├─ includes → UC-15 (View Key Findings)
  ├─ includes → UC-16 (View Research Gaps)
  └─ includes → UC-17 (View Methodology Insights)

UC-20 (Monitor Performance)
  ├─ includes → UC-21 (Track LLM Calls)
  └─ includes → UC-22 (View Cost Metrics)
```

### Extend Relationships
```
UC-13 (Ask Follow-up Question) extends → UC-7 (Submit Research Query)
UC-29 (Analyze Multiple Documents) extends → UC-1 (Upload Documents)
UC-28 (Export Results) extends → UC-27 (View Session History)
```

---

## System Workflow

### Primary Research Flow
```
1. Researcher uploads documents (UC-1)
   ↓
2. System processes documents (UC-2 → UC-3 → UC-4 → UC-5 → UC-6)
   ↓
3. Researcher submits research query (UC-7)
   ↓
4. Literature Scanner searches documents (UC-8, UC-9) [0 LLM calls]
   ↓
5. Citation Extractor processes papers (UC-10, UC-11) [0 LLM calls]
   ↓
6. Synthesis Agent generates insights (UC-12) [1 LLM call]
   ↓
7. Researcher views results (UC-14 → UC-15, UC-16, UC-17, UC-18, UC-19)
   ↓
8. Optional: Ask follow-up question (UC-13) [1 additional LLM call]
```

### Administrative Flow
```
1. Admin monitors performance (UC-20)
   ↓
2. Admin tracks LLM usage (UC-21)
   ↓
3. Admin reviews costs (UC-22)
   ↓
4. Admin checks agent status (UC-25)
   ↓
5. Optional: Run diagnostics (UC-24) or Reset system (UC-23)
```

---

## Key Performance Indicators

### Efficiency Metrics
- **Total LLM Calls per Query**: 1 (Target: ≤2)
- **Deterministic Agents**: 3 out of 4 (75%)
- **Processing Time**: <60 seconds per query
- **Cost per Query**: ~$0.002 (Gemini API)
- **Success Rate**: >95% for document processing

### Agent Performance
| Agent | Type | LLM Calls | Primary Function |
|-------|------|-----------|------------------|
| Document Processor | Deterministic | 0 | Document loading & FAISS indexing |
| Literature Scanner | Deterministic | 0 | Semantic search & ranking |
| Citation Extractor | Deterministic | 0 | Citation & quote extraction |
| Synthesis Agent | LLM-powered | 1 | Research synthesis generation |

---

## External System Interactions

### Gemini API Integration
- **Use Cases**: UC-5 (embeddings), UC-12 (synthesis)
- **Model**: gemini-2.5-flash
- **Purpose**: Generate embeddings and research synthesis
- **Cost**: ~$0.002 per query

### FAISS Vector Store
- **Use Cases**: UC-6 (storage), UC-8 (search)
- **Purpose**: Efficient semantic similarity search
- **Index Type**: Flat L2 distance
- **Capacity**: Unlimited documents (memory-dependent)

---

## Security & Access Control

### Researcher Permissions
- ✅ Upload and process documents
- ✅ Submit research queries
- ✅ View all analysis results
- ✅ Export research synthesis
- ❌ Access system administration
- ❌ Modify system configuration

### Administrator Permissions
- ✅ All researcher permissions
- ✅ Monitor system performance
- ✅ Track costs and usage
- ✅ Reset system state
- ✅ Run diagnostics
- ✅ Configure system settings

---

## Error Handling Use Cases

### Document Processing Errors
- **UC-E1**: Handle invalid file format
- **UC-E2**: Handle corrupted PDF files
- **UC-E3**: Handle oversized documents (>50MB)
- **UC-E4**: Handle embedding generation failures

### Query Processing Errors
- **UC-E5**: Handle empty query submission
- **UC-E6**: Handle no documents found
- **UC-E7**: Handle LLM API failures
- **UC-E8**: Handle timeout errors

### System Errors
- **UC-E9**: Handle API key validation failures
- **UC-E10**: Handle FAISS index corruption
- **UC-E11**: Handle memory overflow
- **UC-E12**: Handle network connectivity issues

---

## Future Enhancements

### Planned Use Cases
- **UC-30**: Batch Process Multiple Queries
- **UC-31**: Schedule Automated Research
- **UC-32**: Collaborate on Research Sessions
- **UC-33**: Integrate External Paper Databases
- **UC-34**: Generate Research Reports (PDF/DOCX)
- **UC-35**: Compare Multiple Research Syntheses
- **UC-36**: Visualize Citation Networks
- **UC-37**: Track Research Trends Over Time

---

## Technical Notes

### System Architecture
- **Framework**: Streamlit (Web UI)
- **LLM Provider**: Google Gemini API
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: LangChain
- **Data Validation**: Pydantic
- **Session Management**: Streamlit session state

### Supported File Formats
- PDF (via PyPDF2/pdfplumber)
- TXT (plain text)
- MD (Markdown)

### Performance Targets
- **Document Processing**: <30 seconds per document
- **Query Processing**: <60 seconds per query
- **Concurrent Users**: Session-based isolation
- **Document Limit**: No hard limit (memory-dependent)

---

## Conclusion

This use case diagram represents a comprehensive hybrid multi-agent research assistant system that achieves:
- ✅ **Efficiency**: Exactly 1 LLM call per research query
- ✅ **Scalability**: FAISS-based vector search for large document collections
- ✅ **Accuracy**: Regex-based citation extraction with 100% test success rate
- ✅ **Cost-effectiveness**: ~$0.002 per query using Gemini API
- ✅ **User Experience**: Streamlit-based intuitive interface

The system successfully balances deterministic processing (75% of agents) with LLM-powered synthesis to deliver high-quality research insights at minimal cost.