"""

Streamlit UI for Research Assistant System



Research Assistant System: Upload documents and ask research questions

Agents: Document Processor (0 LLM), Literature Scanner (0 LLM), Synthesis Agent (1 LLM)

"""



import streamlit as st

import pandas as pd

import json

import time

import os

from datetime import datetime

from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

from pathlib import Path



# Import system components

from core.coordinator import ResearchCoordinator

from config.settings import SystemConfig



def setup_page_config():

    """Configure Streamlit page settings"""

    st.set_page_config(

        page_title="🔬 Research Assistant AI",

        page_icon="🔬", 

        layout="wide",

        initial_sidebar_state="expanded"

    )



def load_custom_css():

    """Load custom CSS for modern, professional styling"""

    st.markdown("""

    <style>

    .main-header {

        font-size: 2.5rem;

        font-weight: bold;

        text-align: center;

        margin-bottom: 1rem;

        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

    }

    

    .sub-header {

        text-align: center;

        color: #666;

        margin-bottom: 2rem;

    }

    

    .step-card {

        border: 2px solid #e0e0e0;

        border-radius: 15px;

        padding: 1.5rem;

        margin: 1rem 0;

        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);

        color: white;

        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

    }

    

    .metric-card {

        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);

        color: white;

        padding: 1.5rem;

        border-radius: 15px;

        margin: 0.5rem;

        text-align: center;

        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);

    }

    

    .agent-status {

        padding: 0.5rem;

        border-radius: 8px;

        margin: 0.25rem 0;

        font-weight: bold;

    }

    

    .agent-llm { background-color: #ffeb3b; color: #000; }

    .agent-deterministic { background-color: #4caf50; color: white; }

    

    #MainMenu {visibility: hidden;}

    footer {visibility: hidden;}

    header {visibility: hidden;}

    </style>

    """, unsafe_allow_html=True)



def initialize_session_state():

    """Initialize Streamlit session state"""

    

    if 'research_system' not in st.session_state:

        # Load environment variables from .env file

        load_dotenv()

        

        # Get API key from environment - Try Gemini first, then fallback to OpenAI

        api_key = os.getenv("GEMINI_API_KEY", None)

        if not api_key:

            api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

        

        # Check if API key is valid

        if not api_key or (api_key == "your-openai-api-key-here" and not os.getenv("GEMINI_API_KEY")):

            st.session_state.api_key_valid = False

            st.session_state.research_system = None

        else:

            st.session_state.api_key_valid = True

            st.session_state.research_system = ResearchCoordinator(api_key)

        

        # Initialize session state

        st.session_state.current_step = "upload"  # upload, process, question

        st.session_state.uploaded_files = []

        st.session_state.documents_processed = False

        st.session_state.research_questions = []

        st.session_state.current_question = ""

        st.session_state.analyze_additional = False  # Flag for analyzing another document

        

        # Document processing stats

        st.session_state.processing_stats = {

            'total_documents': 0,

            'total_chunks': 0,

            'llm_calls_made': 0

        }



    # Always ensure recommendation state exists (survives reruns)
    if 'recommended_questions' not in st.session_state:
        st.session_state.recommended_questions = []
    if 'selected_recommendation' not in st.session_state:
        st.session_state.selected_recommendation = ''
    if 'recs_generated' not in st.session_state:
        st.session_state.recs_generated = False


def render_header():

    """Render main application header"""

    st.markdown('<div class="main-header">🔬 Research Assistant AI</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header"><strong>CSYE 7374 Final Project</strong> | <em>Hybrid Multi-Agent Research System</em></div>', unsafe_allow_html=True)

    

    # Show current step

    steps = ["📄 Upload Documents", "🔧 Process Documents", "❓ Ask Research Questions"]

    current_step_idx = {"upload": 0, "process": 1, "question": 2}[st.session_state.current_step]

    

    cols = st.columns(3)

    for i, (col, step) in enumerate(zip(cols, steps)):

        if i <= current_step_idx:

            col.success(step)

        else:

            col.info(step)

    

    st.divider()



def render_sidebar():

    """Render sidebar with system information and controls"""

    

    with st.sidebar:

        st.header("🏗️ System Architecture")

        

        # System status

        if st.session_state.api_key_valid:

            st.success("✅ System Ready (Gemini API)")

        else:

            st.error("❌ API Key Required")

            st.markdown("""

            **Setup Instructions:**

            1. Get Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

            2. Create `.env` file with: `GEMINI_API_KEY=your-key-here`

            3. Restart the app

            """)

            return

        

        # Agent status display

        st.markdown("### 🤖 3-Agent System Status")

        

        # Document Processor (0 LLM calls)

        st.markdown("""

        <div class="agent-status agent-deterministic">

            📄 Document Processor: Deterministic (0 LLM calls)

        </div>

        """, unsafe_allow_html=True)

        

        # Literature Scanner (0 LLM calls)

        st.markdown("""

        <div class="agent-status agent-deterministic">

            🔍 Literature Scanner: Deterministic (0 LLM calls)

        </div>

        """, unsafe_allow_html=True)

        

        # Citation Extractor (0 LLM calls)

        st.markdown("""

        <div class="agent-status agent-deterministic">

            📑 Citation Extractor: Deterministic (0 LLM calls)

        </div>

        """, unsafe_allow_html=True)

        

        # Synthesis Agent (1 LLM call)

        st.markdown("""

        <div class="agent-status agent-llm">

            🤖 Synthesis Agent: LLM-based (1 LLM call)

        </div>

        """, unsafe_allow_html=True)

        

        st.info("🎯 **Total: 1 LLM call per research question**")

        

        # Performance metrics

        if st.session_state.research_system:

            stats = st.session_state.research_system.get_system_stats()

            

            st.markdown("### 📊 Performance Metrics")

            

            col1, col2 = st.columns(2)

            with col1:

                st.metric("Total LLM Calls", stats["total_llm_calls"])

                st.metric("Sessions", stats["total_research_sessions"])

            

            with col2:

                st.metric("Total Cost", f"${stats['total_cost']:.4f}")

                st.metric("Efficiency", stats["efficiency_score"])

            

            # LLM call efficiency

            if stats["total_research_sessions"] > 0:

                avg_calls = stats["average_llm_calls_per_session"]

                st.markdown("### 🎯 LLM Call Efficiency")

                

                if avg_calls <= 1.5:

                    st.success(f"🏆 Excellent: {avg_calls:.1f} calls/query")

                elif avg_calls <= 2.0:

                    st.success(f"✅ Target Met: {avg_calls:.1f} calls/query")

                else:

                    st.error(f"❌ Target Missed: {avg_calls:.1f} calls/query")

        

        # Document Management

        st.markdown("### 📚 Document Status")

        

        if st.session_state.documents_processed and st.session_state.research_system:

            processor = st.session_state.research_system.document_processor

            if processor:

                stats = processor.get_processing_stats()

                

                if stats['vector_store_initialized']:

                    st.success("✅ FAISS Vector Store Active")

                    st.info(f"📄 {stats['total_documents']} documents")

                    st.info(f"🔧 {stats['total_chunks']} chunks")

                    st.info(f"🤖 {stats['llm_calls_made']} LLM calls")

                else:

                    st.info("📚 No documents processed yet")

        

        # System controls

        st.markdown("### ⚙️ System Controls")

        

        if st.button("🔄 Reset System", help="Clear all data and reset system"):

            if st.session_state.research_system:

                st.session_state.research_system.reset_system()

                st.session_state.current_step = "upload"

                st.session_state.uploaded_files = []

                st.session_state.documents_processed = False
                st.session_state.recommended_questions = []
                st.session_state.recs_generated = False
                st.session_state.selected_recommendation = ''

                st.session_state.research_questions = []

                st.session_state.processing_stats = {

                    'total_documents': 0,

                    'total_chunks': 0,

                    'llm_calls_made': 0

                }

                st.success("System reset complete!")

                st.rerun()

        

def render_upload_section():

    """Step 1: Document Upload"""

    

    # Check if adding another document

    is_additional = st.session_state.get('analyze_additional', False)

    

    if is_additional:

        st.header("📄 Add Another Research Document")

        st.markdown("Upload an additional PDF, TXT, or MD file to analyze alongside your existing documents.")

        st.info("ℹ️ This document will be added to your knowledge base for comprehensive analysis.")

    else:

        st.header("📄 Step 1: Upload Research Documents")

        st.markdown("Upload PDF, TXT, or MD files to build your research knowledge base.")

    

    # File uploader

    uploaded_files = st.file_uploader(

        "Choose research documents",

        type=['pdf', 'txt', 'md'],

        accept_multiple_files=True,

        help="Supported formats: PDF, TXT, MD. Documents will be processed and added to FAISS vector store."

    )

    

    if uploaded_files:

        st.session_state.uploaded_files = uploaded_files

        st.success(f"📚 {len(uploaded_files)} file(s) selected")

        

        # Show file details

        st.markdown("**Selected Files:**")

        for file in uploaded_files:

            st.info(f"📄 {file.name} ({file.size / 1024:.1f} KB)")

        

        # Proceed to processing

        if st.button("🚀 Proceed to Document Processing", type="primary", use_container_width=True):

            st.session_state.current_step = "process"

            st.rerun()

    

    # Show example documents

    with st.expander("📖 Example Research Documents"):

            st.markdown("""

        **For testing, you can use:**

        - Research papers (PDF)

        - Technical documentation (TXT)

        - Academic articles (PDF)

        - Literature reviews (MD)

        

        **Note:** The system will automatically chunk documents and create embeddings for semantic search.

        """)



def render_processing_section():

    """Step 2: Document Processing"""

    

    st.header("🔧 Step 2: Process Documents")

    st.markdown("Documents are being processed and added to the FAISS vector store for semantic search.")

    

    if not st.session_state.uploaded_files:

        st.error("❌ No files uploaded. Please go back to Step 1.")

        if st.button("⬅️ Back to Upload"):

            st.session_state.current_step = "upload"

            st.rerun()

        return

    

    # Show files to be processed

    st.markdown("**Files to Process:**")

    for file in st.session_state.uploaded_files:

        st.info(f"📄 {file.name} ({file.size / 1024:.1f} KB)")

    

    # Process documents button

    if not st.session_state.documents_processed:

        if st.button("🔧 Process Documents", type="primary", use_container_width=True):

            process_documents()

    

    # Show processing status

    if st.session_state.documents_processed:

        st.success("✅ Documents processed successfully!")

        

        # Show processing stats

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric("Documents", st.session_state.processing_stats['total_documents'])

        with col2:

            st.metric("Chunks", st.session_state.processing_stats['total_chunks'])

        with col3:

            st.metric("LLM Calls", st.session_state.processing_stats['llm_calls_made'])

        

        # Proceed to questions

        if st.button("❓ Proceed to Research Questions", type="primary", use_container_width=True):

            st.session_state.current_step = "question"

            st.rerun()

    

    # Back button

    if st.button("⬅️ Back to Upload"):

        st.session_state.current_step = "upload"

        st.rerun()



def process_documents():

    """Process uploaded documents using the coordinator's document processor"""

    

    if not st.session_state.research_system:

        st.error("❌ Research system not initialized")

        return

    

    # Process each file

    with st.spinner("🔧 Processing documents and building FAISS vector store..."):

        total_docs = 0

        total_chunks = 0

        total_llm_calls = 0

        

        for uploaded_file in st.session_state.uploaded_files:

            st.info(f"📄 Processing {uploaded_file.name}...")

            

            # Process the uploaded file using the document processor

            result = st.session_state.research_system.document_processor.process_uploaded_file(uploaded_file)

            

            if result['success']:

                st.success(f"✅ {uploaded_file.name} processed successfully!")

                st.info(f"📊 Created {result.get('chunks_created', 0)} chunks")

                total_docs += 1

                total_chunks += result.get('chunks_created', 0)

                total_llm_calls += result.get('llm_calls_made', 0)

            else:

                st.error(f"❌ Failed to process {uploaded_file.name}: {result.get('error', 'Unknown error')}")

        

        # Update session state

        st.session_state.documents_processed = True

        st.session_state.processing_stats = {

            'total_documents': total_docs,

            'total_chunks': total_chunks,

            'llm_calls_made': total_llm_calls

        }

        

        st.success(f"🎉 Document processing complete! Processed {total_docs} documents.")

        

        # Show final stats from document processor

        if st.session_state.research_system.document_processor:

            final_stats = st.session_state.research_system.document_processor.get_processing_stats()

            st.info(f"📈 Final Stats: {final_stats['total_documents']} docs, {final_stats['total_chunks']} chunks, {final_stats['llm_calls_made']} LLM calls")





def generate_recommended_questions(coordinator) -> list:
    """Generate 6 questions from paper content via Gemini. No FAISS needed."""
    import json as _j, re as _re

    try:
        dp       = coordinator.document_processor
        raw_docs = getattr(dp, 'documents', [])
        if not raw_docs:
            return []

        step     = max(1, len(raw_docs) // 8)
        selected = raw_docs[::step][:8]
        parts, total = [], 0
        for doc in selected:
            t = getattr(doc, 'page_content', '').strip()
            if t and total < 2500:
                parts.append(t[:400])
                total += len(t[:400])

        paper_text = '\n\n'.join(parts)
        if len(paper_text) < 50:
            return []

        messages = [
            {
                "role": "system",
                "content": "Respond ONLY with a valid JSON array of exactly 6 question strings. No markdown, no explanation."
            },
            {
                "role": "user",
                "content": (
                    "Generate exactly 6 specific research questions from this paper content.\n"
                    "Return ONLY a JSON array like: [\"Q1?\",\"Q2?\",\"Q3?\",\"Q4?\",\"Q5?\",\"Q6?\"]\n\n"
                    f"Paper:\n{paper_text}"
                )
            }
        ]

        resp = coordinator.llm.make_call(messages)
        if not resp:
            return []

        raw = resp.content.strip()
        # strip fences
        raw = _re.sub(r'```[a-z]*', '', raw).strip().strip('`').strip()
        # find array
        m = _re.search(r'\[[\s\S]*?\]', raw)
        if m:
            qs = _j.loads(m.group(0))
            if isinstance(qs, list):
                return [str(q).strip() for q in qs if str(q).strip()][:6]
        # last resort
        qs = _j.loads(raw)
        if isinstance(qs, list):
            return [str(q).strip() for q in qs if str(q).strip()][:6]
        return []

    except Exception:
        return []


def render_question_section():

    """Step 3: Ask Research Questions"""

    

    st.header("❓ Step 3: Ask Research Questions")

    st.markdown("Ask questions about your uploaded documents. The system will use semantic search and synthesis to provide answers.")

    

    if not st.session_state.documents_processed:

        st.error("❌ Documents not processed yet. Please complete Step 2 first.")

        if st.button("⬅️ Back to Processing"):

            st.session_state.current_step = "process"

            st.rerun()

        return

    

    # ── Smart Question Recommendations ─────────────────────────────────────
    if not st.session_state.get('recs_generated', False):
        if st.session_state.get('research_system'):
            with st.spinner('Generating question suggestions from your paper...'):
                qs = generate_recommended_questions(st.session_state.research_system)
                st.session_state.recommended_questions = qs
                st.session_state.recs_generated = True
                if qs:
                    st.rerun()

    recs = st.session_state.get('recommended_questions', [])
    if recs:
        st.markdown('#### 💡 Suggested Questions — click to use')
        cols = st.columns(2)
        for idx, q in enumerate(recs):
            with cols[idx % 2]:
                if st.button(q, key=f'rec_{idx}', use_container_width=True):
                    st.session_state.selected_recommendation = q
                    st.rerun()
        st.divider()


    # Pre-fill textarea from clicked recommendation
    prefill = st.session_state.get('selected_recommendation', '')

        # Research question form

    with st.form("research_question_form"):

        st.markdown("**Ask a research question about your documents:**")

        research_question = st.text_area(
            "Research Question:",
            value=prefill,
            placeholder="e.g., What are the main findings about transformer attention mechanisms?",
            height=120,
            label_visibility="collapsed"
        )

            

        col1, col2 = st.columns([3, 1])

        with col1:

            max_results = st.slider("Max Results:", 3, 10, 5)

        

        with col2:

            st.markdown("**Expected LLM Calls:**")

            st.info("1 call")

            

            submitted = st.form_submit_button(

            "🔍 Ask Question", 

                type="primary",

                use_container_width=True

            )

        

    # Clear prefill and process after submit
    if submitted:
        st.session_state.selected_recommendation = ''
    if submitted and research_question:
        process_research_question(research_question, max_results)

    

    # Follow-up question section (appears after first question is answered)

    if st.session_state.research_questions:

        st.markdown("---")

        st.markdown("### 🔄 Follow-up Question")

        st.markdown("Ask a follow-up question based on the previous analysis:")

        

        with st.form("followup_question_form"):

            followup_question = st.text_area(

                "Follow-up Question:",

                placeholder="e.g., How do these attention mechanisms compare in terms of computational complexity?",

                height=100,

                label_visibility="collapsed"

            )

            

            # Create a clean layout for controls

            col1, col2, col3 = st.columns([2, 1, 1])

            

            with col1:

                followup_max_results = st.slider("Max Results:", 3, 10, 5, key="followup_slider")

            

            with col2:

                st.markdown("**Expected LLM Calls:**")

                st.info("1 call")

            

            with col3:

                st.markdown("&nbsp;")  # Add spacing

                followup_submitted = st.form_submit_button(

                    "🔍 Ask Follow-up", 

                type="primary",

                use_container_width=True

            )

        

        # Process follow-up question

        if followup_submitted and followup_question:

            process_research_question(followup_question, followup_max_results)

    

    # Show previous questions

    if st.session_state.research_questions:

        st.markdown("### 📚 Previous Questions")

        for i, qa in enumerate(st.session_state.research_questions, 1):

            with st.expander(f"Q{i}: {qa['question'][:50]}...", expanded=False):

                st.markdown(f"**Question:** {qa['question']}")

                

                # Show full research synthesis when expanded

                st.markdown("**Full Research Synthesis:**")

                st.markdown(qa['answer'])  # Show complete answer without truncation

                

                # Show performance metrics

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric("LLM Calls", qa['llm_calls'])

                with col2:

                    st.metric("Processing Time", f"{qa['processing_time']:.2f}s")

                with col3:

                    if 'papers_found' in qa:

                        st.metric("Documents Found", qa['papers_found'])

                with col4:

                    if 'quotes_extracted' in qa:

                        st.metric("Quotes Extracted", qa['quotes_extracted'])

                

                # Show additional details if available

                if 'synthesis' in qa and qa['synthesis']:

                    synthesis = qa['synthesis']

                    

                    # Show confidence score

                    st.markdown(f"**Confidence Score:** {synthesis.get('confidence', 0):.2f}/1.0")

                    

                    # Show research gaps if available

                    if 'research_gaps' in synthesis and synthesis['research_gaps']:

                        st.markdown("**🎯 Research Gaps Identified:**")

                        for gap in synthesis['research_gaps']:

                            st.markdown(f"• {gap}")

                    

                    # Show methodology insights if available

                    if 'methodology_insights' in synthesis and synthesis['methodology_insights']:

                        st.markdown("**🔬 Methodology Insights:**")

                        for insight in synthesis['methodology_insights']:

                            st.markdown(f"• {insight}")

                

                st.divider()

    

    # Navigation buttons

    col1, col2, col3 = st.columns(3)

    

    with col1:

        if st.button("⬅️ Back to Processing"):

            st.session_state.current_step = "process"

            st.rerun()

    

    with col2:

        if st.button("➕ Analyze Another Document"):

            st.session_state.current_step = "upload"

            st.session_state.analyze_additional = True
            st.session_state.recommended_questions = []
            st.session_state.recs_generated = False
            st.session_state.selected_recommendation = ''

            st.rerun()

    

    with col3:

        if st.button("🔄 New Analysis"):

            st.session_state.current_step = "upload"

            st.session_state.uploaded_files = []

            st.session_state.documents_processed = False
            st.session_state.recommended_questions = []
            st.session_state.recs_generated = False
            st.session_state.selected_recommendation = ''

            st.session_state.research_questions = []

            st.rerun()



def process_research_question(question: str, max_results: int):

    """Process research question using the full 3-agent pipeline"""

    

    if not st.session_state.research_system:

        st.error("❌ Research system not initialized")

        return

    

    if not st.session_state.research_system.document_processor:

        st.error("❌ Document processor not available")

        return

    

    # Track start time

    start_time = time.time()

    

    with st.spinner("🔍 Running full research pipeline with 3 agents..."):

        try:

            # Use the Coordinator's proper research pipeline that orchestrates all 3 agents

            st.info("🤖 Starting 3-Agent Research Pipeline...")

            st.info("📚 → Literature Scanner (0 LLM calls): Finding relevant documents")

            st.info("📑 → Citation Extractor (0 LLM calls): Extracting citations and quotes")

            st.info("🤖 → Synthesis Agent (1 LLM call): Creating research synthesis")

            

            # Call the coordinator's research pipeline

            result = st.session_state.research_system.research_query(

                query=question,

                domain="other",

                max_papers=max_results

            )

            

            if not result['success']:

                st.error(f"❌ Research pipeline failed: {result.get('error', 'Unknown error')}")

                return

            

            # Extract comprehensive results from the pipeline

            papers_found = result['papers_found']['count']

            citations_extracted = result['extracted_insights']['total_citations']

            quotes_extracted = result['extracted_insights']['total_quotes']

            synthesis = result['research_synthesis']

            performance = result['performance_metrics']

            

            st.success(f"✅ Research Pipeline Complete!")

            st.success(f"📚 Found {papers_found} relevant documents")

            st.success(f"📑 Extracted {citations_extracted} citations and {quotes_extracted} quotes")

            st.success(f"🤖 Generated synthesis with {len(synthesis['key_findings'])} key findings")

            

            # Create answer from synthesis

            answer = "## Research Synthesis\n\n"

            

            # Key findings

            answer += "### 🔍 Key Findings:\n"

            for i, finding in enumerate(synthesis['key_findings'], 1):

                answer += f"{i}. {finding}\n"

            answer += "\n"

            

            # Methodology insights

            if synthesis['methodology_insights']:

                answer += "### 🔬 Methodology Insights:\n"

                for insight in synthesis['methodology_insights']:

                    answer += f"• {insight}\n"

                answer += "\n"

            

            # Research gaps

            if synthesis['research_gaps']:

                answer += "### 🎯 Research Gaps Identified:\n"

                for gap in synthesis['research_gaps']:

                    answer += f"• {gap}\n"

                answer += "\n"

            

            # Recommended papers

            if synthesis['recommended_papers']:

                answer += "### 📖 Recommended Documents:\n"

                for paper in synthesis['recommended_papers']:

                    answer += f"• {paper}\n"

                answer += "\n"

            

            answer += f"**Confidence Score:** {synthesis['confidence']:.2f}/1.0\n"

            answer += f"**Analysis Quality:** {synthesis['completeness']['quality_rating']}\n"

            

            # Calculate processing time

            processing_time = time.time() - start_time

            

            # Store Q&A with full pipeline results

            qa_pair = {

                'question': question,

                'answer': answer,

                'llm_calls': performance['total_llm_calls'],

                'processing_time': processing_time,

                'papers_found': papers_found,

                'quotes_extracted': quotes_extracted,

                'citations_extracted': citations_extracted,

                'research_result': result,  # Store the full result

                'synthesis': synthesis,

                'performance': performance

            }

            st.session_state.research_questions.append(qa_pair)

            

            # Display comprehensive results using the proper pipeline data

            display_research_pipeline_results(qa_pair, result)

            

        except Exception as e:

            st.error(f"❌ Error processing question: {str(e)}")

            st.error("Please check if documents were processed correctly.")

            import traceback

            st.error(f"Technical details: {traceback.format_exc()}")



def display_document_search_results(qa_pair: Dict[str, Any], search_results: List[Dict[str, Any]]):

    """Display document search results"""

    

    st.success("✅ Question Answered Using Document Search!")

    

    # Performance metrics

    col1, col2, col3, col4 = st.columns(4)

    

    with col1:

        st.markdown(f"""

        <div class="metric-card">

            <h3>🤖 LLM Calls</h3>

            <h2>{qa_pair['llm_calls']}</h2>

        </div>

        """, unsafe_allow_html=True)

    

    with col2:

        st.markdown(f"""

        <div class="metric-card">

            <h3>📚 Results</h3>

            <h2>{len(search_results)}</h2>

        </div>

        """, unsafe_allow_html=True)

    

    with col3:

        st.markdown(f"""

        <div class="metric-card">

            <h3>⏱️ Time</h3>

            <h2>{qa_pair['processing_time']:.2f}s</h2>

        </div>

        """, unsafe_allow_html=True)

    

    with col4:

        st.markdown(f"""

        <div class="metric-card">

            <h3>⭐ Efficiency</h3>

            <h2>✅ Target Met</h2>

        </div>

        """, unsafe_allow_html=True)

    

    st.divider()

    

    # Display the answer

    st.subheader("💡 Answer")

    st.markdown(qa_pair['answer'])

    

    # Show search results

    st.subheader("🔍 Search Results")

    st.info(f"Found {len(search_results)} relevant document chunks")

    

    for i, result in enumerate(search_results, 1):

        with st.expander(f"📄 Result {i}: Similarity Score {result['similarity_score']:.3f}", expanded=False):

            st.markdown(f"**Source:** {result['source']}")

            st.markdown(f"**Content:**")

            # Show full content without truncation

            st.text(result['content'])

            

            # Show metadata if available

            if result['metadata']:

                st.markdown("**Metadata:**")

                for key, value in result['metadata'].items():

                    if key != 'source':

                        st.write(f"• {key}: {value}")

    

    # Agent performance breakdown

    st.subheader("🤖 Agent Performance")

    

    col1, col2, col3 = st.columns(3)

    

    with col1:

            st.markdown("""

        <div class="step-card">

            <h4>📄 Document Processor</h4>

                <p><strong>LLM Calls:</strong> 0</p>

            <p><strong>Function:</strong> Document chunking & vector storage</p>

            <p><strong>Status:</strong> ✅ Complete</p>

            </div>

        """, unsafe_allow_html=True)

    

    with col2:

        st.markdown("""

        <div class="step-card">

            <h4>🔍 FAISS Search</h4>

                <p><strong>LLM Calls:</strong> 0</p>

            <p><strong>Function:</strong> Semantic similarity search</p>

            <p><strong>Status:</strong> ✅ Complete</p>

            </div>

        """, unsafe_allow_html=True)

    

    with col3:

            st.markdown("""

        <div class="step-card">

            <h4>🤖 Synthesis</h4>

                <p><strong>LLM Calls:</strong> 1</p>

            <p><strong>Function:</strong> Answer generation</p>

            <p><strong>Status:</strong> ✅ Complete</p>

            </div>

        """, unsafe_allow_html=True)

    

def display_research_pipeline_results(qa_pair: Dict[str, Any], research_result: Dict[str, Any]):

    """Display comprehensive results from the 3-agent pipeline"""

    

    st.success("✅ Research Pipeline Complete! All 3 agents executed successfully.")

    

    performance = research_result['performance_metrics']

    extracted_insights = research_result['extracted_insights']

    synthesis = research_result['research_synthesis']

    

    # Performance metrics

    col1, col2, col3, col4 = st.columns(4)

    

    with col1:

        st.markdown(f"""

        <div class="metric-card">

            <h3>🤖 LLM Calls</h3>

            <h2>{performance['total_llm_calls']}</h2>

        </div>

        """, unsafe_allow_html=True)

    

    with col2:

        st.markdown(f"""

        <div class="metric-card">

            <h3>📚 Documents</h3>

            <h2>{performance['papers_analyzed']}</h2>

            </div>

        """, unsafe_allow_html=True)

    

    with col3:

        st.markdown(f"""

        <div class="metric-card">

            <h3>💬 Quotes</h3>

            <h2>{extracted_insights['total_quotes']}</h2>

        </div>

        """, unsafe_allow_html=True)

    

    with col4:

        st.markdown(f"""

        <div class="metric-card">

            <h3>⏱️ Time</h3>

            <h2>{performance['processing_time']:.2f}s</h2>

        </div>

        """, unsafe_allow_html=True)

    

    st.divider()

    

    # Display the answer - Full response

    st.subheader("💡 Research Synthesis")

    

    # Show answer with full content in expandable section

    with st.expander("📖 Full Research Answer", expanded=True):

        st.markdown(qa_pair['answer'])

    

    # Display answer in text area for copying

    st.text_area(

        "Research Answer (copyable):",

        value=qa_pair['answer'],

        height=200,

        disabled=True,

        key="research_answer"

    )

    

    # Show detailed research insights

    st.subheader("🔍 Detailed Research Insights")

    

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["?? Literature Discovery", "?? Citation Analysis", "?? Key Quotes", "?? Research Gaps", "?? Limitations", "?? Performance Metrics"])

    

    with tab1:

        st.markdown("**Literature Discovery Results:**")

        papers_count = research_result.get('papers_found', {}).get('count', 0)

        domain = research_result.get('domain', 'N/A')

        year_span = extracted_insights.get('year_span', 'N/A')

        venues = extracted_insights.get('venues', [])

        

        st.write(f"• **Documents Found:** {papers_count}")

        st.write(f"• **Domain:** {domain}")

        st.write(f"• **Year Span:** {year_span}")

        st.write(f"• **Venues:** {', '.join(venues[:3]) if venues else 'N/A'}")

        

        papers_list = research_result.get('papers_found', {}).get('papers', [])

        if papers_list:

            st.markdown("**Documents Analyzed:**")

            for i, paper in enumerate(papers_list[:5], 1):

                display_title = paper.get('title', 'Unknown')

                if display_title.startswith("Document: "):

                    display_title = display_title[10:]

                

                year = paper.get('year', 'N/A')

                venue = paper.get('venue', 'N/A')

                relevance = paper.get('relevance_score', 0)

                

                st.write(f"{i}. **{display_title}** ({year})")

                st.write(f"   Source: {venue}")

                st.write(f"   Relevance Score: {relevance:.3f}")

                

                metadata = paper.get('metadata', {})

                if 'page' in metadata:

                    st.write(f"   Page: {metadata['page']}")

                if 'heading' in metadata:

                    st.write(f"   Section: {metadata['heading']}")

                if 'chunk_count' in metadata:

                    st.write(f"   Chunks: {metadata['chunk_count']}")

                st.write("")

    

    with tab2:

        st.markdown("**Citation Analysis:**")

        total_citations = extracted_insights.get('total_citations', 0)

        top_authors = extracted_insights.get('top_authors', [])

        citation_network = extracted_insights.get('citation_network', {})

        

        st.write(f"• **Total Citations:** {total_citations}")

        

        if top_authors:

            authors_str = ', '.join([author for author, _ in top_authors[:5]])

            st.write(f"• **Top Authors:** {authors_str}")

        else:

            st.write("• **Top Authors:** No citation data available")

        

        if citation_network:

            connections = citation_network.get('total_connections', 0)

            unique_authors = citation_network.get('unique_authors', 0)

            st.write(f"• **Citation Network:** {connections} connections")

            st.write(f"• **Unique Authors:** {unique_authors}")

        else:

            st.info("No citation network data available for uploaded documents")

    

    with tab3:

        st.markdown("**Key Quotes Extracted:**")

        key_quotes = extracted_insights.get('key_quotes', [])

        

        if key_quotes:

            for i, quote in enumerate(key_quotes[:10], 1):

                if isinstance(quote, dict):

                    quote_text = quote.get('text', '')

                    confidence = quote.get('confidence', 0)

                    page = quote.get('page', 'N/A')

                    heading = quote.get('heading', 'N/A')

                else:

                    quote_text = str(quote)

                    confidence = 0

                    page = 'N/A'

                    heading = 'N/A'

                

                st.markdown(f"**Quote {i}:** (Page {page}, Section: {heading})")

                st.write(f"\"{quote_text}\"")

                st.write(f"Confidence: {confidence:.2f}")

                st.divider()

        else:

            st.info("No key quotes extracted. This is normal for uploaded documents without citation patterns.")

    

    with tab4:

        research_gaps = synthesis.get('research_gaps', [])

        methodology_insights = synthesis.get('methodology_insights', [])

        

        st.markdown("**Research Gaps Identified:**")

        if research_gaps:

            for i, gap in enumerate(research_gaps, 1):

                st.write(f"{i}. {gap}")

        else:

            st.info("No research gaps identified in current analysis")

        

        st.markdown("**Methodology Insights:**")

        if methodology_insights:

            for i, insight in enumerate(methodology_insights, 1):

                st.write(f"{i}. {insight}")

        else:

            st.info("No methodology insights available")

    with tab5:

        limitations = synthesis.get('limitations', [])

        st.markdown("**Project Limitations:**")

        if limitations:

            for i, limitation in enumerate(limitations, 1):

                st.write(f"{i}. {limitation}")

        else:

            st.info("No limitations explicitly mentioned in the paper")

    

    with tab6:

        performance_metrics = synthesis.get('performance_metrics', [])

        st.markdown("**Performance Metrics & Baselines:**")

        if performance_metrics:

            for i, metric in enumerate(performance_metrics, 1):

                st.write(f"{i}. {metric}")

            st.markdown("---")

            st.info("💡 Use these metrics as baselines for your project")

        else:

            st.info("No quantitative performance metrics found in the paper")



    

    # Agent performance breakdown

    st.subheader("🤖 3-Agent Pipeline Performance")

    

    col1, col2, col3 = st.columns(3)

    

    with col1:

        st.markdown(f"""

        <div class="step-card">

            <h4>🔍 Literature Scanner</h4>

            <p><strong>LLM Calls:</strong> 0</p>

            <p><strong>Function:</strong> Vector similarity search</p>

            <p><strong>Documents Found:</strong> {performance['papers_analyzed']}</p>

            <p><strong>Status:</strong> ✅ Complete</p>

        </div>

        """, unsafe_allow_html=True)

    

    with col2:

        st.markdown(f"""

        <div class="step-card">

            <h4>📑 Citation Extractor</h4>

            <p><strong>LLM Calls:</strong> 0</p>

            <p><strong>Function:</strong> Regex/parsing extraction</p>

            <p><strong>Citations:</strong> {extracted_insights['total_citations']}</p>

            <p><strong>Quotes:</strong> {extracted_insights['total_quotes']}</p>

            <p><strong>Status:</strong> ✅ Complete</p>

        </div>

        """, unsafe_allow_html=True)

    

    with col3:

        st.markdown(f"""

        <div class="step-card">

            <h4>🤖 Synthesis Agent</h4>

            <p><strong>LLM Calls:</strong> {performance['llm_agent_calls']}</p>

            <p><strong>Function:</strong> Research synthesis</p>

            <p><strong>Findings:</strong> {len(synthesis['key_findings'])}</p>

            <p><strong>Confidence:</strong> {synthesis['confidence']:.2f}</p>

            <p><strong>Status:</strong> ✅ Complete</p>

        </div>

        """, unsafe_allow_html=True)

    

    # System efficiency summary

    st.subheader("⚡ System Efficiency")

    

    col1, col2 = st.columns(2)

    

    with col1:

        efficiency_status = "✅ EXCELLENT" if performance['total_llm_calls'] <= 1 else "🟡 GOOD" if performance['total_llm_calls'] <= 2 else "🔴 NEEDS IMPROVEMENT"

        st.markdown(f"""

        **Efficiency Rating:** {efficiency_status}

        - **Target:** ≤2 LLM calls per query

        - **Actual:** {performance['total_llm_calls']} LLM calls

        - **Cost:** ${performance['estimated_cost']:.4f}

        - **Deterministic Agents:** {performance['deterministic_agent_calls']}/3

        """)

    

    with col2:

        st.markdown(f"""

        **Research Value Delivered:**

        - ✅ Literature Discovery: {performance['papers_analyzed']} documents

        - ✅ Citation Analysis: {extracted_insights['total_citations']} citations

        - ✅ Key Quote Extraction: {extracted_insights['total_quotes']} quotes

        - ✅ Research Synthesis: {len(synthesis['key_findings'])} findings

        - ✅ Research Gap Identification: {len(synthesis['research_gaps'])} gaps

        """)



def render_main_interface():

    """Render main interface based on current step"""

    

    if st.session_state.current_step == "upload":

        render_upload_section()

    elif st.session_state.current_step == "process":

        render_processing_section()

    elif st.session_state.current_step == "question":

        render_question_section()



def main():

    """Main Streamlit application entry point"""

    

    # Load environment variables

    load_dotenv()

    

    # Setup page

    setup_page_config()

    load_custom_css()

    initialize_session_state()

    

    # Render main interface

    render_header()

    render_sidebar()

    render_main_interface()



if __name__ == "__main__":

    main()







