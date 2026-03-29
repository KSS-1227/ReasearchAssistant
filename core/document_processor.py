"""
Document Processor with LangChain Integration

Uses LangChain for document loading and FAISS for vector storage
with Google embeddings for 100% Google ecosystem integration.
"""

import os
import tempfile
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# LangChain imports for document processing and vector storage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import our Google embeddings
from core.google_embeddings import GoogleEmbeddings

# ---------------------------------------------------------------------------
# Upload validation constants
# ---------------------------------------------------------------------------
_MAX_FILE_SIZE_MB  = 50
_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
# Block path traversal and shell-injection characters in filenames
_UNSAFE_FILENAME_RE = re.compile(r'[\\/:\*\?"<>\|\x00]')


class DocumentProcessor:
    """
    Document processor using LangChain components with Google embeddings.
    
    100% Google ecosystem integration:
    - Uses LangChain for document loading and vector storage
    - Uses Google embeddings instead of OpenAI
    - Manually tracks operations for transparency
    - No OpenAI dependency
    """
    
    def __init__(self, google_api_key: str):
        """
        Initialize document processor with Google API key.
        
        Args:
            google_api_key: Google API key for embeddings (same as Gemini)
        """
        self.google_api_key = google_api_key
        self.embeddings = GoogleEmbeddings(google_api_key)
        self.vector_store: Optional[FAISS] = None
        self.documents: List[Document] = []
        self.llm_call_count = 0  # Manual tracking as required
        
        # Document processing stats
        self.processing_stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_tokens_estimated': 0,
            'vector_store_size': 0
        }
    
    def process_document(self, file_path: str, file_type: str = None, original_filename: str = None) -> Dict[str, Any]:
        """
        Process a single document and add to vector store.
        
        Args:
            file_path: Path to the document file
            file_type: Type of document (pdf, txt, etc.)
            
        Returns:
            Processing result with metadata
        """
        try:
            # Determine file type if not provided
            if not file_type:
                file_type = Path(file_path).suffix.lower()
            
            # Load document using appropriate LangChain loader
            if file_type == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_type in ['.txt', '.md']:
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Load documents (no LLM calls here)
            raw_docs = loader.load()
            
            # Split documents into chunks (no LLM calls here)
            from config.settings import SystemConfig
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=SystemConfig.DOCUMENT_CONFIG["chunk_size"],
                chunk_overlap=SystemConfig.DOCUMENT_CONFIG["chunk_overlap"],
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
                
            chunks = text_splitter.split_documents(raw_docs)
            
            # Add source file information to each chunk's metadata
            # Use original filename if provided, otherwise use the file path name
            filename = original_filename if original_filename else Path(file_path).name
            import time
            processing_timestamp = time.time()
            
            for i, chunk in enumerate(chunks):
                # Extract heading from chunk content
                content_lines = chunk.page_content.split('\n')
                heading = "Content"  # Default
                
                # Look for section headings in first 10 lines
                for line in content_lines[:10]:
                    line_stripped = line.strip()
                    if line_stripped and len(line_stripped) < 100:
                        # Check if it looks like a heading
                        if (line_stripped.isupper() or 
                            any(keyword in line_stripped.lower() for keyword in 
                                ['abstract', 'introduction', 'method', 'result', 'conclusion', 
                                 'discussion', 'background', 'related work', 'experiment', 
                                 'evaluation', 'implementation', 'analysis', 'approach',
                                 'system', 'architecture', 'design', 'framework'])):
                            heading = line_stripped[:80]  # Limit length
                            break
                
                # Get actual page number from PDF metadata
                page_num = chunk.metadata.get('page', 0)
                
                # Enhance metadata
                chunk.metadata.update({
                    'source_file': filename,
                    'original_filename': filename,
                    'file_path': file_path,
                    'file_type': file_type,
                    'processing_timestamp': processing_timestamp,
                    'page': page_num,
                    'chunk_id': i,
                    'heading': heading,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
            
            # Update processing stats
            self.processing_stats['total_documents'] += 1
            self.processing_stats['total_chunks'] += len(chunks)
            self.processing_stats['total_tokens_estimated'] += sum(len(chunk.page_content.split()) for chunk in chunks)
            
            # Add to documents list
            self.documents.extend(chunks)
            
            # Create or update vector store
            if self.vector_store is None:
                # First document - create new vector store
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                self.llm_call_count += len(chunks)  # Google embeddings generation counts as LLM calls
            else:
                # Add to existing vector store
                self.vector_store.add_documents(chunks)
                self.llm_call_count += len(chunks)  # Google embeddings generation counts as LLM calls
            
            # Update vector store size
            self.processing_stats['vector_store_size'] = len(self.vector_store.index_to_docstore_id)
            
            return {
                'success': True,
                'document_path': file_path,
                'file_type': file_type,
                'chunks_created': len(chunks),
                'total_chunks': self.processing_stats['total_chunks'],
                'llm_calls_made': self.llm_call_count
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'document_path': file_path,
                'file_type': file_type
            }
    
    # ------------------------------------------------------------------
    # Upload validation
    # ------------------------------------------------------------------

    def _validate_upload(self, uploaded_file) -> Optional[str]:
        """
        Validate a Streamlit UploadedFile before any disk I/O.

        Returns an error message string if invalid, or None if the file
        passes all checks.

        Checks performed:
        1. File size  — rejects files larger than _MAX_FILE_SIZE_MB
        2. Extension  — only .pdf / .txt / .md are accepted
        3. Empty file — zero-byte uploads are rejected
        4. Filename safety — blocks path traversal and shell-injection chars
        """
        name = uploaded_file.name
        size = uploaded_file.size

        # 1. Size guard
        max_bytes = _MAX_FILE_SIZE_MB * 1024 * 1024
        if size > max_bytes:
            return (
                f"File '{name}' is {size / (1024*1024):.1f} MB — "
                f"maximum allowed size is {_MAX_FILE_SIZE_MB} MB."
            )

        # 2. Extension guard
        ext = Path(name).suffix.lower()
        if ext not in _ALLOWED_EXTENSIONS:
            return (
                f"File '{name}' has unsupported extension '{ext}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            )

        # 3. Empty file guard
        if size == 0:
            return f"File '{name}' is empty (0 bytes)."

        # 4. Filename safety guard
        if _UNSAFE_FILENAME_RE.search(name):
            return (
                f"File '{name}' contains unsafe characters. "
                "Please rename the file and try again."
            )

        return None  # all checks passed

    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Process an uploaded file from Streamlit.

        Validates the file first, then writes to a temp path, processes it
        through the RAG pipeline, and cleans up the temp file.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Processing result dict with 'success' bool and metadata.
        """
        # --- Validate before touching the filesystem ---
        error = self._validate_upload(uploaded_file)
        if error:
            return {
                'success':   False,
                'error':     error,
                'filename':  uploaded_file.name,
                'file_size': uploaded_file.size,
            }

        tmp_file_path: Optional[str] = None
        try:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_file_path = tmp.name

            result = self.process_document(
                tmp_file_path, suffix, uploaded_file.name
            )

            if result.get('success'):
                result['filename']          = uploaded_file.name
                result['file_size']         = uploaded_file.size
                result['original_filename'] = uploaded_file.name

            return result

        except Exception as exc:
            return {
                'success':   False,
                'error':     str(exc),
                'filename':  uploaded_file.name,
                'file_size': uploaded_file.size,
            }
        finally:
            # Always clean up the temp file, even if processing raised
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents using vector similarity - returns ALL relevant chunks with proper metadata.
        
        Args:
            query: Search query
            k: Number of results to return (increased to get more pages)
            
        Returns:
            List of relevant document chunks with scores and metadata
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        if not query or len(query.strip()) == 0:
            logger.warning("Empty search query provided")
            return []
        
        try:
            # Validate and sanitize k parameter
            if not isinstance(k, int) or k <= 0:
                k = 5
            # Increase k to get more chunks from different pages, but cap at 100 for safety
            results = self.vector_store.similarity_search_with_score(query, k=min(k * 3, 100))
            
            # Format results with complete metadata
            formatted_results = []
            seen_pages = set()
            
            for doc, score in results:
                page_num = doc.metadata.get('page', 'N/A')
                heading = doc.metadata.get('heading', 'N/A')
                
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score),
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': page_num,
                    'heading': heading,
                    'source_file': doc.metadata.get('source_file', 'Unknown')
                })
                
                seen_pages.add(page_num)
            
            logger.info("Retrieved %d chunks from %d pages", len(formatted_results), len(seen_pages))
            return formatted_results
            
        except Exception as e:
            logger.error("Document search failed: %s", e)
            return []
    
    def get_document_summary(self, query: str, max_chunks: int = 3) -> Dict[str, Any]:
        """
        Get a summary of relevant document chunks for a query.
        
        Args:
            query: Query to find relevant chunks
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Summary with relevant chunks and metadata
        """
        # Search for relevant chunks
        relevant_chunks = self.search_documents(query, k=max_chunks)
        
        if not relevant_chunks:
            return {
                'success': False,
                'error': 'No relevant documents found',
                'llm_calls_made': self.llm_call_count
            }
        
        # Create summary without additional LLM calls
        summary = {
            'success': True,
            'query': query,
            'relevant_chunks': relevant_chunks,
            'total_chunks_found': len(relevant_chunks),
            'average_similarity_score': sum(chunk['similarity_score'] for chunk in relevant_chunks) / len(relevant_chunks),
            'llm_calls_made': self.llm_call_count,
            'processing_stats': self.processing_stats.copy()
        }
        
        return summary
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self.processing_stats,
            'llm_calls_made': self.llm_call_count,
            'vector_store_initialized': self.vector_store is not None
        }
    
    def reset_processor(self):
        """Reset the document processor and clear all data."""
        self.vector_store = None
        self.documents = []
        self.llm_call_count = 0
        self.processing_stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_tokens_estimated': 0,
            'vector_store_size': 0
        }