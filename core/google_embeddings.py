"""
Google Embeddings for Research Assistant System

Uses google.genai SDK (replaces deprecated google.generativeai).
Compatible with LangChain's Embeddings interface for FAISS integration.
"""

import logging
from typing import List
from google import genai
from google.genai import types as genai_types
from langchain_core.embeddings import Embeddings
from config.settings import SystemConfig

logger = logging.getLogger(__name__)

# Embedding dimension for models/embedding-001
_EMBEDDING_DIM = SystemConfig.RAG_CONFIG["embedding_dimension"]


class GoogleEmbeddings(Embeddings):
    """
    Google Embeddings using the google.genai SDK.
    Compatible with LangChain's Embeddings interface for FAISS.
    """

    def __init__(
        self,
        api_key: str,
        model: str = SystemConfig.DOCUMENT_CONFIG["embedding_model"],
    ):
        self.model      = model
        self.call_count = 0
        self.total_tokens = 0
        self._client    = genai.Client(api_key=api_key)
        logger.info("GoogleEmbeddings ready | model=%s", model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document chunks for FAISS indexing."""
        embeddings = []
        for text in texts:
            try:
                result = self._client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=genai_types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    ),
                )
                embeddings.append(result.embeddings[0].values)
                self.call_count   += 1
                self.total_tokens += len(text.split())
            except Exception as exc:
                logger.error("Embedding failed for chunk (len=%d): %s", len(text), exc)
                embeddings.append([0.0] * _EMBEDDING_DIM)

        logger.info("Embedded %d documents | model=%s", len(texts), self.model)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query for FAISS similarity search."""
        try:
            result = self._client.models.embed_content(
                model=self.model,
                contents=text,
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY"
                ),
            )
            self.call_count   += 1
            self.total_tokens += len(text.split())
            logger.debug("Query embedded | model=%s", self.model)
            return result.embeddings[0].values
        except Exception as exc:
            logger.error("Query embedding failed: %s", exc)
            return [0.0] * _EMBEDDING_DIM

    def get_usage_stats(self) -> dict:
        return {
            "model":        self.model,
            "total_calls":  self.call_count,
            "total_tokens": self.total_tokens,
        }


class HuggingFaceEmbeddings(Embeddings):
    """
    Free local embeddings using HuggingFace sentence-transformers.
    No API calls — runs completely offline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.call_count = 0
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            logger.info("HuggingFaceEmbeddings ready | model=%s", model_name)
        except ImportError:
            logger.error("sentence-transformers not installed: pip install sentence-transformers")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts)
        self.call_count += len(texts)
        logger.info("Embedded %d documents locally", len(texts))
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode([text])
        self.call_count += 1
        return embedding[0].tolist()

    def get_usage_stats(self) -> dict:
        return {
            "model":       self.model_name,
            "total_calls": self.call_count,
            "total_cost":  0.0,
            "type":        "local",
        }