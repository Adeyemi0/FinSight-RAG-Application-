# backend/app/__init__.py
"""FinSight RAG Application."""

__version__ = "1.0.0"


# backend/app/rag/__init__.py
"""RAG components for document retrieval and processing."""

from app.rag.retriever import ZillizRetriever
from app.rag.query_expander import QueryExpander
from app.rag.reranker import MMRReranker
from app.rag.compressor import ContextualCompressor
from app.rag.chain import RAGChain

__all__ = [
    "ZillizRetriever",
    "QueryExpander",
    "MMRReranker",
    "ContextualCompressor",
    "RAGChain",
]


# backend/app/utils/__init__.py
"""Utility functions and helpers."""

from app.utils.citations import CitationTracker, extract_citations_from_answer
from app.utils.conversation import (
    ConversationMessage,
    ConversationHistory,
    SessionManager,
    session_manager
)
from app.utils.cache import (
    CacheEntry,
    EmbeddingCache,
    QueryResponseCache,
    DocumentCache,
    CacheManager,
    cache_manager
)

__all__ = [
    "CitationTracker",
    "extract_citations_from_answer",
    "ConversationMessage",
    "ConversationHistory",
    "SessionManager",
    "session_manager",
    "CacheEntry",
    "EmbeddingCache",
    "QueryResponseCache",
    "DocumentCache",
    "CacheManager",
    "cache_manager",
]