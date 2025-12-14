"""Zilliz retriever with hybrid search capabilities."""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from app.config import settings
from app.utils.cache import cache_manager


class CachedEmbeddings(OpenAIEmbeddings):
    """Wrapper for OpenAI embeddings with caching."""
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with caching."""
        if settings.ENABLE_EMBEDDING_CACHE:
            cached = cache_manager.embedding_cache.get(text)
            if cached is not None:
                return cached
        
        # Get embedding from OpenAI
        embedding = super().embed_query(text)
        
        # Cache it
        if settings.ENABLE_EMBEDDING_CACHE:
            cache_manager.embedding_cache.set(text, embedding)
        
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if settings.ENABLE_EMBEDDING_CACHE:
                cached = cache_manager.embedding_cache.get(text)
                if cached is not None:
                    embeddings.append(cached)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    embeddings.append(None)  # Placeholder
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)
        
        # Get embeddings for uncached texts
        if uncached_texts:
            new_embeddings = super().embed_documents(uncached_texts)
            
            # Fill in the placeholders and cache
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                if settings.ENABLE_EMBEDDING_CACHE:
                    cache_manager.embedding_cache.set(texts[idx], embedding)
        
        return embeddings


class ZillizRetriever:
    """Retriever for Zilliz vector database with metadata filtering."""
    
    def __init__(self):
        """Initialize Zilliz connection and embeddings."""
        # Initialize OpenAI embeddings with caching
        self.embeddings = CachedEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            dimensions=settings.OPENAI_EMBEDDING_DIMENSION
        )
        
        # Initialize Milvus vector store
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=settings.COLLECTION_NAME,
            connection_args={
                "uri": settings.ZILLIZ_URI,
                "token": settings.ZILLIZ_TOKEN,
            },
            auto_id=True,
        )
    
    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        top_k: int = 30
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search (semantic + metadata filtering).
        
        Args:
            query: User query text
            ticker: Optional ticker symbol filter (e.g., 'ACM')
            doc_types: Optional list of document types to filter
            top_k: Number of documents to retrieve
            
        Returns:
            List of Document objects with metadata and similarity scores
        """
        # Check document cache
        if settings.ENABLE_DOCUMENT_CACHE:
            cached_docs = cache_manager.document_cache.get(query, ticker, doc_types)
            if cached_docs is not None:
                return cached_docs[:top_k]  # Return requested number
        
        # Build metadata filter expression
        filter_expr = self._build_filter_expression(ticker, doc_types)
        
        # Perform similarity search with metadata filtering
        if filter_expr:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                expr=filter_expr
            )
        else:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
        
        # Convert results to Documents with similarity scores in metadata
        documents = []
        for doc, score in results:
            # Add similarity score to metadata
            doc.metadata['similarity_score'] = float(score)
            documents.append(doc)
        
        # Cache the results
        if settings.ENABLE_DOCUMENT_CACHE:
            cache_manager.document_cache.set(query, documents, ticker, doc_types)
        
        return documents
    
    def _build_filter_expression(
        self,
        ticker: Optional[str],
        doc_types: Optional[List[str]]
    ) -> Optional[str]:
        """
        Build Milvus filter expression from parameters.
        
        Args:
            ticker: Optional ticker symbol
            doc_types: Optional list of document types
            
        Returns:
            Filter expression string or None
        """
        conditions = []
        
        if ticker:
            conditions.append(f'ticker == "{ticker}"')
        
        if doc_types:
            # Build OR condition for multiple doc types
            doc_type_conditions = [f'doc_type == "{dt}"' for dt in doc_types]
            conditions.append(f'({" or ".join(doc_type_conditions)})')
        
        if not conditions:
            return None
        
        # Combine with AND
        return " and ".join(conditions)
    
    async def aretrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        top_k: int = 30
    ) -> List[Document]:
        """
        Async version of retrieve method.
        
        Note: Current Milvus client is synchronous, so this wraps the sync call.
        For true async, would need async Milvus client.
        """
        import asyncio
        return await asyncio.to_thread(
            self.retrieve,
            query,
            ticker,
            doc_types,
            top_k
        )
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get collection info
            collection = self.vector_store.col
            
            stats = {
                "collection_name": settings.COLLECTION_NAME,
                "total_documents": collection.num_entities,
                "embedding_dimension": settings.OPENAI_EMBEDDING_DIMENSION,
            }
            
            return stats
        except Exception as e:
            return {
                "error": str(e),
                "collection_name": settings.COLLECTION_NAME
            }