"""Reranking retrieved documents using MMR (Maximal Marginal Relevance)."""
from typing import List
import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from app.config import settings


class MMRReranker:
    """Reranks documents using Maximal Marginal Relevance algorithm."""
    
    def __init__(self):
        """Initialize embeddings for MMR computation."""
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            dimensions=settings.OPENAI_EMBEDDING_DIMENSION
        )
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10,
        diversity_score: float = 0.3
    ) -> List[Document]:
        """
        Rerank documents using MMR to balance relevance and diversity.
        
        MMR Formula:
        MMR = argmax [λ * Sim(D_i, Q) - (1-λ) * max Sim(D_i, D_j)]
        where D_j are already selected documents
        
        Args:
            query: Original query text
            documents: List of retrieved documents
            top_k: Number of documents to return
            diversity_score: Lambda parameter (0 = max diversity, 1 = max relevance)
            
        Returns:
            Reranked list of top_k documents
        """
        if not documents:
            return []
        
        # If we have fewer documents than top_k, return all
        if len(documents) <= top_k:
            return documents
        
        try:
            # Get embeddings
            query_embedding = self.embeddings.embed_query(query)
            doc_texts = [doc.page_content for doc in documents]
            doc_embeddings = self.embeddings.embed_documents(doc_texts)
            
            # Convert to numpy arrays
            query_vec = np.array(query_embedding)
            doc_vecs = np.array(doc_embeddings)
            
            # Compute similarity to query for all documents
            query_similarities = self._cosine_similarity(query_vec, doc_vecs)
            
            # MMR selection
            selected_indices = []
            remaining_indices = list(range(len(documents)))
            
            for _ in range(min(top_k, len(documents))):
                if not remaining_indices:
                    break
                
                mmr_scores = []
                
                for idx in remaining_indices:
                    # Relevance to query
                    relevance = query_similarities[idx]
                    
                    # Redundancy with already selected documents
                    if selected_indices:
                        selected_vecs = doc_vecs[selected_indices]
                        redundancy = np.max(
                            self._cosine_similarity(doc_vecs[idx], selected_vecs)
                        )
                    else:
                        redundancy = 0
                    
                    # MMR score
                    mmr = diversity_score * relevance - (1 - diversity_score) * redundancy
                    mmr_scores.append((idx, mmr))
                
                # Select document with highest MMR score
                best_idx = max(mmr_scores, key=lambda x: x[1])[0]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # Return reranked documents
            return [documents[i] for i in selected_indices]
            
        except Exception as e:
            print(f"Reranking error: {e}")
            # Fallback: return top_k by original similarity score
            return self._fallback_rerank(documents, top_k)
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between vectors.
        
        Args:
            vec1: Single vector or array of vectors
            vec2: Array of vectors
            
        Returns:
            Similarity scores
        """
        if vec1.ndim == 1:
            vec1 = vec1.reshape(1, -1)
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)
        
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
        vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
        
        # Compute dot product
        similarity = np.dot(vec1_norm, vec2_norm.T)
        
        return similarity.flatten() if similarity.shape[0] == 1 else similarity
    
    def _fallback_rerank(
        self,
        documents: List[Document],
        top_k: int
    ) -> List[Document]:
        """
        Fallback reranking using existing similarity scores.
        
        Args:
            documents: List of documents with similarity_score in metadata
            top_k: Number of documents to return
            
        Returns:
            Top-k documents sorted by similarity score
        """
        # Sort by similarity score (higher is better)
        sorted_docs = sorted(
            documents,
            key=lambda d: d.metadata.get('similarity_score', 0),
            reverse=True
        )
        
        return sorted_docs[:top_k]
    
    async def arerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10,
        diversity_score: float = 0.3
    ) -> List[Document]:
        """
        Async version of rerank method.
        
        Args:
            query: Original query text
            documents: List of retrieved documents
            top_k: Number of documents to return
            diversity_score: Lambda parameter for MMR
            
        Returns:
            Reranked list of top_k documents
        """
        import asyncio
        return await asyncio.to_thread(
            self.rerank,
            query,
            documents,
            top_k,
            diversity_score
        )