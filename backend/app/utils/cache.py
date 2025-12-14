"""Caching system for embeddings, queries, and responses."""
import hashlib
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json


class CacheEntry:
    """Represents a cached item with expiration."""
    
    def __init__(self, value: Any, ttl: int = 3600):
        """
        Initialize cache entry.
        
        Args:
            value: Value to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.hit_count = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl
    
    def increment_hits(self):
        """Increment hit counter."""
        self.hit_count += 1


class EmbeddingCache:
    """Cache for query embeddings to avoid re-computing."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 86400):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of entries (default: 1000)
            ttl: Time to live in seconds (default: 24 hours)
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding.
        
        Args:
            text: Query text
            
        Returns:
            Cached embedding vector or None
        """
        key = self._generate_key(text)
        
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                entry.increment_hits()
                self.hits += 1
                return entry.value
            else:
                # Remove expired entry
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, text: str, embedding: List[float]):
        """
        Cache an embedding.
        
        Args:
            text: Query text
            embedding: Embedding vector
        """
        key = self._generate_key(text)
        
        # If cache is full, remove oldest entries
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = CacheEntry(embedding, ttl=self.ttl)
    
    def _evict_oldest(self):
        """Remove oldest 10% of entries."""
        num_to_remove = max(1, self.max_size // 10)
        
        # Sort by creation time and remove oldest
        sorted_keys = sorted(
            self.cache.keys(),
            key=lambda k: self.cache[k].created_at
        )
        
        for key in sorted_keys[:num_to_remove]:
            del self.cache[key]
    
    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }


class QueryResponseCache:
    """Cache for complete query responses."""
    
    def __init__(self, max_size: int = 500, ttl: int = 3600):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum number of entries (default: 500)
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _generate_key(
        self,
        query: str,
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        top_k: int = 10
    ) -> str:
        """Generate cache key from query parameters."""
        # Normalize inputs
        query_normalized = query.lower().strip()
        ticker_normalized = ticker.lower() if ticker else ""
        doc_types_normalized = sorted(doc_types) if doc_types else []
        
        # Create key string
        key_parts = [
            query_normalized,
            ticker_normalized,
            ",".join(doc_types_normalized),
            str(top_k)
        ]
        key_string = "|".join(key_parts)
        
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response.
        
        Args:
            query: Query text
            ticker: Ticker filter
            doc_types: Document type filters
            top_k: Number of results
            
        Returns:
            Cached response or None
        """
        key = self._generate_key(query, ticker, doc_types, top_k)
        
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                entry.increment_hits()
                self.hits += 1
                return entry.value
            else:
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(
        self,
        query: str,
        response: Dict[str, Any],
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        top_k: int = 10
    ):
        """
        Cache a response.
        
        Args:
            query: Query text
            response: Response to cache
            ticker: Ticker filter
            doc_types: Document type filters
            top_k: Number of results
        """
        key = self._generate_key(query, ticker, doc_types, top_k)
        
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = CacheEntry(response, ttl=self.ttl)
    
    def _evict_lru(self):
        """Remove least recently used 10% of entries."""
        num_to_remove = max(1, self.max_size // 10)
        
        # Sort by last access time (hit count and creation time)
        sorted_keys = sorted(
            self.cache.keys(),
            key=lambda k: (self.cache[k].hit_count, self.cache[k].created_at)
        )
        
        for key in sorted_keys[:num_to_remove]:
            del self.cache[key]
    
    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate cost savings (assuming $0.0001 per query)
        cost_per_query = 0.0001  # Approximate cost per LLM call
        estimated_savings = self.hits * cost_per_query
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests,
            "estimated_savings_usd": round(estimated_savings, 4)
        }


class DocumentCache:
    """Cache for retrieved documents to avoid vector searches."""
    
    def __init__(self, max_size: int = 200, ttl: int = 7200):
        """
        Initialize document cache.
        
        Args:
            max_size: Maximum number of entries (default: 200)
            ttl: Time to live in seconds (default: 2 hours)
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _generate_key(
        self,
        query: str,
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None
    ) -> str:
        """Generate cache key from search parameters."""
        query_normalized = query.lower().strip()
        ticker_normalized = ticker.lower() if ticker else ""
        doc_types_normalized = sorted(doc_types) if doc_types else []
        
        key_string = f"{query_normalized}|{ticker_normalized}|{','.join(doc_types_normalized)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None
    ) -> Optional[List[Any]]:
        """Get cached documents."""
        key = self._generate_key(query, ticker, doc_types)
        
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                entry.increment_hits()
                self.hits += 1
                return entry.value
            else:
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(
        self,
        query: str,
        documents: List[Any],
        ticker: Optional[str] = None,
        doc_types: Optional[List[str]] = None
    ):
        """Cache retrieved documents."""
        key = self._generate_key(query, ticker, doc_types)
        
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = CacheEntry(documents, ttl=self.ttl)
    
    def _evict_oldest(self):
        """Remove oldest 10% of entries."""
        num_to_remove = max(1, self.max_size // 10)
        sorted_keys = sorted(
            self.cache.keys(),
            key=lambda k: self.cache[k].created_at
        )
        
        for key in sorted_keys[:num_to_remove]:
            del self.cache[key]
    
    def clear(self):
        """Clear all cached documents."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }


class CacheManager:
    """Centralized cache management."""
    
    def __init__(self):
        """Initialize all caches."""
        self.embedding_cache = EmbeddingCache(max_size=1000, ttl=86400)  # 24h
        self.response_cache = QueryResponseCache(max_size=500, ttl=3600)  # 1h
        self.document_cache = DocumentCache(max_size=200, ttl=7200)  # 2h
    
    def clear_all(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        self.response_cache.clear()
        self.document_cache.clear()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            "embedding_cache": self.embedding_cache.get_stats(),
            "response_cache": self.response_cache.get_stats(),
            "document_cache": self.document_cache.get_stats(),
            "timestamp": datetime.now().isoformat()
        }


# Global cache manager instance
cache_manager = CacheManager()