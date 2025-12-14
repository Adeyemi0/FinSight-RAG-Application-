"""Configuration management for the RAG application."""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    OPENAI_EMBEDDING_DIMENSION: int = 3072
    
    # Zilliz Configuration
    ZILLIZ_URI: str
    ZILLIZ_TOKEN: str
    COLLECTION_NAME: str = "financial_documents"
    
    # RAG Configuration
    DEFAULT_TOP_K: int = 10
    RETRIEVAL_TOP_K: int = 30  # Retrieve more for reranking
    MAX_CONTEXT_TOKENS: int = 8000
    LLM_TIMEOUT: int = 30
    
    # Query Expansion
    ENABLE_QUERY_EXPANSION: bool = True
    MAX_QUERY_VARIATIONS: int = 3
    
    # Reranking
    ENABLE_RERANKING: bool = True
    MMR_DIVERSITY_SCORE: float = 0.3  # Balance between relevance and diversity
    
    # Compression
    ENABLE_COMPRESSION: bool = True
    
    # Caching
    ENABLE_QUERY_CACHE: bool = True
    ENABLE_EMBEDDING_CACHE: bool = True
    EMBEDDING_CACHE_SIZE: int = 1000
    EMBEDDING_CACHE_TTL: int = 86400  # 24 hours
    QUERY_CACHE_SIZE: int = 100
    QUERY_CACHE_TTL: int = 3600  # 1 hour
    
    # CORS - IMPORTANT: Update for Hugging Face
    ALLOWED_ORIGINS: list = [
        "*",  # Allow all origins for Hugging Face Spaces
        "https://huggingface.co",
        "https://*.hf.space",
    ]
    
    # Server Configuration - Hugging Face uses port 7860
    PORT: int = int(os.getenv("PORT", "7860"))
    HOST: str = "0.0.0.0"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()