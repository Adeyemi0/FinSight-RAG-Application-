"""Pydantic models for request/response validation."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG query endpoint."""
    
    query: str = Field(..., description="User's financial question")
    ticker: Optional[str] = Field(None, description="Filter by company ticker (e.g., 'ACM')")
    doc_types: Optional[List[str]] = Field(
        None, 
        description="Filter by document types: balance_sheet, cash_flow, income_statement, 10k"
    )
    top_k: int = Field(10, ge=1, le=20, description="Number of sources to retrieve")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What was ACM's revenue in 2024?",
                "ticker": "ACM",
                "doc_types": ["income_statement"],
                "top_k": 10,
                "session_id": "abc123"
            }
        }


class Source(BaseModel):
    """Source citation information."""
    
    source_id: int = Field(..., description="Source reference number [Source N]")
    filename: str = Field(..., description="Source filename")
    doc_type: str = Field(..., description="Document type")
    ticker: Optional[str] = Field(None, description="Company ticker")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    text_preview: str = Field(..., description="Preview of source text (first 200 chars)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_id": 1,
                "filename": "ACM_balance_sheet.md",
                "doc_type": "balance_sheet",
                "ticker": "ACM",
                "similarity_score": 0.89,
                "chunk_id": "chunk_0",
                "text_preview": "Total Current Assets for FY 2025: $6.73B..."
            }
        }


class QueryResponse(BaseModel):
    """Response model for RAG query endpoint."""
    
    answer: str = Field(..., description="AI-generated answer with citations")
    sources: List[Source] = Field(..., description="List of sources used")
    query: str = Field(..., description="Original query")
    processing_time: float = Field(..., description="Total processing time in seconds")
    expanded_queries: Optional[List[str]] = Field(None, description="Query variations used")
    num_documents_retrieved: int = Field(..., description="Number of documents retrieved")
    session_id: str = Field(..., description="Session ID for this conversation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "ACM's revenue in FY 2024 was $16.11B [Source 1]...",
                "sources": [
                    {
                        "source_id": 1,
                        "filename": "ACM_income_statement.md",
                        "doc_type": "income_statement",
                        "ticker": "ACM",
                        "similarity_score": 0.92,
                        "chunk_id": "chunk_0",
                        "text_preview": "Contract Revenue FY 2024: $16.11B..."
                    }
                ],
                "query": "What was ACM's revenue in 2024?",
                "processing_time": 2.34,
                "expanded_queries": ["What was ACM's revenue in 2024?"],
                "num_documents_retrieved": 5,
                "session_id": "abc123"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str = "1.0.0"
    

class StatsResponse(BaseModel):
    """Collection statistics response."""
    
    collection_name: str
    total_documents: int
    embedding_dimension: int
    available_tickers: List[str]
    available_doc_types: List[str]