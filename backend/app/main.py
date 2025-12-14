"""FastAPI application entry point - Hugging Face optimized."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import logging
import os

from app.config import settings
from app.models import QueryRequest, QueryResponse, HealthResponse, StatsResponse
from app.rag.chain import RAGChain
from app.rag.retriever import ZillizRetriever
from app.utils.cache import cache_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG chain instance
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global rag_chain
    
    # Startup
    logger.info("Initializing RAG chain...")
    logger.info(f"Running on port: {settings.PORT}")
    logger.info(f"CORS origins: {settings.ALLOWED_ORIGINS}")
    
    try:
        rag_chain = RAGChain()
        logger.info("RAG chain initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="FinSight RAG API",
    description="Production-ready LangChain RAG application for financial document Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware - Important for Hugging Face Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
# Check if frontend directory exists
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    logger.info(f"Frontend mounted at /static from {frontend_path}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - serve frontend."""
    frontend_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    return {
        "message": "FinSight RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "frontend": "Frontend not found. Use API directly."
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    """
    Get collection statistics.
    
    Returns:
        StatsResponse with collection information
    """
    try:
        retriever = ZillizRetriever()
        stats = retriever.get_collection_stats()
        
        return StatsResponse(
            collection_name=stats.get("collection_name", settings.COLLECTION_NAME),
            total_documents=stats.get("total_documents", 0),
            embedding_dimension=stats.get("embedding_dimension", settings.OPENAI_EMBEDDING_DIMENSION),
            available_tickers=["ACM"],  # Hardcoded for now
            available_doc_types=["balance_sheet", "cash_flow", "income_statement", "10k"]
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Main RAG query endpoint with conversation memory.
    
    Process a financial query through the RAG pipeline and return an answer with sources.
    Maintains conversation history within sessions for follow-up questions.
    
    Args:
        request: QueryRequest with query text, optional filters, and session_id
        
    Returns:
        QueryResponse with answer, sources, and session_id
    """
    global rag_chain
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    try:
        logger.info(f"Processing query: {request.query} [Session: {request.session_id or 'new'}]")
        
        # Process query through RAG chain
        response = await rag_chain.aprocess_query(request)
        
        logger.info(f"Query processed successfully in {response.processing_time}s [Session: {response.session_id}]")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.post("/query/sync", response_model=QueryResponse, tags=["Query"])
def query_documents_sync(request: QueryRequest):
    """
    Synchronous version of query endpoint with conversation memory.
    
    Args:
        request: QueryRequest with query text, optional filters, and session_id
        
    Returns:
        QueryResponse with answer, sources, and session_id
    """
    global rag_chain
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    try:
        logger.info(f"Processing query (sync): {request.query} [Session: {request.session_id or 'new'}]")
        
        # Process query through RAG chain
        response = rag_chain.process_query(request)
        
        logger.info(f"Query processed successfully in {response.processing_time}s [Session: {response.session_id}]")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.delete("/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """
    Clear conversation history for a session.
    
    Args:
        session_id: Session identifier to clear
        
    Returns:
        Success message
    """
    global rag_chain
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    try:
        rag_chain.clear_conversation(session_id)
        logger.info(f"Cleared session: {session_id}")
        return {"message": f"Session {session_id} cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")


@app.get("/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """
    Get cache statistics showing hit rates and cost savings.
    
    Returns:
        Dictionary with cache statistics for all cache layers
    """
    try:
        stats = cache_manager.get_all_stats()
        logger.info("Cache stats retrieved")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@app.delete("/cache/clear", tags=["Cache"])
async def clear_all_caches():
    """
    Clear all caches (embeddings and query results).
    
    Use this to force fresh results or manage memory.
    
    Returns:
        Success message
    """
    try:
        cache_manager.clear_all()
        logger.info("All caches cleared")
        return {"message": "All caches cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear caches: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False  # Disable reload in production
    )