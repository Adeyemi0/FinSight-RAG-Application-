"""Utilities for tracking and formatting source citations."""
from typing import List, Dict, Any
from langchain_core.documents import Document


class CitationTracker:
    """Tracks sources and generates citation references."""
    
    def __init__(self):
        self.sources: List[Document] = []
        self.source_map: Dict[str, int] = {}
    
    def add_document(self, doc: Document) -> int:
        """
        Add a document and return its source ID.
        
        Args:
            doc: LangChain Document with metadata
            
        Returns:
            Source ID (1-indexed)
        """
        # Create unique key from metadata
        doc_key = self._create_doc_key(doc)
        
        # Return existing ID if already added
        if doc_key in self.source_map:
            return self.source_map[doc_key]
        
        # Add new source
        source_id = len(self.sources) + 1
        self.sources.append(doc)
        self.source_map[doc_key] = source_id
        
        return source_id
    
    def _create_doc_key(self, doc: Document) -> str:
        """Create unique key for document deduplication."""
        metadata = doc.metadata
        filename = metadata.get('filename', 'unknown')
        chunk_id = metadata.get('chunk_id', 'unknown')
        return f"{filename}_{chunk_id}"
    
    def format_context_with_citations(self, documents: List[Document]) -> str:
        """
        Format documents into context string with source markers.
        
        Args:
            documents: List of LangChain Documents
            
        Returns:
            Formatted context string with [Source N] markers
        """
        context_parts = []
        
        for doc in documents:
            source_id = self.add_document(doc)
            
            # Format: [Source N] content
            context_parts.append(f"[Source {source_id}] {doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def get_sources_list(self) -> List[Dict[str, Any]]:
        """
        Get formatted list of all sources.
        
        Returns:
            List of source dictionaries with metadata
        """
        sources_list = []
        
        for idx, doc in enumerate(self.sources, start=1):
            metadata = doc.metadata
            
            # Get text preview (first 200 chars)
            text_preview = doc.page_content[:200]
            if len(doc.page_content) > 200:
                text_preview += "..."
            
            # Convert chunk_id to string if it exists (FIXED)
            chunk_id = metadata.get('chunk_id')
            if chunk_id is not None:
                chunk_id = str(chunk_id)
            
            source_info = {
                "source_id": idx,
                "filename": metadata.get('filename', 'unknown'),
                "doc_type": metadata.get('doc_type', 'unknown'),
                "ticker": metadata.get('ticker'),
                "similarity_score": float(metadata.get('similarity_score', 0.0)),
                "chunk_id": chunk_id,  # Now properly converted to string
                "text_preview": text_preview
            }
            
            sources_list.append(source_info)
        
        return sources_list
    
    def clear(self):
        """Clear all tracked sources."""
        self.sources.clear()
        self.source_map.clear()


def extract_citations_from_answer(answer: str) -> List[int]:
    """
    Extract citation numbers from answer text.
    
    Args:
        answer: Generated answer with [Source N] citations
        
    Returns:
        List of unique source IDs mentioned in answer
    """
    import re
    
    # Find all [Source N] patterns
    pattern = r'\[Source (\d+)\]'
    matches = re.findall(pattern, answer)
    
    # Convert to integers and remove duplicates
    cited_sources = sorted(set(int(m) for m in matches))
    
    return cited_sources