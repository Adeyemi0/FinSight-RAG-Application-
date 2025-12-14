"""Contextual compression to extract relevant sentences from retrieved chunks."""
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings


class ContextualCompressor:
    """Compresses retrieved documents by extracting only relevant content."""
    
    def __init__(self):
        """Initialize LLM for compression."""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise information extraction assistant.

Given a query and a document chunk, extract ONLY the sentences that are directly relevant to answering the query.

Rules:
1. Extract complete sentences (don't cut off mid-sentence)
2. Maintain the original wording - do not paraphrase
3. Keep financial figures and context together
4. If the entire chunk is relevant, return it as-is
5. If nothing is relevant, return "NOT_RELEVANT"
6. Preserve numerical data and labels exactly as written

Return only the extracted sentences, separated by spaces."""),
            ("user", """Query: {query}

Document:
{document}

Relevant sentences:""")
        ])
    
    def compress(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        Compress documents by extracting relevant sentences.
        
        Args:
            query: Original query text
            documents: List of documents to compress
            
        Returns:
            List of compressed documents
        """
        if not documents:
            return []
        
        compressed_docs = []
        
        for doc in documents:
            try:
                # Skip very short documents (already concise)
                if len(doc.page_content) < 200:
                    compressed_docs.append(doc)
                    continue
                
                # Extract relevant content
                chain = self.prompt | self.llm
                response = chain.invoke({
                    "query": query,
                    "document": doc.page_content
                })
                
                extracted = response.content.strip()
                
                # Skip if nothing relevant found
                if extracted == "NOT_RELEVANT" or not extracted:
                    continue
                
                # Create compressed document with same metadata
                compressed_doc = Document(
                    page_content=extracted,
                    metadata=doc.metadata.copy()
                )
                compressed_docs.append(compressed_doc)
                
            except Exception as e:
                print(f"Compression error for doc: {e}")
                # Fallback: include original document
                compressed_docs.append(doc)
        
        return compressed_docs
    
    def compress_batch(
        self,
        query: str,
        documents: List[Document],
        batch_size: int = 5
    ) -> List[Document]:
        """
        Compress documents in batches for better performance.
        
        Args:
            query: Original query text
            documents: List of documents to compress
            batch_size: Number of documents to process at once
            
        Returns:
            List of compressed documents
        """
        if not documents:
            return []
        
        compressed_docs = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            compressed_batch = self.compress(query, batch)
            compressed_docs.extend(compressed_batch)
        
        return compressed_docs
    
    async def acompress(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        Async version of compress method.
        
        Args:
            query: Original query text
            documents: List of documents to compress
            
        Returns:
            List of compressed documents
        """
        if not documents:
            return []
        
        compressed_docs = []
        
        for doc in documents:
            try:
                if len(doc.page_content) < 200:
                    compressed_docs.append(doc)
                    continue
                
                chain = self.prompt | self.llm
                response = await chain.ainvoke({
                    "query": query,
                    "document": doc.page_content
                })
                
                extracted = response.content.strip()
                
                if extracted == "NOT_RELEVANT" or not extracted:
                    continue
                
                compressed_doc = Document(
                    page_content=extracted,
                    metadata=doc.metadata.copy()
                )
                compressed_docs.append(compressed_doc)
                
            except Exception as e:
                print(f"Compression error for doc: {e}")
                compressed_docs.append(doc)
        
        return compressed_docs