"""Main RAG chain orchestration with conversation memory."""
import time
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
from app.models import QueryRequest, QueryResponse, Source
from app.rag.retriever import ZillizRetriever
from app.rag.query_expander import QueryExpander
from app.rag.reranker import MMRReranker
from app.rag.compressor import ContextualCompressor
from app.utils.citations import CitationTracker
from app.utils.conversation import ConversationHistory
from app.utils.cache import cache_manager


# System prompt from specifications
SYSTEM_PROMPT = """You are an expert financial analyst AI. You provide accurate financial analysis from company financial statements and 10-K filings.

═══════════════════════════════════════════════════════════════════════════
CRITICAL RULE #1: ALWAYS USE "TOTAL" LINE ITEMS FROM FINANCIAL STATEMENTS
═══════════════════════════════════════════════════════════════════════════

Financial statements contain summary rows labeled "Total". These are the ONLY numbers you should use for calculations.

**MANDATORY LABELS TO USE (Look for these EXACT phrases):**

FROM BALANCE SHEET:
✓ "Total Current Assets" - Use this, NOT individual assets
✓ "Total Current Liabilities" - Use this, NOT individual liabilities  
✓ "Total Assets" - Use this
✓ "Total Liabilities" - Use this
✓ "Total Stockholders' Equity" or "Total Equity" - Use this
✓ "Long Term Debt" or "Long-term debt" - Use this
✓ "Short Term Debt" or "Short-term debt" - Use this (if needed)

FROM INCOME STATEMENT:
✓ "Contract Revenue" or "Revenue" or "Total Revenue" - Use this
✓ "Total Cost of Revenue" or "Cost of Revenue" - Use this
✓ "Gross Profit" - Use this (already calculated)
✓ "Operating Income" - Use this (already calculated)
✓ "Net Income" or "Profit or Loss" - Use this
✓ "Income Before Tax" - Use this

FROM CASH FLOW STATEMENT:
✓ "Net Cash from Operating Activities" - Use this
✓ "Net Cash from Investing Activities" - Use this
✓ "Net Cash from Financing Activities" - Use this

═══════════════════════════════════════════════════════════════════════════
FINANCIAL CALCULATIONS - COMPREHENSIVE GUIDE
═══════════════════════════════════════════════════════════════════════════

**LIQUIDITY RATIOS:**

1. Working Capital = Total Current Assets - Total Current Liabilities
   Example: $6.73B - $5.93B = $800M

2. Current Ratio = Total Current Assets ÷ Total Current Liabilities
   Example: $6.73B ÷ $5.93B = 1.13
   Interpretation: >1.0 is good (company can cover short-term obligations)

3. Quick Ratio = (Total Current Assets - Inventory) ÷ Total Current Liabilities
   Note: Only subtract inventory if explicitly asked for quick ratio

**LEVERAGE/SOLVENCY RATIOS:**

4. Debt-to-Equity Ratio = Total Debt ÷ Total Stockholders' Equity
   Where Total Debt = Long Term Debt + Short Term Debt
   Example: ($2.65B + $4.07M) ÷ $2.70B = 0.98
   Interpretation: <1.0 means less debt than equity (generally safer)

5. Debt-to-Assets Ratio = Total Debt ÷ Total Assets
   Example: $2.65B ÷ $12.20B = 0.22 or 22%

6. Equity Ratio = Total Stockholders' Equity ÷ Total Assets
   Example: $2.70B ÷ $12.20B = 0.22 or 22%

**PROFITABILITY RATIOS:**

7. Gross Profit Margin = (Gross Profit ÷ Revenue) × 100
   Example: ($1.22B ÷ $16.14B) × 100 = 7.6%

8. Operating Profit Margin = (Operating Income ÷ Revenue) × 100
   Example: ($1.03B ÷ $16.14B) × 100 = 6.4%

9. Net Profit Margin = (Net Income ÷ Revenue) × 100
   Example: ($561.77M ÷ $16.14B) × 100 = 3.5%

10. Return on Assets (ROA) = (Net Income ÷ Total Assets) × 100
    Example: ($561.77M ÷ $12.20B) × 100 = 4.6%

11. Return on Equity (ROE) = (Net Income ÷ Total Stockholders' Equity) × 100
    Example: ($561.77M ÷ $2.70B) × 100 = 20.8%

**EFFICIENCY RATIOS:**

12. Asset Turnover = Revenue ÷ Total Assets
    Example: $16.14B ÷ $12.20B = 1.32
    Interpretation: Company generates $1.32 in revenue for every $1 of assets

13. Inventory Turnover = Cost of Revenue ÷ Inventory
    (Only calculate if inventory is available in balance sheet)

**CASH FLOW ANALYSIS:**

14. Operating Cash Flow Margin = (Net Cash from Operating Activities ÷ Revenue) × 100
    
15. Free Cash Flow = Net Cash from Operating Activities - Capital Expenditures
    (Capital Expenditures = "Payments for Property, Plant and Equipment" from cash flow)

16. Cash Flow to Net Income Ratio = Net Cash from Operating Activities ÷ Net Income
    Interpretation: >1.0 means high quality earnings (cash backing profits)

**YEAR-OVER-YEAR (YoY) ANALYSIS:**

17. YoY Growth Rate = ((Current Year - Prior Year) ÷ Prior Year) × 100
    Example Revenue Growth: (($16.14B - $16.11B) ÷ $16.11B) × 100 = 0.19%
    
18. YoY Change (Dollar Amount) = Current Year - Prior Year
    Example: $16.14B - $16.11B = $30M increase

**TREND ANALYSIS (Multiple Years):**

19. When analyzing trends over 3+ years:
    - Calculate YoY change for each consecutive year
    - Identify if trend is increasing, decreasing, or stable
    - Note any significant inflection points

═══════════════════════════════════════════════════════════════════════════
HOW TO EXTRACT DATA FROM FINANCIAL STATEMENTS
═══════════════════════════════════════════════════════════════════════════

1. **Identify the fiscal year columns** (usually labeled FY 2025, FY 2024, etc.)

2. **Find the "Total" row** for what you need:
   - Scan the "label" column for rows starting with "Total"
   - Use the value from the appropriate fiscal year column

3. **For balance sheet items**, look in balance sheet sources
   
4. **For income statement items**, look in income statement sources

5. **For cash flow items**, look in cash flow statement sources

6. **NEVER add up individual line items** when a "Total" exists

═══════════════════════════════════════════════════════════════════════════
RESPONSE FORMAT (User-Friendly)
═══════════════════════════════════════════════════════════════════════════

**Structure:**
1. Direct Answer - What's the answer in one sentence?
2. Key Figures - List the relevant numbers with years and [Source X]
3. Calculation - Show the result (not the formula)
4. Analysis - What does this mean? Is it good or bad? What's the trend?
5. Sources - List all sources cited

**Writing Style:**
- Use simple language, not jargon
- Show formulas in plain text: "Revenue ÷ Gross Profit" or "Current Assets - Current Liabilities"
- Then show the calculation with actual numbers: "$16.14B ÷ $1.22B = 13.2"
- Use bullet points
- Compare to prior year when relevant
- State if trend is positive or negative for the company

**Example Good Answer:**
"Direct Answer: ACM's working capital was $800M in FY 2025.

Key Figures:
• FY 2025: Total Current Assets $6.73B, Total Current Liabilities $5.93B [Source 1]
• FY 2024: Total Current Assets $7.18B, Total Current Liabilities $6.37B [Source 1]

Calculation:
• Formula: Total Current Assets - Total Current Liabilities = Working Capital
• FY 2025: $6.73B - $5.93B = $800M
• FY 2024: $7.18B - $6.37B = $810M
• Change: $800M - $810M = -$10M decline (1.2% decrease)

Analysis:
Working capital decreased by $10M (1.2% decline). This slight reduction means ACM has marginally less liquidity to cover short-term obligations compared to last year, though the company still maintains positive working capital.

Sources: [Source 1] ACM_balance_sheet.md"

═══════════════════════════════════════════════════════════════════════════
FOR 10-K NARRATIVE SECTIONS
═══════════════════════════════════════════════════════════════════════════

When answering questions about 10-K narrative content (business description, risks, strategy):
- Summarize key points clearly
- Use bullet points for multiple items
- Quote important phrases when relevant
- Cite sources for each major point
- Group related information together

═══════════════════════════════════════════════════════════════════════════
CONVERSATION CONTEXT AND FOLLOW-UP QUESTIONS
═══════════════════════════════════════════════════════════════════════════

If the user asks a follow-up question that refers to previous context:
- Use the conversation history provided to understand the context
- Reference previous questions/answers when relevant (e.g., "As mentioned earlier...")
- Maintain consistency with previous responses
- If the follow-up requires new data, retrieve it from the documents

For pronoun references (e.g., "What about last year?" or "How does that compare?"):
- Infer what "that" or "it" refers to from the conversation history
- Explicitly state what you're comparing in your answer

═══════════════════════════════════════════════════════════════════════════
CRITICAL CONSTRAINTS
═══════════════════════════════════════════════════════════════════════════

**NEVER FABRICATE NUMBERS**: If specific information is not present in the provided context, explicitly state "This information is not available in the financial documents provided" and suggest consulting the company's official SEC filings or investor relations for complete information.

**DATA CUTOFF**: All financial data was collected on December 7, 2025. Information or events after this date are not available in this system.

**ACCURACY OVER COMPLETENESS**: It is better to say "I don't have this information" than to make up numbers or calculations."""


class RAGChain:
    """Main RAG chain for financial Q&A with conversation memory."""
    
    def __init__(self):
        """Initialize all RAG components."""
        self.retriever = ZillizRetriever()
        self.query_expander = QueryExpander()
        self.reranker = MMRReranker()
        self.compressor = ContextualCompressor()
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
            timeout=settings.LLM_TIMEOUT
        )
        
        # Conversation histories keyed by session_id
        self.conversations: dict[str, ConversationHistory] = {}
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", """{conversation_history}

Context from financial documents:

{context}

Question: {query}

Please provide a detailed answer using the context above. If this is a follow-up question, use the conversation history to understand the context. Remember to cite sources using [Source N] notation.""")
        ])
    
    def _get_or_create_conversation(self, session_id: str) -> ConversationHistory:
        """Get existing conversation or create new one."""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationHistory(max_tokens=4000)
        return self.conversations[session_id]
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query through the full RAG pipeline with conversation memory.
        
        Args:
            request: QueryRequest with query and filters
            
        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        
        # Check response cache (only for queries without session history)
        if settings.ENABLE_RESPONSE_CACHE and not request.session_id:
            cached_response = cache_manager.response_cache.get(
                query=request.query,
                ticker=request.ticker,
                doc_types=request.doc_types,
                top_k=request.top_k
            )
            if cached_response is not None:
                # Add processing time and return cached response
                cached_response['processing_time'] = round(time.time() - start_time, 2)
                cached_response['from_cache'] = True
                return QueryResponse(**cached_response)
        
        citation_tracker = CitationTracker()
        
        # Get or create session
        session_id = request.session_id or f"session_{int(time.time())}"
        conversation = self._get_or_create_conversation(session_id)
        
        try:
            # Step 1: Query Expansion
            if settings.ENABLE_QUERY_EXPANSION:
                expanded_queries = self.query_expander.expand(
                    request.query,
                    num_variations=settings.MAX_QUERY_VARIATIONS - 1
                )
            else:
                expanded_queries = [request.query]
            
            # Step 2: Retrieve documents for each query
            all_documents = []
            for query in expanded_queries:
                docs = self.retriever.retrieve(
                    query=query,
                    ticker=request.ticker,
                    doc_types=request.doc_types,
                    top_k=settings.RETRIEVAL_TOP_K
                )
                all_documents.extend(docs)
            
            # Step 3: Deduplicate documents
            unique_docs = self._deduplicate_documents(all_documents)
            
            # Step 4: Rerank documents
            if settings.ENABLE_RERANKING and len(unique_docs) > request.top_k:
                reranked_docs = self.reranker.rerank(
                    query=request.query,
                    documents=unique_docs,
                    top_k=request.top_k,
                    diversity_score=settings.MMR_DIVERSITY_SCORE
                )
            else:
                reranked_docs = unique_docs[:request.top_k]
            
            # Step 5: Contextual Compression
            if settings.ENABLE_COMPRESSION:
                compressed_docs = self.compressor.compress(
                    query=request.query,
                    documents=reranked_docs
                )
            else:
                compressed_docs = reranked_docs
            
            # Step 6: Prepare context with citations
            context = citation_tracker.format_context_with_citations(compressed_docs)
            
            # Step 7: Get conversation history
            conversation_history = ""
            if conversation.messages:
                history_msgs = conversation.get_messages()
                # Format last few exchanges
                recent_history = history_msgs[-6:]  # Last 3 exchanges
                if recent_history:
                    conversation_history = "Previous conversation:\n"
                    for msg in recent_history:
                        role_label = "User" if msg["role"] == "user" else "Assistant"
                        # Truncate long messages
                        content = msg["content"][:300]
                        if len(msg["content"]) > 300:
                            content += "..."
                        conversation_history += f"{role_label}: {content}\n\n"
            
            # Step 8: Generate answer
            chain = self.prompt | self.llm
            response = chain.invoke({
                "conversation_history": conversation_history,
                "context": context,
                "query": request.query
            })
            
            answer = response.content
            
            # Step 9: Update conversation history
            conversation.add_message("user", request.query)
            conversation.add_message("assistant", answer)
            
            # Step 10: Get sources list
            sources_list = citation_tracker.get_sources_list()
            sources = [Source(**src) for src in sources_list]
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            response_dict = {
                "answer": answer,
                "sources": [src.dict() if hasattr(src, 'dict') else src for src in sources],
                "query": request.query,
                "processing_time": round(processing_time, 2),
                "expanded_queries": expanded_queries if len(expanded_queries) > 1 else None,
                "num_documents_retrieved": len(compressed_docs),
                "session_id": session_id
            }
            
            # Cache response (only for queries without session history)
            if settings.ENABLE_RESPONSE_CACHE and not request.session_id:
                cache_manager.response_cache.set(
                    query=request.query,
                    response=response_dict,
                    ticker=request.ticker,
                    doc_types=request.doc_types,
                    top_k=request.top_k
                )
            
            return QueryResponse(**response_dict)
            
        except Exception as e:
            print(f"RAG chain error: {e}")
            raise
    
    def _deduplicate_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Remove duplicate documents based on content and metadata.
        
        Args:
            documents: List of documents
            
        Returns:
            Deduplicated list of documents
        """
        seen = set()
        unique_docs = []
        
        for doc in documents:
            # Create unique key
            metadata = doc.metadata
            key = f"{metadata.get('filename', '')}_{metadata.get('chunk_id', '')}_{doc.page_content[:100]}"
            
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        
        return unique_docs
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    async def aprocess_query(self, request: QueryRequest) -> QueryResponse:
        """
        Async version of process_query.
        
        Args:
            request: QueryRequest with query and filters
            
        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        
        # Check response cache (only for queries without session history)
        if settings.ENABLE_RESPONSE_CACHE and not request.session_id:
            cached_response = cache_manager.response_cache.get(
                query=request.query,
                ticker=request.ticker,
                doc_types=request.doc_types,
                top_k=request.top_k
            )
            if cached_response is not None:
                # Add processing time and return cached response
                cached_response['processing_time'] = round(time.time() - start_time, 2)
                cached_response['from_cache'] = True
                return QueryResponse(**cached_response)
        
        citation_tracker = CitationTracker()
        
        # Get or create session
        session_id = request.session_id or f"session_{int(time.time())}"
        conversation = self._get_or_create_conversation(session_id)
        
        try:
            # Query expansion
            if settings.ENABLE_QUERY_EXPANSION:
                expanded_queries = await self.query_expander.aexpand(
                    request.query,
                    num_variations=settings.MAX_QUERY_VARIATIONS - 1
                )
            else:
                expanded_queries = [request.query]
            
            # Retrieve documents
            all_documents = []
            for query in expanded_queries:
                docs = await self.retriever.aretrieve(
                    query=query,
                    ticker=request.ticker,
                    doc_types=request.doc_types,
                    top_k=settings.RETRIEVAL_TOP_K
                )
                all_documents.extend(docs)
            
            # Deduplicate
            unique_docs = self._deduplicate_documents(all_documents)
            
            # Rerank
            if settings.ENABLE_RERANKING and len(unique_docs) > request.top_k:
                reranked_docs = await self.reranker.arerank(
                    query=request.query,
                    documents=unique_docs,
                    top_k=request.top_k,
                    diversity_score=settings.MMR_DIVERSITY_SCORE
                )
            else:
                reranked_docs = unique_docs[:request.top_k]
            
            # Compress
            if settings.ENABLE_COMPRESSION:
                compressed_docs = await self.compressor.acompress(
                    query=request.query,
                    documents=reranked_docs
                )
            else:
                compressed_docs = reranked_docs
            
            # Prepare context
            context = citation_tracker.format_context_with_citations(compressed_docs)
            
            # Get conversation history
            conversation_history = ""
            if conversation.messages:
                history_msgs = conversation.get_messages()
                recent_history = history_msgs[-6:]
                if recent_history:
                    conversation_history = "Previous conversation:\n"
                    for msg in recent_history:
                        role_label = "User" if msg["role"] == "user" else "Assistant"
                        content = msg["content"][:300]
                        if len(msg["content"]) > 300:
                            content += "..."
                        conversation_history += f"{role_label}: {content}\n\n"
            
            # Generate answer
            chain = self.prompt | self.llm
            response = await chain.ainvoke({
                "conversation_history": conversation_history,
                "context": context,
                "query": request.query
            })
            
            answer = response.content
            
            # Update conversation history
            conversation.add_message("user", request.query)
            conversation.add_message("assistant", answer)
            
            sources_list = citation_tracker.get_sources_list()
            sources = [Source(**src) for src in sources_list]
            processing_time = time.time() - start_time
            
            response_dict = {
                "answer": answer,
                "sources": [src.dict() if hasattr(src, 'dict') else src for src in sources],
                "query": request.query,
                "processing_time": round(processing_time, 2),
                "expanded_queries": expanded_queries if len(expanded_queries) > 1 else None,
                "num_documents_retrieved": len(compressed_docs),
                "session_id": session_id
            }
            
            # Cache response (only for queries without session history)
            if settings.ENABLE_RESPONSE_CACHE and not request.session_id:
                cache_manager.response_cache.set(
                    query=request.query,
                    response=response_dict,
                    ticker=request.ticker,
                    doc_types=request.doc_types,
                    top_k=request.top_k
                )
            
            return QueryResponse(**response_dict)
            
        except Exception as e:
            print(f"RAG chain error: {e}")
            raise