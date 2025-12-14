"""Query expansion for improved retrieval with multi-part question decomposition."""
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings


class QueryExpander:
    """Expands queries into multiple variations and decomposes multi-part questions."""
    
    def __init__(self):
        """Initialize LLM for query expansion."""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Prompt for detecting and decomposing multi-part questions
        self.decompose_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial query analyzer. Analyze if this is a multi-part question with distinct sub-questions.

If the query contains multiple DISTINCT questions (numbered or clearly separated topics), break it down into individual sub-queries.
If it's a single complex question, return it as-is.

Rules:
- Each sub-query should be standalone and answerable independently
- Preserve the ticker/company context in each sub-query
- Keep financial terminology intact
- Number sub-queries if there are multiple

Examples:
Input: "For ACM: 1. What was revenue? 2. What are the risks?"
Output:
1. What was ACM's revenue for the most recent fiscal year?
2. What are the major risks for ACM according to the latest 10-K?

Input: "What was ACM's revenue growth rate?"
Output:
What was ACM's revenue growth rate?

Return ONLY the decomposed queries, one per line. If single query, return it unchanged."""),
            ("user", "Query: {query}")
        ])
        
        # Prompt for expanding individual queries
        self.expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial query expansion expert. 
Generate {num_variations} alternative phrasings of the user's query that would help retrieve relevant financial information.

Focus on:
- Different financial terminology (e.g., "revenue" vs "sales" vs "contract revenue")
- Different ways to ask about financial metrics
- Explicit mention of financial statement types if relevant (balance sheet, income statement, cash flow, 10-K)
- Keeping the core intent of the original query

Return ONLY the query variations, one per line, without numbering or explanations."""),
            ("user", "Original query: {query}")
        ])
    
    def expand(self, query: str, num_variations: int = 2) -> List[str]:
        """
        Expand a query into multiple variations.
        Handles multi-part questions by decomposing them first.
        
        Args:
            query: Original query text
            num_variations: Number of variations to generate per sub-query (default: 2)
            
        Returns:
            List of query variations including the original and decomposed parts
        """
        # Always include the original query first
        all_queries = [query]
        
        try:
            # Step 1: Check if this is a multi-part question and decompose
            sub_queries = self._decompose_query(query)
            
            # Step 2: If decomposed into multiple parts, expand each part
            if len(sub_queries) > 1:
                print(f"Decomposed into {len(sub_queries)} sub-queries")
                for sub_query in sub_queries:
                    # Add the sub-query itself
                    if sub_query not in all_queries:
                        all_queries.append(sub_query)
                    
                    # Generate variations for this sub-query if it's complex enough
                    if self._should_expand(sub_query):
                        variations = self._generate_variations(sub_query, num_variations)
                        all_queries.extend([v for v in variations if v not in all_queries])
            else:
                # Single query - just expand normally if complex enough
                if self._should_expand(query):
                    variations = self._generate_variations(query, num_variations)
                    all_queries.extend([v for v in variations if v not in all_queries])
            
            return all_queries
            
        except Exception as e:
            print(f"Query expansion error: {e}")
            # Fallback to original query only
            return [query]
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a multi-part query into individual sub-queries.
        
        Args:
            query: Original query text
            
        Returns:
            List of sub-queries (or single query if not multi-part)
        """
        try:
            # Check if query has clear multi-part indicators
            multi_part_indicators = [
                '1.' in query and '2.' in query,
                '1)' in query and '2)' in query,
                query.count('?') > 1,  # Multiple question marks
                ' and ' in query.lower() and len(query.split()) > 15  # Long query with 'and'
            ]
            
            if not any(multi_part_indicators):
                return [query]
            
            # Use LLM to decompose
            chain = self.decompose_prompt | self.llm
            response = chain.invoke({"query": query})
            
            # Parse response
            sub_queries = response.content.strip().split('\n')
            sub_queries = [q.strip() for q in sub_queries if q.strip()]
            
            # Remove numbering if present (1., 2., etc.)
            cleaned_queries = []
            for q in sub_queries:
                # Remove leading numbering like "1. " or "1) "
                import re
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', q)
                if cleaned:
                    cleaned_queries.append(cleaned)
            
            return cleaned_queries if cleaned_queries else [query]
            
        except Exception as e:
            print(f"Query decomposition error: {e}")
            return [query]
    
    def _generate_variations(self, query: str, num_variations: int) -> List[str]:
        """
        Generate variations of a single query.
        
        Args:
            query: Query text to expand
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        try:
            chain = self.expansion_prompt | self.llm
            response = chain.invoke({
                "query": query,
                "num_variations": num_variations
            })
            
            # Parse response
            variations = response.content.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip()]
            
            return variations[:num_variations]
            
        except Exception as e:
            print(f"Variation generation error: {e}")
            return []
    
    def _should_expand(self, query: str) -> bool:
        """
        Determine if query should be expanded.
        
        Simple queries (< 5 words) or yes/no questions typically don't need expansion.
        
        Args:
            query: Original query text
            
        Returns:
            True if query should be expanded
        """
        # Don't expand very short queries
        word_count = len(query.split())
        if word_count < 5:
            return False
        
        # Don't expand simple what/when/where questions
        query_lower = query.lower().strip()
        simple_patterns = [
            query_lower.startswith("what is"),
            query_lower.startswith("what was"),
            query_lower.startswith("when did"),
            query_lower.startswith("where is"),
            "yes or no" in query_lower,
        ]
        
        if any(simple_patterns):
            return False
        
        # Expand complex queries
        return True
    
    async def aexpand(self, query: str, num_variations: int = 2) -> List[str]:
        """
        Async version of expand method.
        
        Args:
            query: Original query text
            num_variations: Number of variations to generate per sub-query
            
        Returns:
            List of query variations including the original and decomposed parts
        """
        all_queries = [query]
        
        try:
            # Decompose if multi-part
            sub_queries = await self._adecompose_query(query)
            
            if len(sub_queries) > 1:
                print(f"Decomposed into {len(sub_queries)} sub-queries")
                for sub_query in sub_queries:
                    if sub_query not in all_queries:
                        all_queries.append(sub_query)
                    
                    if self._should_expand(sub_query):
                        variations = await self._agenerate_variations(sub_query, num_variations)
                        all_queries.extend([v for v in variations if v not in all_queries])
            else:
                if self._should_expand(query):
                    variations = await self._agenerate_variations(query, num_variations)
                    all_queries.extend([v for v in variations if v not in all_queries])
            
            return all_queries
            
        except Exception as e:
            print(f"Query expansion error: {e}")
            return [query]
    
    async def _adecompose_query(self, query: str) -> List[str]:
        """Async version of _decompose_query."""
        try:
            multi_part_indicators = [
                '1.' in query and '2.' in query,
                '1)' in query and '2)' in query,
                query.count('?') > 1,
                ' and ' in query.lower() and len(query.split()) > 15
            ]
            
            if not any(multi_part_indicators):
                return [query]
            
            chain = self.decompose_prompt | self.llm
            response = await chain.ainvoke({"query": query})
            
            sub_queries = response.content.strip().split('\n')
            sub_queries = [q.strip() for q in sub_queries if q.strip()]
            
            # Remove numbering
            import re
            cleaned_queries = []
            for q in sub_queries:
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', q)
                if cleaned:
                    cleaned_queries.append(cleaned)
            
            return cleaned_queries if cleaned_queries else [query]
            
        except Exception as e:
            print(f"Query decomposition error: {e}")
            return [query]
    
    async def _agenerate_variations(self, query: str, num_variations: int) -> List[str]:
        """Async version of _generate_variations."""
        try:
            chain = self.expansion_prompt | self.llm
            response = await chain.ainvoke({
                "query": query,
                "num_variations": num_variations
            })
            
            variations = response.content.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip()]
            
            return variations[:num_variations]
            
        except Exception as e:
            print(f"Variation generation error: {e}")
            return []