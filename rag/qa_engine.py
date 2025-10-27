"""
Complete Fixed Q&A Engine - rag/qa_engine.py

INSTRUCTIONS:
1. Replace the ENTIRE contents of rag/qa_engine.py with this file
2. This fixes metadata extraction and result formatting
"""

from typing import List, Dict, Optional
from .memory.memory_manager import MemoryManager
from core.model_manager import LazyModelManager
import logging

class QAEngine:
    """Question answering over stored documents - FIXED VERSION"""
    
    def __init__(self,
                 memory_manager: MemoryManager,
                 model_manager: LazyModelManager):
        self.memory = memory_manager
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
    
    def answer_question(self,
                       question: str,
                       file_path: Optional[str] = None,
                       top_k: int = 5) -> Dict:
        """
        Answer a question using RAG
        
        Args:
            question: User's question
            file_path: Specific file to query (optional - not used yet)
            top_k: Number of chunks to retrieve
            
        Returns:
            Dict with answer and sources
        """
        
        self.logger.info(f"🔍 Answering question: {question}")
        
        # Retrieve relevant memories
        try:
            relevant_memories = self.memory.retrieve_relevant_memories(
                query=question,
                top_k=top_k,
                memory_types=["semantic"]  # Query semantic collection
            )
        except Exception as e:
            self.logger.error(f"Failed to retrieve memories: {e}")
            import traceback
            traceback.print_exc()
            relevant_memories = []
        
        self.logger.info(f"   Retrieved {len(relevant_memories)} memories")
        
        # Debug: Log what we got
        if relevant_memories:
            self.logger.debug(f"   First memory metadata: {relevant_memories[0].metadata}")
        
        if not relevant_memories:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Build context from memories
        context = self._build_context(relevant_memories)
        
        # Generate answer using LLM
        prompt = self._build_qa_prompt(question, context)
        
        try:
            model = self.model_manager.get_model_for_agent("summarization")
            response = model(prompt, max_tokens=400, temperature=0.3)
            answer = response['choices'][0]['text'].strip()
        except Exception as e:
            self.logger.error(f"Failed to generate answer: {e}")
            answer = "I encountered an error generating the answer."
        
        # Format sources with proper metadata extraction
        sources = self._format_sources(relevant_memories)
        confidence = self._calculate_confidence(relevant_memories)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }
    
    def _build_context(self, memories: List) -> str:
        """Build context from retrieved memories"""
        context_parts = []
        
        for i, mem in enumerate(memories, 1):
            # Extract metadata safely
            file_path = mem.metadata.get('file_path', 'Unknown')
            chunk_num = mem.metadata.get('chunk_number', '?')
            
            # Get content preview
            content = mem.content
            if len(content) > 500:
                content = content[:500] + "..."
            
            context_parts.append(
                f"[Source {i} - {file_path}, Chunk {chunk_num}]\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_qa_prompt(self, question: str, context: str) -> str:
        """Build prompt for LLM"""
        return f"""Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer (cite sources by number when relevant):"""
    
    def _format_sources(self, memories: List) -> List[Dict]:
        """
        Format source citations with proper metadata extraction
        
        FIXED: Handles all metadata types safely
        """
        sources = []
        
        for i, mem in enumerate(memories, 1):
            # Extract metadata with robust fallbacks
            metadata = mem.metadata if hasattr(mem, 'metadata') else {}
            
            # Get file path
            file_path = metadata.get('file_path', 'Unknown Document')
            
            # Get chunk number - handle different types
            chunk_num = metadata.get('chunk_number')
            if chunk_num is None:
                chunk_num = '?'
            elif isinstance(chunk_num, (int, float)):
                chunk_num = int(chunk_num)
            else:
                chunk_num = str(chunk_num)
            
            # Get relevance score
            score = mem.importance_score if hasattr(mem, 'importance_score') else 0.0
            
            # Get memory type
            mem_type = mem.memory_type if hasattr(mem, 'memory_type') else 'unknown'
            
            # Build source entry
            source = {
                "source_number": i,
                "file_path": file_path,
                "chunk_number": chunk_num,
                "relevance_score": float(score),
                "source_type": mem_type,
                "content_preview": mem.content[:100] if len(mem.content) > 100 else mem.content
            }
            
            sources.append(source)
            
            # Debug log first source
            if i == 1:
                self.logger.debug(f"Source 1 formatted: {source}")
        
        return sources
    
    def _calculate_confidence(self, memories: List) -> float:
        """
        Calculate confidence based on retrieval scores
        
        FIXED: Handles negative and missing scores
        """
        if not memories:
            return 0.0
        
        # Get scores, handling missing/negative values
        scores = []
        for mem in memories:
            if hasattr(mem, 'importance_score'):
                score = mem.importance_score
                # Normalize negative scores
                if score < 0:
                    score = 1.0 / (1.0 + abs(score))  # Convert to 0-1 range
                scores.append(score)
        
        if not scores:
            return 0.0
        
        # Average score, capped at 1.0
        avg_score = sum(scores) / len(scores)
        return min(avg_score, 1.0)