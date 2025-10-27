#!/usr/bin/env python3
"""
Q&A Handler - Foundation for Phase 4

Enables users to ask questions about:
- Execution results
- Agent decisions
- Session history
- Stored context (via RAG)

This is a foundation - full implementation in Phase 4.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from interactive.modes import InteractiveSession
from interactive.terminal_ui import TerminalUI


class QAHandler:
    """
    Handle Q&A during and after execution
    
    Features:
    - Ask questions about results
    - Query execution history
    - Search stored context with RAG
    - Get agent explanations
    """
    
    def __init__(self, session: InteractiveSession, ui: TerminalUI):
        """
        Initialize Q&A handler
        
        Args:
            session: Current interactive session
            ui: Terminal UI instance
        """
        self.session = session
        self.ui = ui
        self.logger = logging.getLogger(__name__)
        
        # RAG retriever (optional)
        self.rag_retriever = None
        
        # Model manager for generating answers
        self.model_manager = None
    
    def enable_rag(self, rag_retriever):
        """
        Enable RAG for context-aware answers
        
        Args:
            rag_retriever: RAG retriever instance
        """
        self.rag_retriever = rag_retriever
        self.logger.info("✓ RAG enabled for Q&A")
    
    def set_model_manager(self, model_manager):
        """
        Set model manager for answer generation
        
        Args:
            model_manager: Model manager instance
        """
        self.model_manager = model_manager
        self.logger.info("✓ Model manager set for Q&A")
    
    def start_qa_session(self) -> bool:
        """
        Start interactive Q&A session
        
        Returns:
            True if session completed normally
        """
        self.ui.print_header("💬 Q&A Session")
        self.ui.print_info("Ask questions about the execution. Type 'done' to exit.")
        self.ui.print_info("Type 'help' for available commands.")
        print()
        
        while True:
            try:
                # Get question
                question = input("❓ Your question: ").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question.lower() in ['done', 'exit', 'quit']:
                    self.ui.print_success("Q&A session ended")
                    return True
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if question.lower() == 'summary':
                    self._show_summary()
                    continue
                
                if question.lower() == 'history':
                    self._show_history()
                    continue
                
                # Answer question
                answer = self.answer_question(question)
                
                print()
                self.ui.print_info(f"💡 Answer: {answer}")
                print()
                
                # Record interaction
                self.session.add_user_interaction(
                    'qa',
                    question,
                    answer
                )
                
            except KeyboardInterrupt:
                print()
                self.ui.print_warning("Q&A session interrupted")
                return False
            except Exception as e:
                self.logger.error(f"Q&A error: {e}", exc_info=True)
                self.ui.print_error(f"Error: {e}")
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question about the session
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        question_lower = question.lower()
        
        # Quick answers for common questions
        if 'how many' in question_lower and 'command' in question_lower:
            return self._answer_command_count()
        
        if 'what did' in question_lower or 'what happened' in question_lower:
            return self._answer_what_happened()
        
        if 'why' in question_lower:
            return self._answer_why_question(question)
        
        if 'error' in question_lower or 'fail' in question_lower:
            return self._answer_error_question()
        
        # Try RAG-enhanced answer
        if self.rag_retriever:
            return self._answer_with_rag(question)
        
        # Fallback: basic answer from session data
        return self._answer_from_session(question)
    
    def _answer_command_count(self) -> str:
        """Answer questions about command count"""
        execution = self.session.accumulated_results.get('execution', {})
        
        if 'total_commands' in execution:
            count = execution['total_commands']
            return f"Executed {count} command(s) during this session."
        
        return "No commands have been executed yet."
    
    def _answer_what_happened(self) -> str:
        """Answer 'what happened' questions"""
        if not self.session.execution_history:
            return "No actions have been taken yet."
        
        stages = []
        for entry in self.session.execution_history:
            stage = entry.get('stage', entry.get('action', 'unknown'))
            if stage not in stages:
                stages.append(stage)
        
        summary = f"The session went through {len(stages)} stages: "
        summary += " → ".join(stages)
        
        return summary
    
    def _answer_why_question(self, question: str) -> str:
        """Answer 'why' questions about decisions"""
        # Check triage reasoning
        triage = self.session.accumulated_results.get('triage', {})
        
        if triage and 'reasoning' in triage:
            return f"Triage reasoning: {triage['reasoning']}"
        
        return "Decision reasoning not available. Try asking about specific stages."
    
    def _answer_error_question(self) -> str:
        """Answer questions about errors"""
        # Check for errors in execution history
        errors = []
        
        for entry in self.session.execution_history:
            result = entry.get('result', {})
            if isinstance(result, dict):
                if not result.get('success', True):
                    errors.append(entry)
        
        if not errors:
            return "No errors were encountered during execution."
        
        return f"Found {len(errors)} error(s) during execution. Check the execution history for details."
    
    def _answer_with_rag(self, question: str) -> str:
        """
        Answer question using RAG
        
        Args:
            question: User's question
            
        Returns:
            RAG-enhanced answer
        """
        try:
            # Retrieve relevant context
            context = self.rag_retriever.build_context(
                question,
                k=3,
                max_tokens=1000
            )
            
            if not context:
                return self._answer_from_session(question)
            
            # If model manager available, use it to generate answer
            if self.model_manager:
                prompt = f"""Based on this context, answer the question:

Context:
{context}

Question: {question}

Answer:"""
                
                # This would use the model manager to generate answer
                # For now, just return context
                return f"Relevant context found:\n\n{context}"
            
            return f"Found relevant information:\n\n{context}"
            
        except Exception as e:
            self.logger.warning(f"RAG answer failed: {e}")
            return self._answer_from_session(question)
    
    def _answer_from_session(self, question: str) -> str:
        """
        Answer from session data without RAG
        
        Args:
            question: User's question
            
        Returns:
            Answer from session
        """
        # Search accumulated results
        question_lower = question.lower()
        relevant_data = []
        
        for key, value in self.session.accumulated_results.items():
            if key in question_lower or str(value).lower().find(question_lower) >= 0:
                relevant_data.append((key, value))
        
        if relevant_data:
            answer = "Found relevant information:\n"
            for key, value in relevant_data[:3]:  # Top 3
                answer += f"\n{key}: {value}"
            return answer
        
        return "I don't have enough information to answer that question. Try asking about specific stages or results."
    
    def _show_help(self):
        """Show help message"""
        help_text = """
Available commands:
  • Ask questions about execution results
  • 'summary' - Show session summary
  • 'history' - Show execution history
  • 'done' - Exit Q&A session
  • 'help' - Show this help

Example questions:
  • "How many commands were executed?"
  • "What happened during execution?"
  • "Why was this approach chosen?"
  • "Were there any errors?"
"""
        print(help_text)
    
    def _show_summary(self):
        """Show session summary"""
        self.ui.print_header("Session Summary")
        
        print(f"Session ID: {self.session.session_id}")
        print(f"Request: {self.session.user_request}")
        print(f"Current stage: {self.session.current_stage}")
        print(f"Status: {'Complete' if self.session.is_complete else 'In Progress'}")
        print(f"Checkpoints: {len(self.session.checkpoints)}")
        print(f"Execution steps: {len(self.session.execution_history)}")
        
        if self.session.accumulated_results:
            print(f"\nResults collected:")
            for key in self.session.accumulated_results.keys():
                print(f"  • {key}")
        
        print()
    
    def _show_history(self):
        """Show execution history"""
        self.ui.print_header("Execution History")
        
        if not self.session.execution_history:
            print("No execution history yet.")
            return
        
        for i, entry in enumerate(self.session.execution_history, 1):
            timestamp = entry.get('timestamp', 'unknown')
            action = entry.get('action', 'unknown')
            agent = entry.get('agent', 'system')
            
            print(f"{i}. [{timestamp}] {agent}: {action}")
        
        print()
    
    def ask_during_execution(self, question: str) -> str:
        """
        Quick question during execution (non-interactive)
        
        Args:
            question: Question to answer
            
        Returns:
            Answer string
        """
        self.logger.info(f"Quick Q&A: {question}")
        answer = self.answer_question(question)
        
        # Record but don't enter full Q&A mode
        self.session.add_user_interaction('quick_qa', question, answer)
        
        return answer


# Convenience function
def create_qa_handler(session: InteractiveSession,
                     ui: TerminalUI,
                     rag_retriever=None,
                     model_manager=None) -> QAHandler:
    """
    Create configured Q&A handler
    
    Args:
        session: Interactive session
        ui: Terminal UI
        rag_retriever: Optional RAG retriever
        model_manager: Optional model manager
        
    Returns:
        Configured QAHandler
    """
    handler = QAHandler(session, ui)
    
    if rag_retriever:
        handler.enable_rag(rag_retriever)
    
    if model_manager:
        handler.set_model_manager(model_manager)
    
    return handler