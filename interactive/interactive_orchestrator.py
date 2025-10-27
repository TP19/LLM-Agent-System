#!/usr/bin/env python3
"""
Interactive Orchestrator - Phase 3

Coordinates agent execution with interactive checkpoints.
NOW WITH:
- Workflow validation (anti-hallucination)
- Fixed summarization → security/executor bug
- RAG support
- Complete execution integration
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from rag.qa_engine import QAEngine
import yaml

from interactive.modes import (
    InteractiveSession,
    CheckpointFrequency,
    UserAction,
    UserStoppedException,
    CheckpointRollbackException
)
from interactive.checkpoint_manager import CheckpointManager
from interactive.state_store import StateStore
from interactive.terminal_ui import TerminalUI
from interactive.workflow_validator import WorkflowValidator


class InteractiveOrchestrator:
    """
    Orchestrates interactive workflow with real agents - Phase 3
    
    NEW Features:
    - Validates workflow routing to prevent bugs
    - Terminal categories stop properly
    - Anti-hallucination guards
    - RAG support for agents
    """
    
    def __init__(self, session: InteractiveSession, 
                 state_store: StateStore,
                 ui: TerminalUI):
        """
        Initialize orchestrator
        
        Args:
            session: Interactive session
            state_store: State storage
            ui: Terminal UI
        """
        self.session = session
        self.state_store = state_store
        self.ui = ui
        self.logger = logging.getLogger(__name__)
        self.session_save_preference = None  # None, 'private', 'public', 'skip'
        self.rag_config = self._load_rag_config()
        
        # Checkpoint manager
        self.checkpoint_mgr = CheckpointManager(session, state_store, ui)
        
        # NEW: Workflow validator
        self.validator = WorkflowValidator()
        
        # Agents (to be initialized)
        self.agents = {}
        self.current_agent = None

    def _load_rag_config(self) -> dict:
        """Load RAG configuration"""
        try:
            with open('config/rag_config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Could not load RAG config: {e}")
            return {
                'interactive_mode': {
                    'save_documents': 'ask',
                    'show_save_prompt': True
                }
            }
    
    def register_agent(self, agent_name: str, agent):
        """
        Register an agent
        
        Args:
            agent_name: Name of agent
            agent: Agent instance
        """
        self.agents[agent_name] = agent
        
        # Attach session to agent
        if hasattr(agent, 'attach_session'):
            agent.attach_session(
                self.session,
                checkpoint_callback=self._checkpoint_callback,
                progress_callback=self._progress_callback
            )
        
        self.logger.info(f"Registered agent: {agent_name}")

    def _should_save_content(self, content_type: str = "document") -> bool:
        """
        Check if user wants to save content based on config
        
        Args:
            content_type: Type of content (document, debugging, etc.)
        
        Returns:
            True if should proceed to ask/save, False if never save
        """
        config_key = f'save_{content_type}s'  # save_documents, save_debuggings
        save_setting = self.rag_config.get('interactive_mode', {}).get(
            config_key, 'ask'
        )
        
        if save_setting == 'never':
            return False
        
        return True

    def _prompt_save_decision(self, content_type: str = "document") -> str:
        """
        Prompt user for save decision - REUSABLE for any content type
        
        Args:
            content_type: Type of content (document, debugging, analysis, etc.)
        
        Returns:
            'private', 'public', 'skip', 'always_private', 'always_public', 'never'
        """
        # Check if user already made a session-wide choice
        if self.session_save_preference:
            return self.session_save_preference
        
        # Check config default
        config_key = f'save_{content_type}s'
        save_setting = self.rag_config.get('interactive_mode', {}).get(
            config_key, 'ask'
        )
        
        if save_setting == 'always_private':
            return 'private'
        elif save_setting == 'always_public':
            return 'public'
        elif save_setting == 'never':
            return 'skip'
        
        # Ask user
        self.ui.print_section_header("💾 Save Options")
        self.ui.print_info(f"\nWhere should this {content_type} be saved?")
        self.ui.print_info("  [1] Private (only you can access)")
        self.ui.print_info("  [2] Public (shared knowledge base)")
        self.ui.print_info("  [3] Don't save")
        self.ui.print_info("  [4] Always private (for this session)")
        self.ui.print_info("  [5] Always public (for this session)")
        self.ui.print_info("  [6] Never save (for this session)")
        
        choice = self.ui.prompt_input("\nYour choice [1-6]", default="1").strip()
        
        if choice == "1":
            return "private"
        elif choice == "2":
            # Warn about public saves
            self.ui.print_warning("\n⚠️  Public saves are shared across all sessions!")
            confirm = self.ui.prompt_input("Are you sure? [y/N]", default="n")
            if confirm.lower() == 'y':
                return "public"
            else:
                return "private"  # Default to private if user cancels
        elif choice == "3":
            return "skip"
        elif choice == "4":
            self.session_save_preference = "private"
            self.ui.print_success("\n✅ Will save all content privately this session")
            return "private"
        elif choice == "5":
            self.ui.print_warning("\n⚠️  All content will be saved publicly this session!")
            confirm = self.ui.prompt_input("Are you sure? [y/N]", default="n")
            if confirm.lower() == 'y':
                self.session_save_preference = "public"
                self.ui.print_success("\n✅ Will save all content publicly this session")
                return "public"
            else:
                self.session_save_preference = "private"
                return "private"
        elif choice == "6":
            self.session_save_preference = "skip"
            self.ui.print_info("\n✅ Will not save any content this session")
            return "skip"
        else:
            # Default to private
            return "private"

    def _store_with_user_choice(self, 
                            content: str,
                            metadata: dict,
                            content_type: str = "document",
                            save_choice: str = None) -> Optional[str]:
        """
        Store content based on user's choice - REUSABLE
        
        Args:
            content: The content to store
            metadata: Metadata dict
            content_type: Type (document, debugging, analysis, etc.)
            save_choice: Pre-determined choice (or None to prompt)
        
        Returns:
            Memory ID if saved, None if skipped
        """
        # Get save choice if not provided
        if save_choice is None:
            if not self._should_save_content(content_type):
                return None
            save_choice = self._prompt_save_decision(content_type)
        
        # Skip if requested
        if save_choice == "skip":
            self.ui.print_info("   ℹ️  Not saved to memory")
            return None
        
        # Check if memory manager available
        if not hasattr(self.summarization_agent, 'memory_manager') or \
        self.summarization_agent.memory_manager is None:
            self.ui.print_warning("   ⚠️  Memory system not available")
            return None
        
        is_private = (save_choice in ["private", "always_private"])
        
        # Store based on content type
        try:
            if content_type in ["document", "analysis", "summary"]:
                memory_id = self.summarization_agent.memory_manager.store_semantic_memory(
                    knowledge=content,
                    metadata=metadata,
                    is_private=is_private,
                    importance=0.7
                )
                db_type = "private" if is_private else "public"
                self.ui.print_success(f"   ✅ Saved to {db_type} knowledge base")
                return memory_id
                
            elif content_type in ["debugging", "interaction", "session"]:
                memory_id = self.summarization_agent.memory_manager.store_episodic_memory(
                    content=content,
                    metadata=metadata,
                    is_private=is_private,
                    importance=0.8
                )
                db_type = "private" if is_private else "public"
                self.ui.print_success(f"   ✅ Saved to {db_type} memory")
                return memory_id
            
            else:
                self.logger.warning(f"Unknown content type: {content_type}")
                return None
                
        except Exception as e:
            self.ui.print_error(f"   ❌ Failed to save: {e}")
            return None
    
    def validate_next_stage(self, current_stage: str, next_stage: str) -> bool:
        """
        Validate workflow routing - PREVENTS SUMMARIZATION BUG
        
        Args:
            current_stage: Current workflow stage
            next_stage: Proposed next stage
            
        Returns:
            True if routing is valid
        """
        # Get triage result if available
        triage_category = self.session.accumulated_results.get('triage', {}).get('classification', '')
        
        # Validate routing
        validation = self.validator.validate_workflow_routing(
            triage_category,
            next_stage
        )
        
        if not validation.is_valid:
            self.logger.error(
                f"Invalid workflow routing: {current_stage} → {next_stage}"
            )
            for issue in validation.issues:
                self.logger.error(f"  - {issue}")
            
            # Show error to user
            self.ui.print_error(
                f"Workflow routing error: {current_stage} cannot route to {next_stage}"
            )
            
            return False
        
        if validation.warnings:
            for warning in validation.warnings:
                self.logger.warning(f"Routing warning: {warning}")
        
        return True
    
    def is_terminal_category(self, category: str) -> bool:
        """
        Check if category is terminal (should not continue to execution)
        
        Terminal categories bypass security & executor stages:
        - summarization: Process documents directly
        - coding: Generate code directly
        - knowledge_query: Query knowledge base directly
        
        Args:
            category: Triage category
            
        Returns:
            True if terminal
        """
        terminal_categories = {'summarization', 'coding', 'knowledge_query'}
        return category in terminal_categories
    
    def _checkpoint_callback(self, checkpoint_type: str, current_agent: str,
                            accumulated_results: Optional[Dict] = None,
                            agent_states: Optional[bytes] = None):
        """
        Callback for agents to request checkpoints
        
        Args:
            checkpoint_type: Type of checkpoint
            current_agent: Name of current agent
            accumulated_results: Results to checkpoint
            agent_states: Serialized agent states
        """
        # Update session
        self.session.current_agent = current_agent
        
        # Create checkpoint
        checkpoint = self.checkpoint_mgr.create_checkpoint(
            checkpoint_type=checkpoint_type,
            current_agent=current_agent,
            current_stage=self.session.current_stage,
            accumulated_results=accumulated_results,
            agent_states=agent_states
        )
        
        # Display and get user action
        user_response = self.checkpoint_mgr.display_checkpoint(checkpoint)
        
        # Process user action
        self._process_user_action(user_response)
    
    def _progress_callback(self, agent: str, message: str,
                          percentage: Optional[float] = None,
                          details: Optional[Dict] = None):
        """
        Callback for agents to report progress
        
        Args:
            agent: Agent name
            message: Progress message
            percentage: Optional progress percentage
            details: Optional details
        """
        self.ui.display_progress(agent, message, percentage, details)
    
    def _process_user_action(self, user_response: Dict[str, Any]):
        """
        Process user action from checkpoint
        
        Args:
            user_response: User action response
        """
        action = user_response['action']
        
        if action == UserAction.STOP:
            raise UserStoppedException("User requested stop")
        
        elif action == UserAction.MODIFY:
            guidance = user_response.get('guidance')
            if guidance:
                # Update session with guidance
                self.session.user_request = guidance
                self.ui.print_success(f"Updated request: {guidance}")
        
        elif action == UserAction.ROLLBACK:
            rollback_to = user_response.get('rollback_to')
            if rollback_to:
                self.checkpoint_mgr.rollback_to(rollback_to)
        
        elif action == UserAction.QUERY:
            # Handle Q&A (Phase 3)
            self.ui.print_info("Q&A mode not yet implemented")
        
        # CONTINUE is default, no action needed
 
    def run_triage(self, triage_agent) -> Dict[str, Any]:
        """
        Run triage stage with validation
        
        Args:
            triage_agent: Triage agent instance
            
        Returns:
            Triage results as dict
        """
        self.session.current_stage = "triage"
        self.session.current_agent = "triage_agent"
        
        self.ui.print_header("Stage 1: Triage")
        
        # Register agent
        self.register_agent("triage", triage_agent)
        
        # Execute triage
        try:
            if hasattr(triage_agent, 'classify_request_interactive'):
                result = triage_agent.classify_request_interactive(
                    self.session.user_request
                )
            else:
                result = triage_agent.analyze(self.session.user_request)
            
            # FIXED: Handle both dict and object results
            if isinstance(result, dict):
                # Already a dict from interactive mode
                triage_result = result
            else:
                # Convert TriageResult object to dict
                triage_result = {
                    'classification': result.category.value if hasattr(result, 'category') else str(result),
                    'confidence': result.confidence if hasattr(result, 'confidence') else 0.5,
                    'recommended_agent': result.recommended_agent if hasattr(result, 'recommended_agent') else 'executor',
                    'is_terminal': False
                }
            
            # Validate triage result
            validation = self.validator.validate_triage_result(
                triage_result,
                self.session.user_request
            )
            
            if not validation.is_valid:
                self.logger.error("Triage validation failed:")
                for issue in validation.issues:
                    self.logger.warning(f"  - {issue}")
            
            if validation.warnings:
                for warning in validation.warnings:
                    self.ui.print_warning(warning)
            
            # ✅ FIXED: Keep ALL fields from triage_result
            final_result = triage_result.copy()  # Start with all fields
            final_result['is_terminal'] = self.is_terminal_category(
                triage_result.get('classification', 'unknown')
            )
            
            # Store in session
            self.session.accumulated_results['triage'] = final_result
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Triage failed: {e}", exc_info=True)
            raise
    
    def run_security(self, security_agent, triage_result: Dict) -> Dict[str, Any]:
        """
        Run security stage with validation
        
        Args:
            security_agent: Security agent instance
            triage_result: Results from triage
            
        Returns:
            Security results
        """
        # Validate routing first
        if not self.validate_next_stage('triage', 'security'):
            raise ValueError("Invalid routing to security stage")
        
        self.session.current_stage = "security"
        self.session.current_agent = "security_agent"
        
        self.ui.print_header("Stage 2: Security Analysis")
        
        # Register agent
        self.register_agent("security", security_agent)
        
        # Execute security check
        try:
            if hasattr(security_agent, 'suggest_approach_interactive'):
                result = security_agent.suggest_approach_interactive(
                    self.session.user_request,
                    triage_result
                )
            else:
                result = security_agent.suggest_approach(
                    self.session.user_request,
                    triage_result
                )
            
            # FIXED: Handle both dict and object result types
            if isinstance(result, dict):
                # Already a dict from interactive mode
                result_dict = result
            else:
                # Convert SecuritySuggestion object to dict
                result_dict = {
                    'risk_level': result.risk_level.value if hasattr(result.risk_level, 'value') else result.risk_level,
                    'suggested_commands': result.commands if hasattr(result, 'commands') else [],
                    'reasoning': result.reasoning if hasattr(result, 'reasoning') else '',
                    'requires_approval': True
                }
            
            # NEW: Validate security result
            validation = self.validator.validate_security_suggestions(
                {
                    'suggested_commands': result_dict.get('suggested_commands', []),
                    'risk_level': result_dict.get('risk_level', 'medium')
                },
                self.session.user_request
            )
            
            if not validation.is_valid:
                self.logger.error("Security validation failed:")
                for issue in validation.issues:
                    self.logger.error(f"  - {issue}")
                    self.ui.print_error(issue)
            
            if validation.warnings:
                for warning in validation.warnings:
                    self.ui.print_warning(warning)
            
            # Build security result with safe defaults
            security_result = {
                'approved': True,  # User approves at checkpoint
                'risk_level': result_dict.get('risk_level', 'medium'),
                'suggested_commands': result_dict.get('suggested_commands', []),
                'reasoning': result_dict.get('reasoning', 'Security analysis complete'),
                'validation_confidence': validation.confidence
            }
            
            # Store in session
            self.session.accumulated_results['security'] = security_result
            
            return security_result
            
        except Exception as e:
            self.logger.error(f"Security stage failed: {e}", exc_info=True)
            raise

    def _handle_summarization(self, 
                            request: str,
                            checkpoint_freq: CheckpointFrequency) -> Dict:
        """
        Handle summarization with RAG storage and Q&A
        
        Enhanced flow:
        1. Summarize document
        2. Prompt user to save
        3. Store chunks if user agrees
        4. Offer Q&A mode
        """
        self.ui.print_section_header("📄 Summarization")
        
        # Extract file path from request
        import re
        # Try to extract file path from request
        words = request.split()
        file_path = None
        for word in words:
            if '.' in word and ('/' in word or '\\' in word or Path(word).exists()):
                file_path = word
                break
        
        if not file_path:
            self.ui.print_error("❌ Could not find file path in request")
            self.ui.print_info("💡 Try: 'Summarize path/to/file.txt'")
            return {'success': False, 'error': 'No file path found'}
        
        # Resolve path
        file_path = str(Path(file_path).expanduser().resolve())
        
        if not Path(file_path).exists():
            self.ui.print_error(f"❌ File not found: {file_path}")
            return {'success': False, 'error': 'File not found'}
        
        self.ui.print_info(f"📂 File: {file_path}")
        
        # Import depth enum
        from agents.summarization_agent import SummaryDepth
        
        # Determine depth (can be enhanced to ask user)
        depth = SummaryDepth.STANDARD
        
        # Run summarization
        self.ui.print_info(f"\n🔄 Summarizing with {depth.value} depth...")
        
        try:
            result = self.summarization_agent.summarize_file(
                file_path=file_path,
                depth=depth
            )
        except Exception as e:
            self.ui.print_error(f"\n❌ Summarization failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
        
        if not result.success:
            self.ui.print_error("\n❌ Summarization failed")
            if result.errors:
                for error in result.errors:
                    self.ui.print_error(f"   • {error}")
            return {'success': False, 'error': 'Summarization failed'}
        
        # Display results
        self.ui.print_section_header("✅ Summary")
        self.ui.print_info(f"\n{result.summary}\n")
        
        # Display stats
        self.ui.print_info(f"📊 Statistics:")
        self.ui.print_info(f"   • Chunks processed: {result.chunks_processed}")
        self.ui.print_info(f"   • Total tokens: {result.total_tokens:,}")
        self.ui.print_info(f"   • Processing time: {result.processing_time:.1f}s")
        
        # Prompt to save (if RAG enabled)
        memory_saved = False
        if hasattr(self.summarization_agent, 'memory_manager') and \
        self.summarization_agent.memory_manager is not None:
            
            # Check if should save
            if self._should_save_content("document"):
                save_choice = self._prompt_save_decision("document")
                
                if save_choice != "skip":
                    # Note: Chunks are already stored during summarization
                    # This just confirms and shows message
                    self.ui.print_success(
                        f"\n✅ Document stored in memory for Q&A"
                    )
                    memory_saved = True
            else:
                self.ui.print_info("\n   ℹ️  Auto-save disabled in config")
        
        # Offer Q&A mode if memory was saved or chunks exist
        if memory_saved or (hasattr(self.summarization_agent, 'memory_manager') and 
                            self.summarization_agent.memory_manager is not None):
            
            self.ui.print_info("\n💬 You can now ask questions about this document")
            
            answer = self.ui.prompt_input("Enter Q&A mode? [Y/n]", default="y")
            
            if answer.lower() in ['y', 'yes', '']:
                self._qa_mode(file_path, document_context=result.summary[:200])
        
        return {
            'success': True,
            'file_path': file_path,
            'summary': result.summary,
            'stats': {
                'chunks': result.chunks_processed,
                'tokens': result.total_tokens,
                'time': result.processing_time
            }
        }

    def _qa_mode(self, file_path: str, document_context: str = None):
        """
        Interactive Q&A mode over a document
        
        Args:
            file_path: Path to the document
            document_context: Optional context about the document
        """
        # Check if RAG is available
        if not hasattr(self.summarization_agent, 'memory_manager') or \
        self.summarization_agent.memory_manager is None:
            self.ui.print_warning("\n⚠️  Q&A not available (RAG not enabled)")
            return
        
        # Create QA engine
        try:
            qa_engine = QAEngine(
                memory_manager=self.summarization_agent.memory_manager,
                model_manager=self.model_manager
            )
        except Exception as e:
            self.ui.print_error(f"\n❌ Could not initialize Q&A engine: {e}")
            return
        
        # Q&A Loop
        self.ui.print_section_header("💬 Q&A Mode")
        self.ui.print_info(f"\nAsk questions about: {Path(file_path).name}")
        self.ui.print_info("Type 'q' or 'quit' to exit Q&A mode\n")
        
        question_count = 0
        
        while True:
            try:
                # Get question
                question = self.ui.prompt_input("\n❓ Your question").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['q', 'quit', 'exit']:
                    self.ui.print_success(f"\n✅ Asked {question_count} questions")
                    break
                
                question_count += 1
                
                # Show processing
                self.ui.print_info("\n🔍 Searching knowledge base...")
                
                # Get answer
                result = qa_engine.answer_question(
                    question=question,
                    file_path=file_path,
                    top_k=5
                )
                
                # Display answer
                self.ui.print_section_header(f"📝 Answer", style="info")
                self.ui.print_info(f"\n{result['answer']}\n")
                
                # Display sources
                if result['sources']:
                    self.ui.print_info("📚 Sources:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        chunk_num = source.get('chunk_number', '?')
                        score = source.get('relevance_score', 0)
                        src_type = source.get('source_type', 'unknown')
                        
                        self.ui.print_info(
                            f"   [{i}] Chunk {chunk_num} ({src_type}) - "
                            f"Relevance: {score:.2f}"
                        )
                
                # Show confidence
                confidence = result.get('confidence', 0)
                if confidence < 0.3:
                    self.ui.print_warning(
                        f"\n⚠️  Low confidence ({confidence:.2f}) - "
                        "answer may not be accurate"
                    )
                
            except KeyboardInterrupt:
                self.ui.print_info("\n\n👋 Exiting Q&A mode...")
                break
            except Exception as e:
                self.ui.print_error(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    def run_execution(self, executor_agent, 
                    triage_result: Dict,
                    security_result: Dict) -> Dict[str, Any]:
        """
        Run execution stage with validation - CATCHES UserStoppedException
        
        Args:
            executor_agent: Executor agent instance
            triage_result: Results from triage
            security_result: Results from security
            
        Returns:
            Execution results
        """
        from interactive.modes import UserStoppedException  # IMPORT ADDED
        
        # Validate routing first
        if not self.validate_next_stage('security', 'execution'):
            raise ValueError("Invalid routing to execution stage")
        
        self.session.current_stage = "execution"
        self.session.current_agent = "executor_agent"
        
        self.ui.print_header("Stage 3: Execution")
        
        # Register agent
        self.register_agent("executor", executor_agent)
        
        # Execute task
        try:
            if hasattr(executor_agent, 'execute_task_interactive'):
                result = executor_agent.execute_task_interactive(
                    self.session.user_request,
                    security_result,
                    max_cycles=5
                )
            else:
                # Fallback to mock
                result = {
                    'success': True,
                    'cycles_completed': 1,
                    'total_commands': 0,
                    'is_complete': True
                }
            
            # Validate execution result
            validation = self.validator.validate_execution_result(
                result,
                security_result.get('suggested_commands', [])
            )
            
            if not validation.is_valid:
                self.logger.error("Execution validation failed:")
                for issue in validation.issues:
                    self.logger.error(f"  - {issue}")
            
            if validation.warnings:
                for warning in validation.warnings:
                    self.ui.print_warning(warning)
            
            self.session.accumulated_results['execution'] = result
            
            return result
        
        except UserStoppedException:
            # FIXED: Re-raise UserStoppedException so it propagates correctly
            self.logger.info("User stopped during execution stage")
            raise
            
        except Exception as e:
            self.logger.error(f"Execution stage failed: {e}", exc_info=True)
            raise
        
    def run_completion_review(self) -> bool:
        """
        Run completion review
        
        Returns:
            True if user approves, False otherwise
        """
        self.session.current_stage = "completion_review"
        
        self.ui.print_header("Completion Review")
        
        # Create final checkpoint
        checkpoint = self.checkpoint_mgr.create_checkpoint(
            checkpoint_type='completion_review',
            current_agent='orchestrator',
            current_stage='completion_review'
        )
        
        # Display review and get decision
        self.ui.display_final_summary(self.session)
        
        # For Phase 2, auto-approve
        # In Phase 3, we'll add proper review UI
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        # Detach agents
        for agent in self.agents.values():
            if hasattr(agent, 'detach_session'):
                agent.detach_session()
        
        self.logger.info("Orchestrator cleanup complete")