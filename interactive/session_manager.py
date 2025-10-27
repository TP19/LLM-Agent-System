#!/usr/bin/env python3
"""
Session Manager - Phase 3 COMPLETE

Now with:
- Fixed summarization → security/executor bug
- Anti-hallucination validation
- RAG support
- Complete agent integration
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from agents.knowledge_agent import KnowledgeAgent

from interactive.modes import (
    InteractiveSession,
    CheckpointFrequency,
    InteractionMode,
    UserAction,
    UserStoppedException,
    CheckpointRollbackException
)
from interactive.state_store import StateStore
from interactive.terminal_ui import TerminalUI
from interactive.checkpoint_manager import CheckpointManager
from interactive.interactive_orchestrator import InteractiveOrchestrator


class SessionManager:
    """
    Session Manager - Phase 3
    
    CRITICAL FIX: Terminal categories (summarization, coding) 
    no longer route to security/executor
    """
    
    def __init__(self, frequency: CheckpointFrequency = CheckpointFrequency.STANDARD,
                db_path: Optional[str] = None,
                model_manager=None):
        """
        Initialize session manager
        
        Args:
            frequency: Checkpoint frequency setting
            db_path: Path to database
            model_manager: Model manager for agents
        """
        self.frequency = frequency
        self.model_manager = model_manager
        
        # Setup database path
        if db_path is None:
            db_path = Path.home() / ".llm_engine" / "interactive.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.state_store = StateStore(str(self.db_path))
        self.ui = TerminalUI()
        self.logger = logging.getLogger(__name__)
        
        # Current session
        self.session: Optional[InteractiveSession] = None
        self.orchestrator: Optional[InteractiveOrchestrator] = None
        
        # NEW: Initialize agents with RAG support
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents with RAG support"""
        self.logger.info("Initializing agents...")
        
        # Initialize RAG system
        try:
            from rag.vector_stores.dual_store_manager import DualStoreManager
            from rag.embedding.embedding_engine import EmbeddingEngine
            from rag.retrieval.reranker import Reranker
            from rag.memory.memory_manager import MemoryManager
            
            db_base = Path.home() / ".llm_engine" / "vector_db"
            db_base.mkdir(parents=True, exist_ok=True)
            
            embedder = EmbeddingEngine(lazy_load=True)
            reranker = Reranker(lazy_load=True)
            
            store_manager = DualStoreManager(
                private_db_path=str(db_base / "private"),
                public_db_path=str(db_base / "public")
            )
            
            self.memory_manager = MemoryManager(
                store_manager=store_manager,
                embedding_engine=embedder,
                reranker=reranker
            )
            
            self.logger.info("✅ RAG system enabled")
            
        except Exception as e:
            self.logger.warning(f"⚠️ RAG system not available: {e}")
            self.memory_manager = None
        
        # Initialize agents
        from agents.triage_agent import ModularTriageAgent
        from agents.security_agent import ModularSecurityAgent
        from agents.executor_agent import ModularExecutorAgent
        from agents.coder_agent import ModularCoderAgent
        from agents.summarization_agent import ModularSummarizationAgent
        
        self.triage_agent = ModularTriageAgent(self.model_manager)
        self.security_agent = ModularSecurityAgent(self.model_manager)
        self.executor_agent = ModularExecutorAgent(self.model_manager)
        self.coder_agent = ModularCoderAgent(self.model_manager)
        self.knowledge_agent = KnowledgeAgent(
            self.model_manager,
            memory_manager=self.memory_manager
        )
        
        # Summarization agent with memory manager
        self.summarization_agent = ModularSummarizationAgent(
            self.model_manager,
            memory_manager=self.memory_manager
        )
        
        self.logger.info("✅ All agents initialized")
    
    def create_session(self, user_request: str) -> InteractiveSession:
        """Create new interactive session"""
        session_id = str(uuid.uuid4())[:8]
        
        session = InteractiveSession(
            session_id=session_id,
            created_at=datetime.now(),
            mode=InteractionMode.INTERACTIVE,
            checkpoint_frequency=self.frequency,
            user_request=user_request,
            original_request=user_request,
            current_stage="initialization",
            current_agent=None,
            current_cycle=0
        )
        
        self.state_store.save_session(session)
        self.logger.info(f"Created session {session_id}")
        return session
    
    def run_interactive(self, user_request: str) -> InteractiveSession:
        """
        Run interactive session - Phase 3 COMPLETE
        
        Properly routes based on triage category:
        - summarization → completion (TERMINAL)
        - coding → completion (TERMINAL)
        - others → security → execution
        """
        # Create session
        self.session = self.create_session(user_request)
        
        # Create orchestrator with validation
        self.orchestrator = InteractiveOrchestrator(
            self.session,
            self.state_store,
            self.ui
        )
        
        # Display welcome
        self.ui.display_welcome()
        self.ui.display_session_start(self.session)
        
        try:
            # Stage 1: Initialization
            self._run_initialization()
            
            # Stage 2: Triage
            triage_result = self._run_triage_with_agent()
            
            # CRITICAL FIX: Check if terminal category
            if triage_result.get('is_terminal', False):
                self.logger.info(
                    f"Terminal category detected: "
                    f"{triage_result['classification']} - skipping execution"
                )
                
                # Handle terminal categories
                category = triage_result['classification']
                
                if category == 'summarization':
                    self._run_summarization(triage_result)
                elif category == 'coding':
                    self._run_coding(triage_result)
                elif category == 'knowledge_query':
                    self._run_knowledge_query(triage_result)
                
                # Go directly to completion
                self.orchestrator.run_completion_review()
                
            else:
                # Non-terminal: continue to security & execution
                self.logger.info(
                    f"Non-terminal category: {triage_result['classification']} "
                    "- proceeding to security & execution"
                )
                
                # Stage 3: Security
                security_result = self._run_security_with_agent(triage_result)
                
                # Stage 4: Execution
                execution_result = self._run_execution_with_agent(
                    triage_result,
                    security_result
                )
                
                # Stage 5: Completion review
                self.orchestrator.run_completion_review()
            
            # Mark complete
            self.session.is_complete = True
            self.session.completion_reason = "Workflow completed successfully"
            self.state_store.save_session(self.session)
            
            self.ui.display_session_complete(self.session)
            
        except UserStoppedException as e:
            self.logger.info(f"User stopped session: {e}")
            self.session.is_complete = True
            self.session.completion_reason = "User stopped execution"
            self.state_store.save_session(self.session)
            self.ui.print_warning("Session stopped by user")
            
        except CheckpointRollbackException as e:
            self.logger.info(f"Rollback requested: {e}")
            self.ui.print_info(f"Rollback to checkpoint: {e.checkpoint_id}")
            
        except Exception as e:
            self.logger.error(f"Session error: {e}", exc_info=True)
            self.session.is_complete = True
            self.session.error_message = str(e)
            self.state_store.save_session(self.session)
            self.ui.print_error(f"Session failed: {e}")
        
        finally:
            if self.orchestrator:
                self.orchestrator.cleanup()
        
        return self.session
    
    def _run_initialization(self):
        """Initialize session"""
        self.session.current_stage = "initialization"
        self.ui.print_info("Initializing session...")
        
        self.session.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'stage': 'initialization',
            'action': 'session_started',
            'details': {
                'session_id': self.session.session_id,
                'request': self.session.user_request
            }
        })
        
        self.state_store.save_session(self.session)
    
    def _run_triage_with_agent(self) -> Dict[str, Any]:
        """Run triage with real agent - FIXED"""
        try:
            from interactive.interactive_triage_agent import create_interactive_triage_agent
            
            self.logger.info("Loading interactive triage agent...")
            
            triage_agent = create_interactive_triage_agent(
                self.model_manager,
                self.session
            )
            
            self.logger.info("Triage agent loaded, running triage...")
            
            result = self.orchestrator.run_triage(triage_agent)
            
            self.logger.info(f"Triage completed: {result.get('classification', 'unknown')}")
            
            return result
            
        except ImportError as e:
            self.logger.error(f"Could not load triage agent: {e}", exc_info=True)
            self.logger.warning("Falling back to mock triage")
            return self._run_mock_triage()
            
        except Exception as e:
            self.logger.error(f"Triage failed: {e}", exc_info=True)
            self.logger.warning("Falling back to mock triage")
            return self._run_mock_triage()
    
    def _run_mock_triage(self) -> Dict[str, Any]:
        """Mock triage for testing"""
        self.session.current_stage = "triage"
        self.ui.print_header("Stage 1: Triage (Mock)")
        self.ui.print_info("Analyzing request...")
        
        # Detect if summarization request
        request_lower = self.session.user_request.lower()
        is_summarization = any(
            word in request_lower
            for word in ['summarize', 'summary', 'analyze file', 'analyze document']
        )
        
        result = {
            'classification': 'summarization' if is_summarization else 'sysadmin',
            'confidence': 0.95,
            'recommended_agent': 'summarization' if is_summarization else 'executor',
            'is_terminal': is_summarization
        }
        
        self.session.accumulated_results['triage'] = result
        return result
    
    def _run_security_with_agent(self, triage_result: Dict) -> Dict[str, Any]:
        """Run security check with real agent - CATCHES UserStoppedException"""
        from interactive.modes import UserStoppedException  # IMPORT ADDED
        
        try:
            from interactive.interactive_security_agent import create_interactive_security_agent
            
            self.logger.info("Loading interactive security agent...")
            
            security_agent = create_interactive_security_agent(
                self.model_manager,
                self.session,
                enable_rag=True  # Enable RAG for command history
            )
            
            self.logger.info("Security agent loaded, running security check...")
            
            result = self.orchestrator.run_security(security_agent, triage_result)
            
            self.logger.info("Security check completed")
            
            return result
        
        except UserStoppedException:
            # FIXED: Re-raise UserStoppedException so start_interactive.py can catch it
            self.logger.info("User stopped during security stage")
            raise
            
        except ImportError as e:
            self.logger.error(f"Could not load security agent: {e}", exc_info=True)
            self.logger.warning("Falling back to mock security")
            return {
                'approved': True,
                'risk_level': 'low',
                'suggested_commands': []
            }
            
        except Exception as e:
            self.logger.error(f"Security stage failed: {e}", exc_info=True)
            self.logger.warning("Falling back to mock security")
            return {
                'approved': True,
                'risk_level': 'low',
                'suggested_commands': []
            }
    
    def _run_execution_with_agent(self, triage_result: Dict, 
                                security_result: Dict) -> Dict[str, Any]:
        """Run execution with real agent - CATCHES UserStoppedException"""
        from interactive.modes import UserStoppedException  # IMPORT ADDED
        
        try:
            from interactive.interactive_executor_agent import create_interactive_executor_agent
            
            self.logger.info("Loading interactive executor agent...")
            
            executor_agent = create_interactive_executor_agent(
                self.model_manager,
                self.session
            )
            
            self.logger.info("Executor agent loaded, running execution...")
            
            result = self.orchestrator.run_execution(
                executor_agent,
                triage_result,
                security_result
            )
            
            self.logger.info("Execution completed")
            
            return result
        
        except UserStoppedException:
            # FIXED: Re-raise UserStoppedException so start_interactive.py can catch it
            self.logger.info("User stopped during execution")
            raise
            
        except ImportError as e:
            self.logger.error(f"Could not load executor agent: {e}", exc_info=True)
            self.logger.warning("Falling back to mock execution")
            return self._run_mock_execution()
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}", exc_info=True)
            self.logger.warning("Falling back to mock execution")
            return self._run_mock_execution()
    
    def _run_summarization(self, triage_result: Dict):
        """
        Handle summarization workflow - TERMINAL CATEGORY
        
        Args:
            triage_result: Results from triage
        """
        self.logger.info("Running summarization workflow")
        
        # Import here to avoid circular imports
        from agents.summarization_agent import SummaryDepth
        from pathlib import Path
        import re
        
        # Extract file path from request
        words = self.session.user_request.split()
        file_path = None
        for word in words:
            if '.' in word and Path(word).expanduser().exists():
                file_path = str(Path(word).expanduser().resolve())
                break
        
        if not file_path:
            self.ui.print_error("❌ Could not find file path in request")
            self.ui.print_info("💡 Try: 'Summarize path/to/file.txt'")
            self.session.accumulated_results['summarization'] = {
                'success': False,
                'error': 'No file path found'
            }
            return
        
        if not Path(file_path).exists():
            self.ui.print_error(f"❌ File not found: {file_path}")
            self.session.accumulated_results['summarization'] = {
                'success': False,
                'error': 'File not found'
            }
            return
        
        self.ui.print_header("📄 Summarization")
        self.ui.print_info(f"📂 File: {file_path}")
        
        # Get summarization agent
        from agents.summarization_agent import ModularSummarizationAgent
        
        if not hasattr(self, 'summarization_agent'):
            self.ui.print_error("❌ Summarization agent not initialized")
            return
        
        # Run summarization
        self.ui.print_info(f"\n🔄 Summarizing...")
        
        try:
            result = self.summarization_agent.summarize_file(
                file_path=file_path,
                depth=SummaryDepth.STANDARD
            )
        except Exception as e:
            self.ui.print_error(f"\n❌ Summarization failed: {e}")
            import traceback
            traceback.print_exc()
            self.session.accumulated_results['summarization'] = {
                'success': False,
                'error': str(e)
            }
            return
        
        if not result.success:
            self.ui.print_error("\n❌ Summarization failed")
            if result.errors:
                for error in result.errors:
                    self.ui.print_error(f"   • {error}")
            self.session.accumulated_results['summarization'] = {
                'success': False,
                'error': 'Summarization failed'
            }
            return
        
        # Display results
        self.ui.print_header("✅ Summary")
        self.ui.print_info(f"\n{result.summary}\n")
        
        # Display stats
        self.ui.print_info(f"📊 Statistics:")
        self.ui.print_info(f"   • Chunks processed: {result.chunks_processed}")
        self.ui.print_info(f"   • Total tokens: {result.total_tokens:,}")
        self.ui.print_info(f"   • Processing time: {result.processing_time:.1f}s")
        
        # Store in session
        self.session.accumulated_results['summarization'] = {
            'success': True,
            'file_path': file_path,
            'summary': result.summary,
            'stats': {
                'chunks': result.chunks_processed,
                'tokens': result.total_tokens,
                'time': result.processing_time
            }
        }
        
        # Offer Q&A if memory manager available
        if hasattr(self.summarization_agent, 'memory_manager') and \
        self.summarization_agent.memory_manager is not None:
            
            self.ui.print_info("\n💬 Document is ready for questions!")
            
            from rich.prompt import Prompt
            answer = Prompt.ask("Enter Q&A mode? [Y/n]", default="y")
            
            if answer.lower() in ['y', 'yes', '']:
                self._qa_mode(file_path, result.summary[:200])

    def _qa_mode(self, file_path: str, document_context: str = None):
        """Interactive Q&A mode over a document"""
        from rag.qa_engine import QAEngine
        from pathlib import Path
        
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
        self.ui.print_header("💬 Q&A Mode")
        self.ui.print_info(f"\nAsk questions about: {Path(file_path).name}")
        self.ui.print_info("Type 'q' or 'quit' to exit Q&A mode\n")
        
        question_count = 0
        
        while True:
            try:
                # Get question
                from rich.prompt import Prompt
                question = Prompt.ask("\n❓ Your question").strip()
                
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
                    file_path=None,  # Search all, not just this file
                    top_k=5
                )
                
                # Check if we got results
                if result.get('confidence', 0) == 0 or not result.get('answer'):
                    self.ui.print_warning(
                        "\n⚠️  No relevant information found. "
                        "This might mean:\n"
                        "   • Document wasn't stored in memory\n"
                        "   • Question is too specific\n"
                        "   • Try rephrasing your question"
                    )
                    continue
                
                # Display answer
                self.ui.print_header("📝 Answer")  # Removed style parameter
                self.ui.print_info(f"\n{result['answer']}\n")
                
                # Display sources
                if result.get('sources'):
                    self.ui.print_info("📚 Sources:")
                    for i, source in enumerate(result.get('sources', [])[:3], 1):
                        chunk_num = source.get('chunk_number', '?')
                        score = source.get('relevance_score', 0)
                        
                        self.ui.print_info(
                            f"   [{i}] Chunk {chunk_num} - Score: {score:.2f}"
                        )
                
                # Show confidence warning
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
            
    def _run_coding(self, triage_result: Dict):
        """Run coding task with real agent"""
        try:
            from interactive.interactive_coder_agent import create_interactive_coder_agent
            
            coder_agent = create_interactive_coder_agent(
                self.model_manager,
                self.session
            )
            
            result = self.orchestrator.run_coding(coder_agent, spec_content)
            return result
            
        except Exception as e:
            self.logger.error(f"Coding stage failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
        
    def _run_knowledge_query(self, triage_result: Dict):
        """Run knowledge query workflow with continuous Q&A"""
        
        self.logger.info("Running knowledge query workflow")
        
        try:
            # Get user query from triage result
            user_query = triage_result.get('original_request', '')
            
            if not user_query:
                self.ui.print_error("❌ No query text found")
                return {'success': False, 'error': 'No query text'}
            
            # Display stage header
            self.ui.print_header("Stage 2: Knowledge Retrieval")
            
            # ✅ NEW: Q&A Loop
            qa_history = []
            
            while True:
                # First iteration: use original query
                # Subsequent iterations: ask for new query
                if qa_history:
                    self.ui.print_header("Ask Another Question")
                    try:
                        user_query = input("\n💬 Your question (or 'done' to finish): ").strip()
                        
                        if not user_query or user_query.lower() in ['done', 'quit', 'exit', 'q']:
                            self.ui.print_info("\n👋 Exiting Q&A mode...")
                            break
                        
                    except (KeyboardInterrupt, EOFError):
                        self.ui.print_info("\n\n👋 Exiting Q&A mode...")
                        break
                
                # Ask for quality mode (only first time)
                if not qa_history:
                    print("\n🎯 Choose quality mode:")
                    print("  [1] Fast (2s)")
                    print("  [2] Balanced (5s) ⭐ Recommended")
                    print("  [3] Accurate (10s)")
                    print("  [4] Thorough (30s+)")
                    
                    quality_choice = input("Your choice [2]: ").strip() or "2"
                    quality_map = {"1": "fast", "2": "balanced", "3": "accurate", "4": "thorough"}
                    quality = quality_map.get(quality_choice, "balanced")
                    
                    print(f"\n🔍 Using {quality.upper()} mode...\n")
                
                # Run interactive query
                result = self.knowledge_agent.query_interactive(user_query, quality=quality)
                
                # Store in history
                qa_history.append({
                    'query': user_query,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Check if user wants to continue
                if not result.get('continue_qa', False):
                    break
            
            # Store all Q&A history in session
            self.session.accumulated_results['knowledge_query'] = {
                'qa_history': qa_history,
                'total_questions': len(qa_history)
            }
            
            # Return summary
            return {
                'success': True,
                'total_questions': len(qa_history),
                'qa_history': qa_history
            }
            
        except Exception as e:
            self.logger.error(f"Error in knowledge query: {e}", exc_info=True)
            self.ui.print_error(f"Error: {e}")
            return {'success': False, 'error': str(e)}
        
    def load_session(self, session_id: str) -> Optional[InteractiveSession]:
        """Load existing session"""
        session = self.state_store.load_session(session_id)
        
        if session:
            self.session = session
            self.orchestrator = InteractiveOrchestrator(
                session,
                self.state_store,
                self.ui
            )
            self.logger.info(f"Loaded session {session_id}")
        
        return session
    
    def list_sessions(self) -> list:
        """List all sessions"""
        return self.state_store.list_sessions()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.state_store.get_stats()
    
    def _should_save_interaction(self, result: Dict) -> bool:
        """Determine if interaction should be saved"""
        mode = self.config.get('mode', 'interactive')
        
        if mode == 'autonomous':
            # Always save in autonomous mode
            return True
        
        # Interactive mode - check user preference
        save_pref = self.rag_config.get('interactive_mode', {}).get(
            'save_interactions', 'ask'
        )
        
        if save_pref == 'always':
            return True
        elif save_pref == 'never':
            return False
        else:  # 'ask'
            prompt = self.rag_config.get('interactive_mode', {}).get(
                'prompt_message',
                'Save this interaction? [y/n]: '
            )
            response = input(prompt).strip().lower()
        return response in ['y', 'yes']

    def _save_interaction_to_memory(self, session_id: str, result: Dict):
        """Save interaction to memory system"""
        if not self.memory_manager:
            return
        
        # Extract relevant information
        request = result.get('user_request', '')
        outcome = result.get('status', 'unknown')
        
        metadata = {
            'session_id': session_id,
            'mode': result.get('mode', 'interactive'),
            'outcome': outcome,
            'agents_used': result.get('agents', []),
            'duration': result.get('duration', 0)
        }
        
        # Store as episodic memory
        content = f"Request: {request}\nOutcome: {outcome}"
        
        self.memory_manager.store_episodic_memory(
            content=content,
            metadata=metadata,
            is_private=True,
            importance=0.7 if outcome == 'success' else 0.5
        )

    def _handle_view_source(self, console, choice, sources):
        """
        Handle source viewing request
        
        For now, show detailed source info.
        Later, integrate with chunk viewer.
        """
        if choice == 'view all':
            # Show all sources
            for i, source in enumerate(sources, 1):
                self._display_source_detail(console, source, i)
        elif choice.startswith('view '):
            # View specific source
            try:
                source_num = int(choice.split()[1])
                if 1 <= source_num <= len(sources):
                    source = sources[source_num - 1]
                    self._display_source_detail(console, source, source_num)
                    
                    # TODO: Later call chunk viewer here
                    # self._launch_chunk_viewer(source)
                else:
                    console.print(f"[red]Invalid source number. Choose 1-{len(sources)}[/red]")
            except (ValueError, IndexError):
                console.print("[red]Invalid format. Use 'view <number>'[/red]")


    def _display_source_detail(self, console, source, source_num):
        """Display detailed information about a source"""
        console.print(f"\n[bold cyan]═══ Source {source_num} ═══[/bold cyan]")
        console.print(f"\n[bold]Document:[/bold] {source.get('doc_title', 'Unknown')}")
        console.print(f"[bold]Chunk:[/bold] {source.get('chunk_number')}/{source.get('total_chunks')}")
        console.print(f"[bold]Relevance:[/bold] {source.get('similarity', 0):.0%}")
        
        if source.get('chapter_info'):
            ch = source['chapter_info']
            console.print(f"[bold]Chapter:[/bold] {ch['number']} - {ch['title']}")
        
        console.print(f"\n[bold]Full Text:[/bold]")
        console.print(source.get('text_preview', ''), style="white")
        console.print()