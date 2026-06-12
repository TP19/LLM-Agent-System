#!/usr/bin/env python3
"""
Agent Chat - Standalone Agent Conversation Script

This script is designed to run in a tmux pane for direct agent interaction.
It maintains conversation context and can communicate with the main Console.

Usage:
    python -m console.agent_chat --agent security --session abc123
    python -m console.agent_chat --agent oracle --session abc123 --context '{"task": "analyze"}'
"""

import sys
import argparse
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rich.console import Console as RichConsole
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

from console.agent_context import AgentContextManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('agent_chat.log')]
)
logger = logging.getLogger(__name__)


class AgentChat:
    """
    Standalone agent chat interface for tmux panes

    Provides:
    - Direct conversation with a specific agent
    - Context preservation
    - Rich terminal UI
    - Communication with main Console via shared context
    """

    # Agent color mapping
    AGENT_COLORS = {
        'oracle': 'magenta',
        'security': 'red',
        'operator': 'green',
        'coder': 'blue',
        'knowledge': 'yellow',
        'triage': 'white',
        'summarizer': 'bright_cyan',
    }

    def __init__(
        self,
        agent_name: str,
        session_id: str,
        initial_context: dict = None
    ):
        self.agent_name = agent_name.lower()
        self.session_id = session_id
        self.console = RichConsole()

        # Initialize context manager
        self._agent_ctx = AgentContextManager(session_id)
        self.agent_context = self._agent_ctx.get_agent(agent_name)

        # Add initial context if provided
        if initial_context:
            self.agent_context.context.update(initial_context)

        # Initialize agent
        self.agent = None
        self.model_manager = None
        self._init_agent()

        self.running = True

    def _init_agent(self):
        """Initialize the agent and model manager"""
        try:
            import yaml
            from core.model_manager import LazyModelManager

            # Load config
            config_path = project_root / 'config' / 'models.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.model_manager = LazyModelManager(
                    models_config=config.get('models', {})
                )

            # Initialize RAG/memory_manager for agents that need it
            self.memory_manager = None
            if self.agent_name in ['knowledge', 'summarizer']:
                self._init_memory_manager()

            # Load the specific agent
            self.agent = self._load_agent(self.agent_name)

            if self.agent:
                logger.info(f"Initialized {self.agent_name} agent")
            else:
                logger.warning(f"Could not load {self.agent_name} agent")

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            self.console.print(f"[yellow]Warning: Agent initialization failed: {e}[/yellow]")

    def _init_memory_manager(self):
        """Initialize RAG memory manager for knowledge/summarizer agents"""
        try:
            import yaml
            from rag.memory.memory_manager import MemoryManager
            from rag.vector_stores.dual_store_manager import DualStoreManager
            from rag.embedding.embedding_engine import EmbeddingEngine

            db_base = Path.home() / ".llm_engine" / "vector_db"

            # Initialize embedding engine (lazy_load defers HF download until first use)
            embedder = EmbeddingEngine()

            # Reranker is opt-in via rag_config.yaml — default is off so first run
            # doesn't trigger a ~1.2GB HuggingFace download of the reranker model.
            reranker = None
            try:
                cfg_path = Path(__file__).resolve().parent.parent / "config" / "rag_config.yaml"
                rag_cfg = yaml.safe_load(cfg_path.read_text())
                if rag_cfg.get("reranking", {}).get("enable", False):
                    from rag.retrieval.reranker import Reranker
                    reranker = Reranker()
            except Exception as _re:
                logger.warning(f"reranker init skipped: {_re}")

            # Initialize dual store manager
            store_manager = DualStoreManager(
                private_db_path=str(db_base / "private"),
                public_db_path=str(db_base / "public")
            )

            self.memory_manager = MemoryManager(
                store_manager=store_manager,
                embedding_engine=embedder,
                reranker=reranker
            )

            logger.info("✅ RAG memory manager initialized for agent chat")
            self.console.print("[dim]RAG system initialized[/dim]")

        except Exception as e:
            logger.warning(f"⚠️ RAG system not available: {e}")
            self.console.print(f"[yellow]RAG not available: {e}[/yellow]")
            self.memory_manager = None
            # Stash the actual error so KnowledgeAgent can surface it on /agent knowledge
            # instead of falling back to the generic "no documents indexed" message.
            self._rag_init_error = str(e)

    def _load_agent(self, agent_name: str):
        """Load a specific agent"""
        if not self.model_manager:
            return None

        try:
            if agent_name == 'oracle':
                from agents.oracle_agent import OracleAgent
                return OracleAgent(self.model_manager)

            elif agent_name == 'security':
                from agents.security_agent import ModularSecurityAgent
                return ModularSecurityAgent(self.model_manager)

            elif agent_name == 'operator':
                from agents.operator_agent import ModularOperatorAgent
                return ModularOperatorAgent(self.model_manager)

            elif agent_name == 'coder':
                from agents.coder_agent import ModularCoderAgent
                return ModularCoderAgent(self.model_manager)

            elif agent_name == 'knowledge':
                from agents.knowledge_agent import KnowledgeAgent
                return KnowledgeAgent(self.model_manager, memory_manager=self.memory_manager)

            elif agent_name == 'triage':
                from agents.triage_agent import ModularTriageAgent
                return ModularTriageAgent(self.model_manager)

            elif agent_name == 'summarizer':
                from agents.enhanced_summarization import EnhancedSummarizationAgent
                return EnhancedSummarizationAgent(self.model_manager, memory_manager=self.memory_manager)

            else:
                logger.warning(f"Unknown agent: {agent_name}")
                return None

        except ImportError as e:
            logger.error(f"Could not import {agent_name} agent: {e}")
            return None

    def display_welcome(self):
        """Display welcome banner"""
        color = self.AGENT_COLORS.get(self.agent_name, 'white')
        agent_title = self.agent_name.title()

        banner = f"""
[bold {color}]{'='*50}
   {agent_title} Agent
   Session: {self.session_id[:8]}
{'='*50}[/bold {color}]

Type your message to chat with {agent_title}.
Commands: /clear, /context, /history, /back

"""
        self.console.print(banner)

    def get_prompt(self) -> str:
        """Get user input with styled prompt"""
        color = self.AGENT_COLORS.get(self.agent_name, 'white')

        try:
            return Prompt.ask(f"[bold {color}]{self.agent_name.title()}>[/bold {color}]")
        except (EOFError, KeyboardInterrupt):
            return "/back"

    def print_response(self, response: str):
        """Print agent response"""
        color = self.AGENT_COLORS.get(self.agent_name, 'white')
        self.console.print()
        self.console.print(f"[{color}]{self.agent_name.title()}:[/{color}] {response}")
        self.console.print()

    def handle_command(self, command: str) -> bool:
        """
        Handle slash commands

        Returns:
            True if should continue, False to exit
        """
        cmd = command.lower().strip()

        if cmd in ['/back', '/exit', '/quit']:
            self.console.print("[dim]Returning to Console...[/dim]")
            return False

        elif cmd == '/clear':
            self._agent_ctx.clear_agent(self.agent_name)
            self.console.print("[dim]Conversation cleared[/dim]")

        elif cmd == '/context':
            ctx = self.agent_context.context
            shared = self._agent_ctx.get_all_shared()
            self.console.print(Panel(
                f"Agent context: {json.dumps(ctx, indent=2)}\n\n"
                f"Shared context: {json.dumps(shared, indent=2)}",
                title="Context"
            ))

        elif cmd == '/history':
            history = self._agent_ctx.get_history(self.agent_name, limit=10, format="text")
            self.console.print(Panel(history or "No history", title="Conversation History"))

        elif cmd == '/help':
            self.console.print("""
[bold]Commands:[/bold]
  /clear   - Clear conversation history
  /context - Show current context
  /history - Show conversation history
  /back    - Return to Console
  /help    - Show this help
""")

        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")

        return True

    def chat(self, message: str) -> str:
        """
        Send message to agent and get response

        Args:
            message: User message

        Returns:
            Agent response
        """
        # Add user message to context
        self._agent_ctx.add_message(self.agent_name, "user", message)

        # Get conversation history
        history = self._agent_ctx.get_history(self.agent_name, limit=10)

        # Generate response
        if self.agent:
            try:
                if hasattr(self.agent, 'chat'):
                    response = self.agent.chat(message, history)
                elif hasattr(self.agent, 'process'):
                    result = self.agent.process(message)
                    response = str(result) if result else "No response"
                else:
                    response = f"[{self.agent_name} has no chat method]"

            except Exception as e:
                logger.error(f"Agent error: {e}")
                response = f"Error: {e}"
        else:
            response = f"[{self.agent_name} agent not available - running in demo mode]"

        # If Oracle returned its triage-trigger JSON, render it as a clean prompt,
        # then re-call Oracle with triage_approved=True if the user says yes.
        # Without this, the raw {"triage_trigger": true, ...} dict leaks to the user.
        response = self._maybe_handle_oracle_triage(response, message, history)

        # Add response to context
        self._agent_ctx.add_message(self.agent_name, "assistant", response)

        return response

    def _maybe_handle_oracle_triage(self, response: str, original_message: str, history) -> str:
        """If the agent's response is a triage-trigger JSON, prompt for approval and
        re-run with triage_approved=True; otherwise return the response unchanged."""
        if self.agent_name != "oracle" or not isinstance(response, str):
            return response
        stripped = response.strip()
        if not (stripped.startswith("{") and "triage_trigger" in stripped):
            return response
        try:
            import json
            data = json.loads(stripped)
        except Exception:
            return response
        if not data.get("triage_trigger"):
            return response

        complexity = data.get("complexity", "moderate")
        reason = data.get("reason", "task detected")
        context_note = data.get("context_note", "")

        # Print the friendly prompt and ask for approval.
        self.console.print(
            f"\n[yellow]This looks like a {complexity} task ({reason}). "
            f"Triage and dispatch it?[/yellow]"
        )
        if context_note:
            self.console.print(f"[dim]{context_note}[/dim]")
        try:
            ans = input("Triage now? [y/N]: ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            return ("Skipped triage. Use /agent <name> to talk to a specialist directly, "
                    "or rephrase the request and try again.")

        # User approved — call Oracle again with triage_approved=True
        try:
            return self.agent.chat(original_message, history, triage_approved=True)
        except TypeError:
            # Older signature without triage_approved
            return self.agent.chat(original_message, history)
        except Exception as e:
            logger.error(f"Triage execution failed: {e}")
            return f"Triage failed: {e}"

    def run(self):
        """Main chat loop"""
        self.display_welcome()

        while self.running:
            try:
                user_input = self.get_prompt()

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue

                # Chat with agent
                response = self.chat(user_input)
                self.print_response(response)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Use /back to return to Console[/dim]")
                continue

        # Cleanup
        self._agent_ctx.close()
        self.console.print("\n[dim]Agent session ended[/dim]")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Agent Chat - Direct agent conversation'
    )

    parser.add_argument(
        '--agent', '-a',
        type=str,
        required=True,
        help='Agent name (oracle, security, operator, etc.)'
    )

    parser.add_argument(
        '--session', '-s',
        type=str,
        required=True,
        help='Console session ID'
    )

    parser.add_argument(
        '--context', '-c',
        type=str,
        default='{}',
        help='Initial context as JSON string'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Parse initial context
    try:
        initial_context = json.loads(args.context)
    except json.JSONDecodeError:
        initial_context = {}

    # Create and run chat
    chat = AgentChat(
        agent_name=args.agent,
        session_id=args.session,
        initial_context=initial_context
    )

    chat.run()


if __name__ == "__main__":
    main()
