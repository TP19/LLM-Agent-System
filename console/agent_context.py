"""
Agent Context - Per-Agent Conversation State Management

Manages conversation context for each agent, allowing parallel
agent chats while preserving history.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class AgentConversation:
    """Conversation state for a single agent"""
    agent_name: str
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    pane_id: Optional[str] = None
    is_active: bool = False

    def add_message(self, role: str, content: str):
        """Add a message to the conversation"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_active = datetime.now().isoformat()

    def get_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history"""
        return self.messages[-limit:]

    def clear(self):
        """Clear conversation history"""
        self.messages = []


class AgentContextManager:
    """
    Manages conversation contexts for multiple agents

    Features:
    - Per-agent conversation history
    - Context preservation across pane switches
    - Shared context between agents (optional)
    - Persistence to disk

    Usage:
        ctx = AgentContextManager(session_id="abc123")

        # Get or create agent context
        security_ctx = ctx.get_agent("security")
        security_ctx.add_message("user", "audit file permissions")
        security_ctx.add_message("assistant", "I'll audit the permissions...")

        # Get conversation history for LLM
        history = ctx.get_history("security", limit=5)

        # Share context between agents
        ctx.share_context("security", "oracle", {"audit_path": "/etc"})
    """

    def __init__(
        self,
        session_id: str,
        storage_path: str = "~/.llm_engine/console/agent_contexts"
    ):
        self.session_id = session_id
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Active agent conversations
        self._agents: Dict[str, AgentConversation] = {}

        # Shared context available to all agents
        self._shared_context: Dict[str, Any] = {}

        # Load existing contexts for this session
        self._load_session()

        logger.info(f"AgentContextManager initialized for session {session_id}")

    def _load_session(self):
        """Load existing agent contexts from disk"""
        session_file = self.storage_path / f"{self.session_id}.json"

        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)

                self._shared_context = data.get('shared_context', {})

                for agent_data in data.get('agents', []):
                    conv = AgentConversation(**agent_data)
                    self._agents[conv.agent_name] = conv

                logger.debug(f"Loaded {len(self._agents)} agent contexts")

            except Exception as e:
                logger.error(f"Failed to load session: {e}")

    def _save_session(self):
        """Save agent contexts to disk"""
        session_file = self.storage_path / f"{self.session_id}.json"

        try:
            data = {
                'session_id': self.session_id,
                'shared_context': self._shared_context,
                'agents': [asdict(conv) for conv in self._agents.values()]
            }

            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self._agents)} agent contexts")

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def get_agent(self, agent_name: str) -> AgentConversation:
        """
        Get or create conversation for an agent

        Args:
            agent_name: Name of the agent

        Returns:
            AgentConversation instance
        """
        agent_name = agent_name.lower()

        if agent_name not in self._agents:
            self._agents[agent_name] = AgentConversation(
                agent_name=agent_name,
                session_id=self.session_id
            )
            logger.debug(f"Created new context for {agent_name}")

        return self._agents[agent_name]

    def get_history(
        self,
        agent_name: str,
        limit: int = 10,
        format: str = "list"
    ) -> Any:
        """
        Get conversation history for an agent

        Args:
            agent_name: Name of the agent
            limit: Maximum messages to return
            format: "list" for list of dicts, "text" for formatted string

        Returns:
            Conversation history in requested format
        """
        conv = self.get_agent(agent_name)
        messages = conv.get_history(limit)

        if format == "text":
            lines = []
            for msg in messages:
                role = msg['role'].capitalize()
                content = msg['content']
                lines.append(f"{role}: {content}")
            return "\n".join(lines)

        return messages

    def add_message(self, agent_name: str, role: str, content: str):
        """Add message to agent's conversation"""
        conv = self.get_agent(agent_name)
        conv.add_message(role, content)
        self._save_session()

    def set_pane(self, agent_name: str, pane_id: str):
        """Set the tmux pane ID for an agent"""
        conv = self.get_agent(agent_name)
        conv.pane_id = pane_id
        conv.is_active = True
        self._save_session()

    def clear_pane(self, agent_name: str):
        """Clear the pane association for an agent"""
        conv = self.get_agent(agent_name)
        conv.pane_id = None
        conv.is_active = False
        self._save_session()

    def get_active_agents(self) -> List[str]:
        """Get list of agents with active panes"""
        return [
            name for name, conv in self._agents.items()
            if conv.is_active
        ]

    def share_context(
        self,
        from_agent: str,
        to_agent: str,
        context: Dict[str, Any]
    ):
        """
        Share context from one agent to another

        Args:
            from_agent: Source agent
            to_agent: Target agent
            context: Context to share
        """
        target = self.get_agent(to_agent)
        target.context.update({
            f"from_{from_agent}": context
        })
        self._save_session()

        logger.debug(f"Shared context from {from_agent} to {to_agent}")

    def set_shared(self, key: str, value: Any):
        """Set a value in shared context (available to all agents)"""
        self._shared_context[key] = value
        self._save_session()

    def get_shared(self, key: str, default: Any = None) -> Any:
        """Get a value from shared context"""
        return self._shared_context.get(key, default)

    def get_all_shared(self) -> Dict[str, Any]:
        """Get all shared context"""
        return self._shared_context.copy()

    def clear_agent(self, agent_name: str):
        """Clear an agent's conversation history"""
        conv = self.get_agent(agent_name)
        conv.clear()
        self._save_session()

    def close(self):
        """Save and close the context manager"""
        self._save_session()
        logger.debug(f"Closed context manager for session {self.session_id}")
