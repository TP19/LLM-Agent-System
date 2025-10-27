from typing import Dict, List, Optional, Type, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class Agent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the agent's functionality."""
        pass

class AgentRegistry:
    """Registry for managing and discovering agents."""
    
    def __init__(self):
        self._agents: Dict[str, Type[Agent]] = {}
        self._instances: Dict[str, Agent] = {}
    
    def register_agent(self, agent_class: Type[Agent]) -> None:
        """Register an agent class with the registry.
        
        Args:
            agent_class: The agent class to register
            
        Raises:
            ValueError: If agent_class is not a subclass of Agent
        """
        if not issubclass(agent_class, Agent):
            raise ValueError(f"{agent_class.__name__} must inherit from Agent")
        
        self._agents[agent_class.__name__] = agent_class
        logger.info(f"Registered agent: {agent_class.__name__}")
    
    def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent class from the registry.
        
        Args:
            agent_name: Name of the agent to unregister
            
        Returns:
            bool: True if agent was unregistered, False if not found
        """
        if agent_name in self._agents:
            del self._agents[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")
            return True
        return False
    
    def get_agent(self, agent_name: str) -> Optional[Type[Agent]]:
        """Get an agent class by name.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            Agent class or None if not found
        """
        return self._agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
    
    def create_instance(self, agent_name: str, **kwargs) -> Agent:
        """Create an instance of a registered agent.
        
        Args:
            agent_name: Name of the agent to instantiate
            **kwargs: Arguments to pass to the agent constructor
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If agent is not registered
        """
        agent_class = self.get_agent(agent_name)
        if not agent_class:
            raise ValueError(f"Agent '{agent_name}' not registered")
        
        instance = agent_class(**kwargs)
        self._instances[agent_name] = instance
        return instance
    
    def get_instance(self, agent_name: str) -> Optional[Agent]:
        """Get an existing instance of an agent.
        
        Args:
            agent_name: Name of the agent instance to retrieve
            
        Returns:
            Agent instance or None if not found
        """
        return self._instances.get(agent_name)
    
    def execute_agent(self, agent_name: str, **kwargs) -> Any:
        """Execute a registered agent.
        
        Args:
            agent_name: Name of the agent to execute
            **kwargs: Arguments to pass to the agent's execute method
            
        Returns:
            Result of the agent's execute method
            
        Raises:
            ValueError: If agent is not registered
        """
        instance = self.get_instance(agent_name)
        if not instance:
            instance = self.create_instance(agent_name)
        
        return instance.execute(**kwargs)
