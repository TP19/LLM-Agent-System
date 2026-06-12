"""
LLM-Agent-System Agent implementations

Release version - Core agents for RAG-powered multi-agent system.
"""

__version__ = "0.1.0"

# Import only core models
from agents.models import (
    LogAnalysis,
    Error,
    Warning,
    ErrorCase,
    CriteriaResult,
    CriteriaCheck,
    ErrorSeverity,
    ErrorType,
    LogStatus
)

# Agents are imported lazily on demand:
#   from agents.oracle_agent import OracleAgent
#   from agents.operator_agent import OperatorAgent
#   from agents.coder_agent import ModularCoderAgent
#   from agents.knowledge_agent import KnowledgeAgent
#   from agents.triage_agent import TriageAgent
#   from agents.workflow_executor import OracleTaskRunner
#   from agents.enhanced_summarization import EnhancedSummarizationAgent

__all__ = [
    'LogAnalysis',
    'Error',
    'Warning',
    'ErrorCase',
    'CriteriaResult',
    'CriteriaCheck',
    'ErrorSeverity',
    'ErrorType',
    'LogStatus'
]
