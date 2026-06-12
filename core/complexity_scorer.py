#!/usr/bin/env python3
"""
Complexity Scorer - Granularity benchmarks for task assessment

Provides complexity scoring for tasks to help Oracle
decide on agent allocation and execution strategy.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    TRIVIAL = "trivial"      # Single step, instant
    SIMPLE = "simple"        # 1-3 steps, few minutes
    MODERATE = "moderate"    # 3-5 steps, needs planning
    COMPLEX = "complex"      # 5+ steps, needs approval
    ENTERPRISE = "enterprise"  # Large-scale, needs decomposition


class RiskLevel(Enum):
    """Task risk levels"""
    LOW = "low"           # Safe, reversible
    MEDIUM = "medium"     # Some risk, recoverable
    HIGH = "high"         # Significant risk, needs review
    CRITICAL = "critical"  # Irreversible, needs approval


@dataclass
class ComplexityScore:
    """
    Complexity assessment for a task.

    Used by Oracle to determine:
    - Whether to auto-execute or ask for approval
    - Which agents to involve
    - Whether to break down into subtasks
    """
    # Core metrics (0.0 - 1.0)
    total_effort: float = 0.0          # How much overall work
    existing_coverage: float = 0.0      # How much already exists
    new_work_required: float = 0.0      # What's missing

    # Classification
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    risk_level: RiskLevel = RiskLevel.LOW

    # Details
    estimated_steps: int = 1
    agents_needed: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Reasoning
    reasoning: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_effort": self.total_effort,
            "existing_coverage": self.existing_coverage,
            "new_work_required": self.new_work_required,
            "complexity": self.complexity.value,
            "risk_level": self.risk_level.value,
            "estimated_steps": self.estimated_steps,
            "agents_needed": self.agents_needed,
            "dependencies": self.dependencies,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }

    def should_require_approval(self) -> bool:
        """Check if this task should require user approval"""
        return (
            self.complexity in [TaskComplexity.COMPLEX, TaskComplexity.ENTERPRISE] or
            self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
            self.new_work_required > 0.5 or
            len(self.agents_needed) > 2
        )


class ComplexityScorer:
    """
    Scores task complexity based on various factors.

    Factors considered:
    - Input type (file, folder, URL, text)
    - Size of input
    - Number of agents needed
    - Dependencies and prerequisites
    - Risk factors
    """

    def __init__(self):
        self.logger = logging.getLogger("ComplexityScorer")

        # Keywords that indicate complexity
        self.complex_keywords = [
            'refactor', 'migrate', 'redesign', 'architect', 'implement',
            'integrate', 'deploy', 'security', 'authentication', 'database',
            'api', 'system', 'infrastructure', 'performance', 'optimize'
        ]

        self.risky_keywords = [
            'delete', 'remove', 'drop', 'destroy', 'production', 'deploy',
            'rollback', 'restore', 'migrate', 'permission', 'sudo', 'admin'
        ]

        self.simple_keywords = [
            'list', 'show', 'get', 'read', 'check', 'status', 'help',
            'explain', 'describe', 'find', 'search', 'count'
        ]

    def score_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ComplexityScore:
        """
        Score the complexity of a task.

        Args:
            task_description: Description of the task
            context: Optional context (file paths, project info, etc.)

        Returns:
            ComplexityScore with assessment
        """
        context = context or {}
        task_lower = task_description.lower()

        # Start with base score
        score = ComplexityScore()

        # Analyze task description
        self._analyze_keywords(task_lower, score)
        self._analyze_scope(task_description, context, score)
        self._analyze_risk(task_lower, score)
        self._determine_agents(task_lower, context, score)

        # Calculate final complexity
        self._calculate_complexity(score)

        return score

    def _analyze_keywords(self, task_lower: str, score: ComplexityScore):
        """Analyze keywords in task description"""

        # Count complex keywords
        complex_count = sum(1 for kw in self.complex_keywords if kw in task_lower)
        simple_count = sum(1 for kw in self.simple_keywords if kw in task_lower)

        if complex_count >= 2:
            score.total_effort = 0.7
            score.estimated_steps = 5 + complex_count
            score.reasoning = f"Found {complex_count} complex indicators"
        elif complex_count == 1:
            score.total_effort = 0.4
            score.estimated_steps = 3
            score.reasoning = "Single complex indicator found"
        elif simple_count >= 2:
            score.total_effort = 0.1
            score.estimated_steps = 1
            score.reasoning = "Simple task keywords detected"
        else:
            score.total_effort = 0.3
            score.estimated_steps = 2
            score.reasoning = "Moderate complexity (no clear indicators)"

    def _analyze_scope(
        self,
        task_description: str,
        context: Dict[str, Any],
        score: ComplexityScore
    ):
        """Analyze scope of the task"""

        # Check for folder/project analysis
        if 'folder' in task_description.lower() or 'project' in task_description.lower():
            score.total_effort = max(score.total_effort, 0.5)
            score.estimated_steps = max(score.estimated_steps, 4)

        # Check for multi-file operations
        if any(kw in task_description.lower() for kw in ['all files', 'every', 'entire', 'whole']):
            score.total_effort = max(score.total_effort, 0.6)
            score.estimated_steps = max(score.estimated_steps, 5)

        # Check context for existing coverage
        if context.get('existing_files'):
            total_files = context.get('total_files', 1)
            existing = len(context.get('existing_files', []))
            score.existing_coverage = existing / max(total_files, 1)
            score.new_work_required = 1.0 - score.existing_coverage

    def _analyze_risk(self, task_lower: str, score: ComplexityScore):
        """Analyze risk level of the task"""

        risky_count = sum(1 for kw in self.risky_keywords if kw in task_lower)

        if 'production' in task_lower or 'deploy' in task_lower:
            score.risk_level = RiskLevel.CRITICAL
        elif risky_count >= 2:
            score.risk_level = RiskLevel.HIGH
        elif risky_count == 1:
            score.risk_level = RiskLevel.MEDIUM
        else:
            score.risk_level = RiskLevel.LOW

    def _determine_agents(
        self,
        task_lower: str,
        context: Dict[str, Any],
        score: ComplexityScore
    ):
        """Determine which agents are needed"""

        agents = []

        # Code-related tasks
        if any(kw in task_lower for kw in ['code', 'implement', 'write', 'function', 'class']):
            agents.append('coder')

        # Analysis tasks
        if any(kw in task_lower for kw in ['analyze', 'review', 'audit', 'quality']):
            agents.append('coder')

        # Security tasks
        if any(kw in task_lower for kw in ['security', 'vulnerability', 'auth', 'permission']):
            agents.append('security')

        # System tasks
        if any(kw in task_lower for kw in ['run', 'execute', 'install', 'configure', 'deploy']):
            agents.append('operator')

        # Summarization tasks
        if any(kw in task_lower for kw in ['summarize', 'document', 'explain', 'report']):
            agents.append('summarizer')

        # Knowledge tasks
        if any(kw in task_lower for kw in ['find', 'search', 'lookup', 'learn']):
            agents.append('knowledge')

        # Default to operator if nothing else matches
        if not agents:
            agents.append('operator')

        score.agents_needed = agents

    def _calculate_complexity(self, score: ComplexityScore):
        """Calculate final complexity classification"""

        # Use weighted factors
        effort_weight = score.total_effort * 0.4
        steps_weight = min(score.estimated_steps / 10, 1.0) * 0.3
        agents_weight = min(len(score.agents_needed) / 4, 1.0) * 0.2
        risk_weight = {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: 0.3,
            RiskLevel.HIGH: 0.6,
            RiskLevel.CRITICAL: 1.0
        }.get(score.risk_level, 0.0) * 0.1

        complexity_score = effort_weight + steps_weight + agents_weight + risk_weight

        if complexity_score < 0.2:
            score.complexity = TaskComplexity.TRIVIAL
        elif complexity_score < 0.4:
            score.complexity = TaskComplexity.SIMPLE
        elif complexity_score < 0.6:
            score.complexity = TaskComplexity.MODERATE
        elif complexity_score < 0.8:
            score.complexity = TaskComplexity.COMPLEX
        else:
            score.complexity = TaskComplexity.ENTERPRISE

        score.confidence = 0.7  # Base confidence

    def score_folder_analysis(
        self,
        folder_path: str,
        file_count: int = 0,
        total_lines: int = 0
    ) -> ComplexityScore:
        """
        Score complexity of analyzing a folder/project.

        Args:
            folder_path: Path to folder
            file_count: Number of files
            total_lines: Total lines of code

        Returns:
            ComplexityScore for folder analysis
        """
        score = ComplexityScore()

        # Base effort on file count
        if file_count < 10:
            score.total_effort = 0.2
            score.complexity = TaskComplexity.SIMPLE
        elif file_count < 50:
            score.total_effort = 0.4
            score.complexity = TaskComplexity.MODERATE
        elif file_count < 200:
            score.total_effort = 0.6
            score.complexity = TaskComplexity.COMPLEX
        else:
            score.total_effort = 0.8
            score.complexity = TaskComplexity.ENTERPRISE

        # Adjust for lines of code
        if total_lines > 10000:
            score.total_effort = min(score.total_effort + 0.2, 1.0)

        score.agents_needed = ['coder', 'summarizer']
        score.estimated_steps = max(3, file_count // 20)
        score.reasoning = f"Folder analysis: {file_count} files, {total_lines} lines"
        score.confidence = 0.8

        return score
