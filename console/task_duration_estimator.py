"""
Task Duration Estimator - Smart Task Routing

Estimates task duration to decide between thread execution and tmux background.
"""

import re
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    QUICK = "quick"       # < 10 seconds
    SHORT = "short"       # 10-30 seconds
    MEDIUM = "medium"     # 30 seconds - 2 minutes
    LONG = "long"         # 2-10 minutes
    EXTENDED = "extended" # > 10 minutes


class ExecutionMode(Enum):
    """Execution mode for tasks"""
    INLINE = "inline"     # Execute inline, block until done
    THREAD = "thread"     # Execute in background thread
    TMUX = "tmux"         # Execute in tmux session


@dataclass
class TaskEstimate:
    """Estimated task properties"""
    complexity: TaskComplexity
    estimated_seconds: int
    execution_mode: ExecutionMode
    confidence: float  # 0.0 to 1.0
    reasoning: str


class TaskDurationEstimator:
    """
    Estimates task duration and recommends execution mode

    Uses pattern matching and heuristics to estimate how long
    a task will take, then recommends the appropriate execution mode.

    Rules:
    - QUICK/SHORT tasks (< 30s): Execute in thread
    - MEDIUM tasks (30s-2min): Offer choice or use thread
    - LONG/EXTENDED tasks (> 2min): Use tmux

    Usage:
        estimator = TaskDurationEstimator()

        estimate = estimator.estimate("analyze this binary for vulnerabilities")
        # Returns TaskEstimate with complexity=LONG, execution_mode=TMUX

        estimate = estimator.estimate("what is 2+2?")
        # Returns TaskEstimate with complexity=QUICK, execution_mode=INLINE
    """

    # Duration threshold for tmux (seconds)
    TMUX_THRESHOLD = 30

    # Pattern-based duration estimates (regex pattern -> seconds)
    DURATION_PATTERNS = {
        # Quick queries (< 10s)
        r'\b(what|who|when|where|why|how)\b.*\?$': 5,
        r'\b(explain|describe|tell me about)\b': 8,
        r'\b(list|show|display)\b': 5,
        r'\b(check|verify|confirm)\b': 10,

        # Short tasks (10-30s)
        r'\b(summarize|brief|overview)\b': 20,
        r'\b(search|find|locate)\b': 15,
        r'\b(read|open|view)\b': 10,
        r'\b(status|health|state)\b': 10,

        # Medium tasks (30s-2min)
        r'\b(analyze|examine|inspect)\b': 60,
        r'\b(debug|troubleshoot|diagnose)\b': 90,
        r'\b(compare|diff|contrast)\b': 45,
        r'\b(test|validate|verify)\b': 60,
        r'\b(review|audit|assess)\b': 75,

        # Long tasks (2-10min)
        r'\b(scan|comprehensive|full|complete)\b': 180,
        r'\b(security|vulnerability|exploit)\b': 240,
        r'\b(binary|executable|elf|pe)\b': 300,
        r'\b(pentest|penetration)\b': 600,
        r'\b(reverse engineer|disassemble)\b': 400,

        # Extended tasks (> 10min)
        r'\b(batch|bulk|all files)\b': 900,
        r'\b(entire|whole|everything)\b': 600,
        r'\b(migrate|refactor|rewrite)\b': 1200,
        r'\b(implement|build|create|develop)\b': 600,
    }

    # Agent-specific duration modifiers
    AGENT_MODIFIERS = {
        'oracle': 1.0,      # Planning is quick
        'security': 1.5,    # Security analysis takes time
        'operator': 0.8,    # Command execution is fast
        'coder': 2.0,       # Code generation takes time
        'knowledge': 0.5,   # Retrieval is fast
    }

    # Keywords that suggest background execution
    BACKGROUND_KEYWORDS = [
        'background', 'async', 'later', 'queue',
        'long', 'comprehensive', 'full', 'complete',
        'all', 'batch', 'bulk', 'entire'
    ]

    # Keywords that suggest inline execution
    INLINE_KEYWORDS = [
        'quick', 'fast', 'brief', 'simple',
        'just', 'only', 'single', 'one'
    ]

    def __init__(self, threshold: int = 30):
        """
        Initialize estimator

        Args:
            threshold: Seconds threshold for tmux vs thread (default 30)
        """
        self.threshold = threshold
        self.logger = logging.getLogger("TaskDurationEstimator")

    def estimate(
        self,
        message: str,
        agent: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskEstimate:
        """
        Estimate task duration and recommend execution mode

        Args:
            message: User message/task description
            agent: Target agent (for modifier)
            context: Additional context

        Returns:
            TaskEstimate with complexity, duration, and mode
        """
        message_lower = message.lower()

        # Start with base estimate from pattern matching
        base_seconds, matched_pattern = self._pattern_estimate(message_lower)

        # Apply agent modifier
        if agent and agent.lower() in self.AGENT_MODIFIERS:
            modifier = self.AGENT_MODIFIERS[agent.lower()]
            base_seconds = int(base_seconds * modifier)

        # Check for explicit background/inline hints
        if any(kw in message_lower for kw in self.BACKGROUND_KEYWORDS):
            base_seconds = max(base_seconds, self.threshold + 10)

        if any(kw in message_lower for kw in self.INLINE_KEYWORDS):
            base_seconds = min(base_seconds, self.threshold - 5)

        # Determine complexity
        complexity = self._classify_complexity(base_seconds)

        # Determine execution mode
        execution_mode = self._determine_mode(base_seconds, message_lower)

        # Calculate confidence based on pattern match
        confidence = 0.8 if matched_pattern else 0.5

        # Build reasoning
        reasoning = self._build_reasoning(
            base_seconds, complexity, execution_mode, matched_pattern, agent
        )

        return TaskEstimate(
            complexity=complexity,
            estimated_seconds=base_seconds,
            execution_mode=execution_mode,
            confidence=confidence,
            reasoning=reasoning
        )

    def _pattern_estimate(self, message: str) -> Tuple[int, Optional[str]]:
        """Estimate duration from pattern matching"""
        max_duration = 15  # Default for unmatched
        matched = None

        for pattern, duration in self.DURATION_PATTERNS.items():
            if re.search(pattern, message, re.IGNORECASE):
                if duration > max_duration:
                    max_duration = duration
                    matched = pattern

        return max_duration, matched

    def _classify_complexity(self, seconds: int) -> TaskComplexity:
        """Classify task complexity from duration"""
        if seconds < 10:
            return TaskComplexity.QUICK
        elif seconds < 30:
            return TaskComplexity.SHORT
        elif seconds < 120:
            return TaskComplexity.MEDIUM
        elif seconds < 600:
            return TaskComplexity.LONG
        else:
            return TaskComplexity.EXTENDED

    def _determine_mode(self, seconds: int, message: str) -> ExecutionMode:
        """Determine execution mode from duration and message"""
        # Very short tasks run inline
        if seconds < 10:
            return ExecutionMode.INLINE

        # Short tasks use threads
        if seconds < self.threshold:
            return ExecutionMode.THREAD

        # Longer tasks use tmux
        return ExecutionMode.TMUX

    def _build_reasoning(
        self,
        seconds: int,
        complexity: TaskComplexity,
        mode: ExecutionMode,
        pattern: Optional[str],
        agent: Optional[str]
    ) -> str:
        """Build human-readable reasoning"""
        parts = []

        # Duration explanation
        if seconds < 30:
            parts.append(f"Estimated duration: ~{seconds}s (quick)")
        elif seconds < 120:
            parts.append(f"Estimated duration: ~{seconds}s ({seconds//60}m {seconds%60}s)")
        else:
            parts.append(f"Estimated duration: ~{seconds//60} minutes")

        # Pattern match
        if pattern:
            parts.append(f"Matched pattern: {pattern[:30]}...")

        # Agent modifier
        if agent:
            modifier = self.AGENT_MODIFIERS.get(agent.lower(), 1.0)
            if modifier != 1.0:
                parts.append(f"Agent modifier ({agent}): {modifier}x")

        # Mode recommendation
        mode_reasons = {
            ExecutionMode.INLINE: "Fast enough to run inline",
            ExecutionMode.THREAD: "Will run in background thread",
            ExecutionMode.TMUX: "Long-running, will spawn tmux session"
        }
        parts.append(mode_reasons.get(mode, ""))

        return "; ".join(parts)

    def should_use_tmux(self, message: str, agent: Optional[str] = None) -> bool:
        """Quick check if task should use tmux"""
        estimate = self.estimate(message, agent)
        return estimate.execution_mode == ExecutionMode.TMUX

    def get_mode(self, message: str, agent: Optional[str] = None) -> ExecutionMode:
        """Get recommended execution mode"""
        return self.estimate(message, agent).execution_mode
