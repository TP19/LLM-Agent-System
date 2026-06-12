#!/usr/bin/env python3
"""
Task Analyzer - Unified interface for analyzing folders/files/projects

Provides task analysis and agent recommendations for Oracle.

Features:
- Detects input type (folder, file, URL, text)
- Routes to appropriate analyzer (Engineer, Summarizer, etc.)
- Returns complexity assessment with coverage %
- Provides agent recommendations
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse

from .complexity_scorer import ComplexityScorer, ComplexityScore, TaskComplexity

logger = logging.getLogger(__name__)


class InputType(Enum):
    """Type of input being analyzed"""
    FOLDER = "folder"
    FILE = "file"
    URL = "url"
    TEXT = "text"
    CODE_SNIPPET = "code_snippet"
    COMMAND = "command"
    UNKNOWN = "unknown"


@dataclass
class AgentRecommendation:
    """Recommendation for which agent to use"""
    agent_name: str
    purpose: str
    priority: int  # 1 = highest
    estimated_contribution: float  # 0.0 - 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "purpose": self.purpose,
            "priority": self.priority,
            "contribution": self.estimated_contribution
        }


@dataclass
class TaskAnalysis:
    """Complete analysis of a task"""
    input_type: InputType
    input_path: Optional[str] = None
    complexity: Optional[ComplexityScore] = None
    agent_recommendations: List[AgentRecommendation] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_type": self.input_type.value,
            "input_path": self.input_path,
            "complexity": self.complexity.to_dict() if self.complexity else None,
            "agent_recommendations": [r.to_dict() for r in self.agent_recommendations],
            "subtasks": self.subtasks,
            "warnings": self.warnings,
            "metadata": self.metadata
        }

    def get_primary_agent(self) -> Optional[str]:
        """Get the primary recommended agent"""
        if self.agent_recommendations:
            return self.agent_recommendations[0].agent_name
        return None

    def get_agent_chain(self) -> List[str]:
        """Get ordered list of agents to use"""
        return [r.agent_name for r in sorted(
            self.agent_recommendations,
            key=lambda x: x.priority
        )]


class TaskAnalyzer:
    """
    Unified task analyzer for Oracle.

    Analyzes user requests to:
    1. Detect input type
    2. Assess complexity
    3. Recommend agents
    4. Suggest subtask breakdown
    """

    def __init__(self, model_manager=None):
        """
        Initialize TaskAnalyzer.

        Args:
            model_manager: Optional model manager for LLM-based analysis
        """
        self.model_manager = model_manager
        self.complexity_scorer = ComplexityScorer()
        self.logger = logging.getLogger("TaskAnalyzer")

        # File extension to language mapping
        self.code_extensions = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
            '.go': 'go', '.rs': 'rust', '.rb': 'ruby', '.php': 'php',
            '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
            '.sh': 'bash', '.bash': 'bash', '.zsh': 'zsh',
            '.sql': 'sql', '.html': 'html', '.css': 'css',
            '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml',
            '.md': 'markdown', '.txt': 'text'
        }

    def analyze(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskAnalysis:
        """
        Analyze a task and provide recommendations.

        Args:
            task_description: User's task description
            context: Optional context (current project, recent actions, etc.)

        Returns:
            TaskAnalysis with recommendations
        """
        context = context or {}

        # Detect input type and extract path
        input_type, input_path = self._detect_input_type(task_description)

        # Build analysis
        analysis = TaskAnalysis(
            input_type=input_type,
            input_path=input_path
        )

        # Get complexity score
        analysis.complexity = self.complexity_scorer.score_task(
            task_description, context
        )

        # Analyze based on input type
        if input_type == InputType.FOLDER:
            self._analyze_folder_task(task_description, input_path, analysis)
        elif input_type == InputType.FILE:
            self._analyze_file_task(task_description, input_path, analysis)
        elif input_type == InputType.URL:
            self._analyze_url_task(task_description, input_path, analysis)
        elif input_type == InputType.COMMAND:
            self._analyze_command_task(task_description, analysis)
        else:
            self._analyze_text_task(task_description, analysis)

        # Generate agent recommendations
        self._generate_recommendations(task_description, analysis)

        # Generate subtask breakdown if complex
        if analysis.complexity and analysis.complexity.complexity in [
            TaskComplexity.COMPLEX, TaskComplexity.ENTERPRISE
        ]:
            self._generate_subtasks(task_description, analysis)

        return analysis

    def _detect_input_type(self, task: str) -> Tuple[InputType, Optional[str]]:
        """Detect type of input and extract path if applicable"""

        # Check for URLs FIRST (before file paths to avoid false matches)
        url_pattern = r'https?://[^\s]+'
        url_match = re.search(url_pattern, task)
        if url_match:
            return InputType.URL, url_match.group(0)

        # Check for file/folder paths
        path_patterns = [
            r'(/[a-zA-Z0-9_./-]+)',  # Unix absolute path
            r'(~/[a-zA-Z0-9_./-]+)',  # Home path
            r'(\./[a-zA-Z0-9_./-]+)',  # Relative path
        ]

        for pattern in path_patterns:
            match = re.search(pattern, task)
            if match:
                path = match.group(1)
                # Expand home directory
                if path.startswith('~'):
                    path = os.path.expanduser(path)

                if os.path.exists(path):
                    if os.path.isdir(path):
                        return InputType.FOLDER, path
                    else:
                        return InputType.FILE, path

        # Check for file with extension (less aggressive - only if file-like)
        file_pattern = r'\b([a-zA-Z0-9_.-]+\.[a-zA-Z]{1,5})\b'
        file_match = re.search(file_pattern, task)
        if file_match:
            potential_file = file_match.group(1)
            # Avoid matching things like "OAuth2.0" or version numbers
            if not re.match(r'.*\d+\.\d+.*', potential_file):
                return InputType.FILE, potential_file

        # Check for commands
        command_indicators = ['run', 'execute', 'install', 'pip', 'npm', 'git', 'docker']
        if any(cmd in task.lower() for cmd in command_indicators):
            return InputType.COMMAND, None

        # Check for code snippets
        if '```' in task or 'def ' in task or 'class ' in task or 'function ' in task:
            return InputType.CODE_SNIPPET, None

        return InputType.TEXT, None

    def _analyze_folder_task(
        self,
        task: str,
        folder_path: str,
        analysis: TaskAnalysis
    ):
        """Analyze a folder-related task"""

        try:
            folder = Path(folder_path)
            if folder.exists():
                # Count files and analyze structure
                files = list(folder.rglob('*'))
                file_count = len([f for f in files if f.is_file()])
                code_files = [f for f in files if f.suffix in self.code_extensions]

                analysis.metadata['file_count'] = file_count
                analysis.metadata['code_files'] = len(code_files)
                analysis.metadata['folder_path'] = str(folder)

                # Re-score with folder info
                analysis.complexity = self.complexity_scorer.score_folder_analysis(
                    folder_path, file_count, 0
                )

                # Determine if this is a project
                project_indicators = ['setup.py', 'package.json', 'Cargo.toml',
                                     'go.mod', 'requirements.txt', 'pyproject.toml']
                is_project = any((folder / ind).exists() for ind in project_indicators)
                analysis.metadata['is_project'] = is_project

        except Exception as e:
            analysis.warnings.append(f"Could not analyze folder: {e}")

    def _analyze_file_task(
        self,
        task: str,
        file_path: str,
        analysis: TaskAnalysis
    ):
        """Analyze a file-related task"""

        try:
            path = Path(file_path)
            ext = path.suffix.lower()

            analysis.metadata['file_extension'] = ext
            analysis.metadata['language'] = self.code_extensions.get(ext, 'unknown')

            if path.exists():
                analysis.metadata['file_size'] = path.stat().st_size
                analysis.metadata['exists'] = True
            else:
                analysis.metadata['exists'] = False
                analysis.warnings.append(f"File does not exist: {file_path}")

        except Exception as e:
            analysis.warnings.append(f"Could not analyze file: {e}")

    def _analyze_url_task(
        self,
        task: str,
        url: str,
        analysis: TaskAnalysis
    ):
        """Analyze a URL-related task"""

        try:
            parsed = urlparse(url)
            analysis.metadata['domain'] = parsed.netloc
            analysis.metadata['scheme'] = parsed.scheme
            analysis.metadata['path'] = parsed.path

            # GitHub-specific analysis
            if 'github.com' in parsed.netloc:
                parts = parsed.path.strip('/').split('/')
                if len(parts) >= 2:
                    analysis.metadata['github_owner'] = parts[0]
                    analysis.metadata['github_repo'] = parts[1]
                    analysis.metadata['is_github'] = True

        except Exception as e:
            analysis.warnings.append(f"Could not analyze URL: {e}")

    def _analyze_command_task(self, task: str, analysis: TaskAnalysis):
        """Analyze a command execution task"""

        # Extract potential commands
        command_patterns = [
            r'`([^`]+)`',  # Backtick wrapped
            r'"([^"]+)"',  # Double quote wrapped
            r"'([^']+)'",  # Single quote wrapped
        ]

        commands = []
        for pattern in command_patterns:
            commands.extend(re.findall(pattern, task))

        if commands:
            analysis.metadata['detected_commands'] = commands

        # Check for dangerous commands
        dangerous = ['rm -rf', 'sudo rm', 'mkfs', 'dd if=', ':(){:|:&};:']
        for cmd in commands:
            if any(d in cmd for d in dangerous):
                analysis.warnings.append(f"Potentially dangerous command detected: {cmd}")
                analysis.complexity.risk_level = 'critical'

    def _analyze_text_task(self, task: str, analysis: TaskAnalysis):
        """Analyze a text-based task"""

        analysis.metadata['task_length'] = len(task)
        analysis.metadata['word_count'] = len(task.split())

    def _generate_recommendations(self, task: str, analysis: TaskAnalysis):
        """Generate agent recommendations based on analysis"""

        task_lower = task.lower()
        recommendations = []

        # Map task types to agents
        agent_mappings = [
            (['analyze', 'review', 'audit', 'quality', 'code review'],
             'coder', 'Code analysis and quality assessment'),
            (['summarize', 'document', 'explain', 'report', 'describe'],
             'summarizer', 'Documentation and summarization'),
            (['implement', 'write', 'create', 'add', 'build', 'code'],
             'coder', 'Code generation and implementation'),
            (['security', 'vulnerability', 'auth', 'permission', 'safe'],
             'security', 'Security analysis and recommendations'),
            (['run', 'execute', 'install', 'configure', 'deploy', 'command'],
             'operator', 'System command execution'),
            (['find', 'search', 'lookup', 'learn', 'discover', 'navigate'],
             'knowledge', 'Resource discovery and navigation'),
        ]

        priority = 1
        for keywords, agent, purpose in agent_mappings:
            if any(kw in task_lower for kw in keywords):
                # Calculate contribution based on keyword matches
                matches = sum(1 for kw in keywords if kw in task_lower)
                contribution = min(matches / len(keywords), 0.9)

                recommendations.append(AgentRecommendation(
                    agent_name=agent,
                    purpose=purpose,
                    priority=priority,
                    estimated_contribution=contribution
                ))
                priority += 1

        # Ensure at least one recommendation
        if not recommendations:
            if analysis.input_type == InputType.FOLDER:
                recommendations.append(AgentRecommendation(
                    agent_name='coder',
                    purpose='Project analysis',
                    priority=1,
                    estimated_contribution=0.8
                ))
            elif analysis.input_type == InputType.COMMAND:
                recommendations.append(AgentRecommendation(
                    agent_name='operator',
                    purpose='Command execution',
                    priority=1,
                    estimated_contribution=0.9
                ))
            else:
                recommendations.append(AgentRecommendation(
                    agent_name='operator',
                    purpose='General task execution',
                    priority=1,
                    estimated_contribution=0.5
                ))

        analysis.agent_recommendations = recommendations

    def _generate_subtasks(self, task: str, analysis: TaskAnalysis):
        """Generate subtask breakdown for complex tasks"""

        subtasks = []

        # Based on input type
        if analysis.input_type == InputType.FOLDER:
            subtasks.extend([
                "Analyze folder structure and contents",
                "Identify key files and entry points",
                "Assess code quality and patterns",
                "Generate summary report"
            ])
        elif analysis.input_type == InputType.URL:
            if analysis.metadata.get('is_github'):
                subtasks.extend([
                    "Clone or fetch repository",
                    "Analyze repository structure",
                    "Review documentation",
                    "Summarize findings"
                ])
            else:
                subtasks.extend([
                    "Fetch URL content",
                    "Parse and analyze content",
                    "Extract relevant information"
                ])

        # Based on task keywords
        task_lower = task.lower()
        if 'implement' in task_lower or 'build' in task_lower:
            subtasks.extend([
                "Review existing code and patterns",
                "Design implementation approach",
                "Implement core functionality",
                "Add tests and documentation"
            ])
        elif 'security' in task_lower:
            subtasks.extend([
                "Scan for known vulnerabilities",
                "Review authentication/authorization",
                "Check for data exposure risks",
                "Generate security report"
            ])

        # Deduplicate
        analysis.subtasks = list(dict.fromkeys(subtasks))

    def quick_analyze(self, task: str) -> Dict[str, Any]:
        """
        Quick analysis for simple routing decisions.

        Returns minimal info for fast decisions.
        """
        input_type, path = self._detect_input_type(task)

        complexity = self.complexity_scorer.score_task(task)

        return {
            "input_type": input_type.value,
            "path": path,
            "complexity": complexity.complexity.value,
            "risk": complexity.risk_level.value,
            "primary_agent": complexity.agents_needed[0] if complexity.agents_needed else "operator",
            "needs_approval": complexity.should_require_approval()
        }
