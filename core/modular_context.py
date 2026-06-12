"""
Modular Context Provider for LLM Self-Maintenance

This module enables LLMs to focus on specific modules rather than the entire codebase.
It integrates with doc-manager to retrieve focused, well-structured context for targeted
development and maintenance tasks.

Architecture:
    ┌─────────────────────┐
    │   LLM Agent         │
    │   (Oracle/Healer)   │
    └────────┬────────────┘
             │ get_module_context("console")
             ▼
    ┌─────────────────────┐
    │ ModularContextProvider │
    └────────┬────────────┘
             │ queries
             ▼
    ┌─────────────────────┐
    │   doc-manager KB    │
    │  (Components, etc)  │
    └─────────────────────┘

Usage:
    from core.modular_context import ModularContextProvider

    provider = ModularContextProvider("LLM-Agent-System")

    # Get focused context for a specific module
    context = provider.get_module_context("console")

    # Get context for related modules (dependencies)
    context = provider.get_module_context("console", include_related=True)

    # Get context for a specific task
    context = provider.get_task_context("fix bug in session handling")
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Add doc-manager to path if available
DOC_MANAGER_PATH = Path.home() / "projects" / "doc-manager"
if DOC_MANAGER_PATH.exists() and str(DOC_MANAGER_PATH) not in sys.path:
    sys.path.insert(0, str(DOC_MANAGER_PATH))

logger = logging.getLogger(__name__)


@dataclass
class ModuleContext:
    """Focused context for a single module"""
    name: str
    type: str
    description: str
    tech_stack: str
    code_location: str
    dependencies: List[str]
    algorithms: List[Dict]
    integrations: List[Dict]
    recent_issues: List[Dict]
    decisions: List[Dict]
    tests: List[Dict]


class ModularContextProvider:
    """
    Provides focused, modular context for LLM agents.

    Instead of giving LLMs the entire codebase context, this provider
    retrieves only the relevant module information from doc-manager,
    allowing LLMs to work efficiently on specific areas.
    """

    def __init__(self, project_name: str):
        """
        Initialize the context provider.

        Args:
            project_name: Name of the project in doc-manager
        """
        self.project_name = project_name
        self.kb = None
        self.semantic_engine = None
        self._initialize_backends()

    def _initialize_backends(self):
        """Initialize doc-manager backends"""
        try:
            from docman.knowledge_base import KnowledgeBase
            self.kb = KnowledgeBase()
            logger.info(f"✅ doc-manager KnowledgeBase initialized for {self.project_name}")

            # Try to initialize semantic search
            try:
                from docman.semantic.semantic_query import SemanticQueryEngine
                self.semantic_engine = SemanticQueryEngine(self.kb, self.project_name)
                if self.semantic_engine.is_available():
                    logger.info("✅ Semantic search available")
                else:
                    logger.info("ℹ️  Semantic search not available")
                    self.semantic_engine = None
            except ImportError:
                logger.info("ℹ️  Semantic search module not available")

        except ImportError as e:
            logger.warning(f"⚠️  doc-manager not available: {e}")
            logger.warning("   Install with: pip install doc-manager or check path")

    def get_module_context(
        self,
        module_name: str,
        include_related: bool = False,
        depth: int = 1
    ) -> Optional[ModuleContext]:
        """
        Get focused context for a specific module.

        Args:
            module_name: Name of the module/component
            include_related: Whether to include related modules (dependencies)
            depth: How deep to follow dependencies (1 = direct only)

        Returns:
            ModuleContext with all relevant information, or None if not found
        """
        if not self.kb:
            logger.warning("KnowledgeBase not available")
            return None

        # Get component details
        component = self.kb.get_component(self.project_name, module_name)
        if 'error' in component:
            logger.warning(f"Module '{module_name}' not found in doc-manager")
            return None

        # Get related decisions
        decisions = self._get_module_decisions(component['id'])

        # Get recent issues
        issues = self._get_module_issues(component['id'])

        # Build context
        context = ModuleContext(
            name=component['name'],
            type=component.get('type', 'unknown'),
            description=component.get('description', ''),
            tech_stack=component.get('tech_stack', ''),
            code_location=component.get('code_location', ''),
            dependencies=component.get('dependencies', []),
            algorithms=component.get('algorithms', []),
            integrations=component.get('integrations', []),
            recent_issues=issues,
            decisions=decisions,
            tests=component.get('tests', [])
        )

        return context

    def get_task_context(
        self,
        task_description: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Get context relevant to a specific task using semantic search.

        Args:
            task_description: Natural language description of the task
            top_k: Number of relevant items to retrieve

        Returns:
            Dict with relevant components, decisions, and issues
        """
        result = {
            'task': task_description,
            'relevant_components': [],
            'relevant_decisions': [],
            'relevant_issues': [],
            'suggested_modules': []
        }

        if self.semantic_engine and self.semantic_engine.is_available():
            # Use semantic search to find relevant context
            search_results = self.semantic_engine.semantic_search(
                query=task_description,
                top_k=top_k
            )

            for item in search_results:
                item_type = item.get('type', 'unknown')
                if item_type == 'component':
                    result['relevant_components'].append(item)
                    result['suggested_modules'].append(
                        item.get('metadata', {}).get('component_name', 'unknown')
                    )
                elif item_type == 'decision':
                    result['relevant_decisions'].append(item)
                elif item_type == 'issue':
                    result['relevant_issues'].append(item)

            # Deduplicate suggested modules
            result['suggested_modules'] = list(set(result['suggested_modules']))
        else:
            # Fallback: keyword-based search
            result['note'] = 'Semantic search unavailable, using keyword matching'

            if self.kb:
                components = self.kb.list_components(self.project_name)
                task_lower = task_description.lower()

                for comp in components:
                    # Simple keyword matching
                    comp_text = f"{comp['name']} {comp.get('description', '')}".lower()
                    if any(word in comp_text for word in task_lower.split()):
                        result['relevant_components'].append(comp)
                        result['suggested_modules'].append(comp['name'])

        return result

    def get_architecture_overview(self) -> Dict[str, Any]:
        """
        Get high-level architecture overview without diving into specifics.

        Useful for LLMs to understand the overall structure before
        drilling into specific modules.

        Returns:
            Dict with architecture summary
        """
        if not self.kb:
            return {'error': 'KnowledgeBase not available'}

        components = self.kb.list_components(self.project_name)
        integration_map = self.kb.get_integration_map(self.project_name)

        # Group components by type
        by_type = {}
        for comp in components:
            comp_type = comp.get('type', 'other')
            if comp_type not in by_type:
                by_type[comp_type] = []
            by_type[comp_type].append({
                'name': comp['name'],
                'description': comp.get('description', ''),
                'status': comp.get('status', 'unknown')
            })

        return {
            'project': self.project_name,
            'total_components': len(components),
            'components_by_type': by_type,
            'integration_count': len(integration_map.get('integrations', [])),
            'key_integrations': integration_map.get('integrations', [])[:5]
        }

    def format_context_for_llm(
        self,
        context: ModuleContext,
        verbosity: str = "medium"
    ) -> str:
        """
        Format module context into a string suitable for LLM consumption.

        Args:
            context: ModuleContext object
            verbosity: "brief", "medium", or "detailed"

        Returns:
            Formatted string for LLM context window
        """
        lines = [
            f"# Module: {context.name}",
            f"Type: {context.type}",
            f"Description: {context.description}",
        ]

        if context.code_location:
            lines.append(f"Location: {context.code_location}")

        if context.tech_stack:
            lines.append(f"Tech Stack: {context.tech_stack}")

        if context.dependencies:
            lines.append(f"\n## Dependencies")
            for dep in context.dependencies:
                lines.append(f"  - {dep}")

        if verbosity in ["medium", "detailed"]:
            if context.integrations:
                lines.append(f"\n## Integrations")
                for integ in context.integrations:
                    lines.append(f"  - {integ.get('connected_component', 'unknown')}: {integ.get('method', 'N/A')}")

            if context.recent_issues:
                lines.append(f"\n## Recent Issues")
                for issue in context.recent_issues[:3]:
                    status = "Resolved" if issue.get('resolved_date') else "Open"
                    lines.append(f"  - [{status}] {issue.get('title', 'N/A')}")

            if context.decisions:
                lines.append(f"\n## Key Decisions")
                for dec in context.decisions[:3]:
                    lines.append(f"  - {dec.get('question', 'N/A')}: {dec.get('decision', 'N/A')}")

        if verbosity == "detailed":
            if context.algorithms:
                lines.append(f"\n## Algorithms")
                for algo in context.algorithms:
                    lines.append(f"  - {algo.get('name', 'N/A')}: {algo.get('description', 'N/A')}")

            if context.tests:
                lines.append(f"\n## Tests")
                for test in context.tests:
                    lines.append(f"  - {test.get('test_file', 'N/A')}: {test.get('what_it_tests', 'N/A')}")

        return "\n".join(lines)

    def _get_module_decisions(self, component_id: int, limit: int = 5) -> List[Dict]:
        """Get decisions related to a component"""
        if not self.kb:
            return []

        cursor = self.kb.conn.cursor()
        cursor.execute('''
            SELECT * FROM decisions
            WHERE component_id = ?
            ORDER BY decision_date DESC
            LIMIT ?
        ''', (component_id, limit))

        return [dict(row) for row in cursor.fetchall()]

    def _get_module_issues(self, component_id: int, limit: int = 5) -> List[Dict]:
        """Get issues related to a component"""
        if not self.kb:
            return []

        cursor = self.kb.conn.cursor()
        cursor.execute('''
            SELECT * FROM issues
            WHERE component_id = ?
            ORDER BY issue_date DESC
            LIMIT ?
        ''', (component_id, limit))

        return [dict(row) for row in cursor.fetchall()]

    def list_modules(self) -> List[str]:
        """List all available modules in the project"""
        if not self.kb:
            return []

        components = self.kb.list_components(self.project_name)
        return [c['name'] for c in components]

    def is_available(self) -> bool:
        """Check if the context provider is properly initialized"""
        return self.kb is not None


# Convenience function for quick access
def get_llm_context(project: str, module: str, verbosity: str = "medium") -> str:
    """
    Quick way to get formatted context for an LLM.

    Args:
        project: Project name in doc-manager
        module: Module/component name
        verbosity: "brief", "medium", or "detailed"

    Returns:
        Formatted context string

    Example:
        context = get_llm_context("LLM-Agent-System", "console", "detailed")
        prompt = f"Given this module context:\n{context}\n\nFix the bug in..."
    """
    provider = ModularContextProvider(project)
    module_context = provider.get_module_context(module)

    if module_context:
        return provider.format_context_for_llm(module_context, verbosity)
    else:
        return f"Module '{module}' not found in project '{project}'"
