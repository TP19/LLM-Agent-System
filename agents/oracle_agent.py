"""
Oracle Agent - Strategic Planning & Decision-Making
===================================================

The Oracle Agent handles high-level strategic planning, task decomposition,
and decision-making for complex multi-agent workflows.

Capabilities:
- Multi-step task decomposition
- Resource allocation planning
- Risk assessment
- Alternative solution generation
- Timeline estimation
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import re
import uuid as uuid_lib

from core.base_agent import BaseAgent

# Lazy imports for validation agents and task analyzer (avoid circular imports)
ValidatorAgent = None
IntelligentFeedbackAgent = None
TaskAnalyzer = None


def _get_validator_agent(model_manager):
    """Lazy load ValidatorAgent. Returns None if the module isn't shipped (v0.2)."""
    global ValidatorAgent
    if ValidatorAgent is None:
        try:
            from agents.validator_agent import ValidatorAgent as VA
            ValidatorAgent = VA
        except ImportError:
            return None
    return ValidatorAgent(model_manager)


def _get_feedback_agent(model_manager):
    """Lazy load IntelligentFeedbackAgent. Returns None if not shipped (v0.2)."""
    global IntelligentFeedbackAgent
    if IntelligentFeedbackAgent is None:
        try:
            from agents.intelligent_feedback_agent import IntelligentFeedbackAgent as IFA
            IntelligentFeedbackAgent = IFA
        except ImportError:
            return None
    return IntelligentFeedbackAgent(model_manager)


def _get_task_analyzer():
    """Lazy load TaskAnalyzer. Returns None if not available."""
    global TaskAnalyzer
    if TaskAnalyzer is None:
        try:
            from core.task_analyzer import TaskAnalyzer as TA
            TaskAnalyzer = TA
        except ImportError:
            return None
    return TaskAnalyzer()


@dataclass
class SubTask:
    """Individual subtask in a plan"""
    task_id: str
    description: str
    assigned_agent: str
    dependencies: List[str]
    estimated_duration: str  # e.g., "2 hours", "1 day"
    priority: str  # high, medium, low
    resources_needed: List[str]


@dataclass
class RiskItem:
    """Risk assessment item"""
    risk_id: str
    description: str
    severity: str  # critical, high, medium, low
    probability: str  # very_likely, likely, possible, unlikely
    mitigation: str
    impact: str


@dataclass
class TaskPlan:
    """Complete task execution plan"""
    plan_id: str
    goal: str
    subtasks: List[SubTask]
    total_estimated_duration: str
    required_agents: List[str]
    dependencies_graph: Dict[str, List[str]]
    risks: List[RiskItem]
    created_at: str


@dataclass
class Alternative:
    """Alternative approach to solving a problem"""
    approach_id: str
    description: str
    pros: List[str]
    cons: List[str]
    estimated_effort: str
    success_probability: str


class OracleAgent(BaseAgent):
    """
    Oracle Agent - Strategic Planning and Decision-Making

    The Oracle specializes in:
    - Breaking down complex goals into actionable subtasks
    - Assigning tasks to appropriate agents
    - Identifying dependencies and sequencing
    - Assessing risks
    - Generating alternative approaches

    Usage:
        oracle = OracleAgent(model_manager)

        # Decompose complex task
        plan = oracle.decompose_task(
            goal="Migrate database from MySQL to PostgreSQL",
            available_agents=["engineer", "operator", "security"],
            constraints=["zero downtime", "under 2 weeks"]
        )

        # Assess risks
        risks = oracle.assess_risk(plan)

        # Generate alternatives if needed
        alternatives = oracle.generate_alternatives(
            problem="Database migration",
            failed_approaches=["direct dump and restore"]
        )
    """

    def __init__(self, model_manager):
        super().__init__("oracle", model_manager)

        self.logger.info("✅ Oracle agent initialized with LLM support")

        # Agent capabilities mapping (for reference)
        self.agent_capabilities = {
            'operator': [
                'infrastructure setup', 'deployment', 'monitoring',
                'scaling', 'maintenance', 'troubleshooting'
            ],
            'security': [
                'security audit', 'vulnerability scanning', 'compliance',
                'access control', 'threat detection', 'secure coding'
            ],
            'knowledge': [
                'information retrieval', 'document search',
                'question answering', 'summarization'
            ],
            'coder': [
                'code generation', 'code analysis', 'refactoring',
                'file creation', 'project structure', 'documentation'
            ],
        }

        # Create LLM planning prompt
        self.planning_prompt = self._create_planning_prompt()

    def _create_planning_prompt(self) -> str:
        """Create prompt for LLM-based task decomposition"""
        return """You are the Oracle Agent - a strategic planning AI that breaks down complex goals into actionable subtasks.

Your role: Decompose goals into 3-7 concrete subtasks and assign them to the best agent for each task.

Available agents and their capabilities:
- engineer: code analysis, code generation, refactoring, documentation, code review, architecture design
- coder: code generation, file creation, project structure, implementation, programming
- security: security audit, vulnerability scanning, compliance, access control, threat detection
- operator: infrastructure setup, deployment, monitoring, scaling, maintenance
- navigator: resource discovery, documentation search, knowledge mapping, learning paths
- coordinator: workflow orchestration, progress tracking, conflict resolution
- knowledge: information retrieval, document search, question answering

CRITICAL RULES:
1. Break the goal into 3-7 concrete, actionable subtasks
2. Assign each subtask to the BEST agent for that specific task
3. For CODE GENERATION tasks, assign to "coder" agent (not engineer)
4. For CODE ANALYSIS tasks, assign to "engineer" agent
5. For SECURITY tasks, assign to "security" agent
6. Identify dependencies between subtasks (use task IDs)
7. Estimate realistic duration for each subtask
8. Set priority: high, medium, or low
9. List required resources

OUTPUT FORMAT (JSON only, no markdown):
{
  "subtasks": [
    {
      "task_id": "task_1",
      "description": "Clear, actionable description of what to do",
      "assigned_agent": "coder",
      "dependencies": [],
      "estimated_duration": "2 hours",
      "priority": "high",
      "resources_needed": ["coding environment", "libraries"]
    }
  ]
}

IMPORTANT:
- Output ONLY valid JSON, no explanations before or after
- No markdown code blocks (no ```json)
- Use double quotes for all strings
- Ensure JSON is valid and parseable

Goal to decompose:
"""

    def decompose_task(
        self,
        goal: str,
        available_agents: List[str] = None,
        constraints: List[str] = None,
        context: Optional[Dict] = None
    ) -> TaskPlan:
        """
        Break complex goal into subtasks

        Args:
            goal: High-level goal to achieve
            available_agents: List of available agent types
            constraints: Constraints to consider (time, resources, etc.)
            context: Additional context information

        Returns:
            TaskPlan with decomposed subtasks
        """
        if available_agents is None:
            available_agents = list(self.agent_capabilities.keys())

        # Generate plan ID
        plan_id = self._generate_id()

        # Use LLM to decompose goal into subtasks
        self.logger.info(f"🔮 Using LLM to decompose goal: {goal[:80]}...")
        subtasks = self._decompose_goal_with_llm(
            goal,
            available_agents,
            constraints or []
        )

        # Build dependency graph
        dependencies = self._build_dependency_graph(subtasks)

        # Estimate total duration
        total_duration = self._estimate_total_duration(subtasks, dependencies)

        # Assess risks
        risks = self._identify_risks(goal, subtasks, constraints or [])

        # Determine required agents
        required_agents = list(set(task.assigned_agent for task in subtasks))

        plan = TaskPlan(
            plan_id=plan_id,
            goal=goal,
            subtasks=subtasks,
            total_estimated_duration=total_duration,
            required_agents=required_agents,
            dependencies_graph=dependencies,
            risks=risks,
            created_at=datetime.now().isoformat()
        )

        return plan

    def _decompose_goal_with_llm(
        self,
        goal: str,
        available_agents: List[str],
        constraints: List[str]
    ) -> List[SubTask]:
        """
        Decompose goal using LLM

        Args:
            goal: High-level goal to achieve
            available_agents: List of available agent types
            constraints: Constraints to consider

        Returns:
            List of SubTask objects
        """
        # Build prompt
        prompt = self.planning_prompt + goal

        # Add constraints if provided
        if constraints:
            prompt += f"\n\nConstraints to consider:\n"
            for constraint in constraints:
                prompt += f"- {constraint}\n"

        # Add available agents info
        prompt += f"\n\nAvailable agents for this task: {', '.join(available_agents)}\n"

        # Generate request ID
        request_id = f"oracle_plan_{uuid_lib.uuid4().hex[:8]}"

        try:
            # Call LLM
            response = self.generate_with_logging(
                prompt=prompt,
                request_id=request_id,
                max_tokens=2000,
                temperature=0.1,  # Low temp for consistent planning
                top_p=0.9
            )

            # Parse LLM response
            subtasks = self._parse_llm_subtasks(response, goal, available_agents)

            self.logger.info(f"✅ LLM generated {len(subtasks)} subtasks")
            return subtasks

        except Exception as e:
            self.logger.error(f"❌ LLM decomposition failed: {e}", exc_info=True)
            self.logger.warning("⚠️  Falling back to rule-based decomposition")
            # Fallback to old method
            return self._decompose_goal(goal, available_agents, constraints)

    def _parse_llm_subtasks(
        self,
        llm_response: str,
        goal: str,
        available_agents: List[str]
    ) -> List[SubTask]:
        """
        Parse LLM JSON response into SubTask objects

        Args:
            llm_response: Raw LLM response
            goal: Original goal (for fallback)
            available_agents: List of available agents

        Returns:
            List of SubTask objects
        """
        json_text = llm_response.strip()

        # Remove markdown code blocks if present
        if '```json' in json_text:
            match = re.search(r'```json\s*(.*?)\s*```', json_text, re.DOTALL)
            if match:
                json_text = match.group(1)
        elif '```' in json_text:
            # Remove any ``` markers
            json_text = re.sub(r'```\w*\n?', '', json_text)
            json_text = json_text.replace('```', '').strip()

        # Try to parse JSON
        try:
            data = json.loads(json_text)
            subtasks_data = data.get('subtasks', [])

            if not subtasks_data:
                self.logger.warning("No subtasks in LLM response, using fallback")
                return self._create_generic_subtasks(goal, available_agents)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.debug(f"Raw response: {json_text[:500]}")
            # Fallback
            return self._create_generic_subtasks(goal, available_agents)

        # Convert to SubTask objects
        subtasks = []
        for i, task_data in enumerate(subtasks_data, 1):
            try:
                # Validate assigned agent is in available list
                assigned_agent = task_data.get('assigned_agent', available_agents[0])
                if assigned_agent not in available_agents:
                    self.logger.warning(
                        f"Agent '{assigned_agent}' not available, using '{available_agents[0]}'"
                    )
                    assigned_agent = available_agents[0]

                subtask = SubTask(
                    task_id=task_data.get('task_id', f'task_{i}'),
                    description=task_data.get('description', 'No description provided'),
                    assigned_agent=assigned_agent,
                    dependencies=task_data.get('dependencies', []),
                    estimated_duration=task_data.get('estimated_duration', '1 hour'),
                    priority=task_data.get('priority', 'medium'),
                    resources_needed=task_data.get('resources_needed', [])
                )
                subtasks.append(subtask)
            except Exception as e:
                self.logger.error(f"Error parsing subtask {i}: {e}")
                continue

        return subtasks if subtasks else self._create_generic_subtasks(goal, available_agents)

    def _decompose_goal(
        self,
        goal: str,
        available_agents: List[str],
        constraints: List[str]
    ) -> List[SubTask]:
        """
        Decompose goal into subtasks

        This is a template method that could be enhanced with LLM integration
        for more intelligent decomposition.
        """
        # Simplified decomposition logic
        # In production, this would use an LLM to intelligently decompose

        subtasks = []

        # Common patterns for different types of goals
        if 'migrate' in goal.lower() or 'migration' in goal.lower():
            subtasks = self._create_migration_subtasks(goal, available_agents)
        elif 'implement' in goal.lower() or 'develop' in goal.lower():
            subtasks = self._create_development_subtasks(goal, available_agents)
        elif 'optimize' in goal.lower() or 'improve' in goal.lower():
            subtasks = self._create_optimization_subtasks(goal, available_agents)
        elif 'analyze' in goal.lower() or 'audit' in goal.lower():
            subtasks = self._create_analysis_subtasks(goal, available_agents)
        elif any(keyword in goal.lower() for keyword in ['create', 'build', 'write', 'generate', 'code']):
            # Code generation pattern
            subtasks = self._create_code_generation_subtasks(goal, available_agents)
        else:
            # Generic decomposition
            subtasks = self._create_generic_subtasks(goal, available_agents)

        return subtasks

    def _create_migration_subtasks(
        self,
        goal: str,
        available_agents: List[str]
    ) -> List[SubTask]:
        """Create subtasks for migration projects"""
        subtasks = []

        # Analysis phase
        if 'coder' in available_agents or 'knowledge' in available_agents:
            subtasks.append(SubTask(
                task_id="migration_1",
                description="Analyze current system architecture and dependencies",
                assigned_agent='coder' if 'coder' in available_agents else 'knowledge',
                dependencies=[],
                estimated_duration="4 hours",
                priority="high",
                resources_needed=["access to current system", "documentation"]
            ))

        # Planning phase
        subtasks.append(SubTask(
            task_id="migration_2",
            description="Design migration strategy and rollback plan",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=["migration_1"],
            estimated_duration="6 hours",
            priority="high",
            resources_needed=["architecture docs", "requirements"]
        ))

        # Security review
        if 'security' in available_agents:
            subtasks.append(SubTask(
                task_id="migration_3",
                description="Security audit of migration plan",
                assigned_agent='security',
                dependencies=["migration_2"],
                estimated_duration="3 hours",
                priority="high",
                resources_needed=["migration plan", "security policies"]
            ))

        # Implementation
        if 'operator' in available_agents or 'coder' in available_agents:
            subtasks.append(SubTask(
                task_id="migration_4",
                description="Set up target environment and test migration",
                assigned_agent='operator' if 'operator' in available_agents else 'coder',
                dependencies=["migration_2", "migration_3"] if 'security' in available_agents else ["migration_2"],
                estimated_duration="2 days",
                priority="high",
                resources_needed=["infrastructure access", "test data"]
            ))

        # Testing
        subtasks.append(SubTask(
            task_id="migration_5",
            description="Execute migration and validate data integrity",
            assigned_agent='operator' if 'operator' in available_agents else available_agents[0],
            dependencies=["migration_4"],
            estimated_duration="1 day",
            priority="critical",
            resources_needed=["backup", "monitoring tools"]
        ))

        return subtasks

    def _create_development_subtasks(
        self,
        goal: str,
        available_agents: List[str]
    ) -> List[SubTask]:
        """Create subtasks for development projects"""
        subtasks = []

        # Requirements analysis
        subtasks.append(SubTask(
            task_id="dev_1",
            description="Gather and analyze requirements",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=[],
            estimated_duration="4 hours",
            priority="high",
            resources_needed=["stakeholder input", "use cases"]
        ))

        # Design
        subtasks.append(SubTask(
            task_id="dev_2",
            description="Design architecture and interfaces",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=["dev_1"],
            estimated_duration="1 day",
            priority="high",
            resources_needed=["requirements doc"]
        ))

        # Implementation
        subtasks.append(SubTask(
            task_id="dev_3",
            description="Implement core functionality",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=["dev_2"],
            estimated_duration="3 days",
            priority="high",
            resources_needed=["development environment", "design docs"]
        ))

        # Testing
        subtasks.append(SubTask(
            task_id="dev_4",
            description="Write and run tests",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=["dev_3"],
            estimated_duration="1 day",
            priority="high",
            resources_needed=["test framework", "test data"]
        ))

        # Security review
        if 'security' in available_agents:
            subtasks.append(SubTask(
                task_id="dev_5",
                description="Security audit and vulnerability scan",
                assigned_agent='security',
                dependencies=["dev_3"],
                estimated_duration="4 hours",
                priority="medium",
                resources_needed=["code access", "security tools"]
            ))

        # Deployment
        if 'operator' in available_agents:
            deps = ["dev_4", "dev_5"] if 'security' in available_agents else ["dev_4"]
            subtasks.append(SubTask(
                task_id="dev_6",
                description="Deploy to production",
                assigned_agent='operator',
                dependencies=deps,
                estimated_duration="2 hours",
                priority="high",
                resources_needed=["deployment pipeline", "production access"]
            ))

        return subtasks

    def _create_optimization_subtasks(
        self,
        goal: str,
        available_agents: List[str]
    ) -> List[SubTask]:
        """Create subtasks for optimization projects"""
        subtasks = []

        # Baseline measurement
        subtasks.append(SubTask(
            task_id="opt_1",
            description="Measure current performance baseline",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=[],
            estimated_duration="3 hours",
            priority="high",
            resources_needed=["monitoring tools", "test environment"]
        ))

        # Analysis
        subtasks.append(SubTask(
            task_id="opt_2",
            description="Identify bottlenecks and optimization opportunities",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=["opt_1"],
            estimated_duration="6 hours",
            priority="high",
            resources_needed=["profiling tools", "performance data"]
        ))

        # Implementation
        subtasks.append(SubTask(
            task_id="opt_3",
            description="Implement optimizations",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=["opt_2"],
            estimated_duration="2 days",
            priority="high",
            resources_needed=["codebase access"]
        ))

        # Validation
        subtasks.append(SubTask(
            task_id="opt_4",
            description="Measure improvements and validate results",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=["opt_3"],
            estimated_duration="4 hours",
            priority="high",
            resources_needed=["test data", "monitoring tools"]
        ))

        return subtasks

    def _create_analysis_subtasks(
        self,
        goal: str,
        available_agents: List[str]
    ) -> List[SubTask]:
        """Create subtasks for analysis/audit projects"""
        subtasks = []

        # Data collection
        subtasks.append(SubTask(
            task_id="analysis_1",
            description="Collect and organize relevant data",
            assigned_agent='knowledge' if 'knowledge' in available_agents else available_agents[0],
            dependencies=[],
            estimated_duration="4 hours",
            priority="high",
            resources_needed=["access to systems", "documentation"]
        ))

        # Analysis
        subtasks.append(SubTask(
            task_id="analysis_2",
            description="Perform detailed analysis",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=["analysis_1"],
            estimated_duration="1 day",
            priority="high",
            resources_needed=["analysis tools"]
        ))

        # Security check if applicable
        if 'security' in available_agents and 'security' in goal.lower():
            subtasks.append(SubTask(
                task_id="analysis_3",
                description="Security analysis and vulnerability assessment",
                assigned_agent='security',
                dependencies=["analysis_1"],
                estimated_duration="6 hours",
                priority="high",
                resources_needed=["security scanning tools"]
            ))

        # Report generation
        subtasks.append(SubTask(
            task_id="analysis_4",
            description="Generate comprehensive analysis report",
            assigned_agent='coder' if 'coder' in available_agents else available_agents[0],
            dependencies=["analysis_2"],
            estimated_duration="4 hours",
            priority="medium",
            resources_needed=["analysis results"]
        ))

        return subtasks

    def _create_code_generation_subtasks(
        self,
        goal: str,
        available_agents: List[str]
    ) -> List[SubTask]:
        """Create subtasks for code generation projects"""
        subtasks = []

        # Determine which coder agent to use
        coder_agent = 'coder' if 'coder' in available_agents else (
            'coder' if 'coder' in available_agents else available_agents[0]
        )

        # Requirement analysis
        subtasks.append(SubTask(
            task_id="codegen_1",
            description=f"Analyze requirements and design approach for: {goal}",
            assigned_agent='coder' if 'coder' in available_agents else coder_agent,
            dependencies=[],
            estimated_duration="2 hours",
            priority="high",
            resources_needed=["requirements", "design patterns"]
        ))

        # Code generation
        subtasks.append(SubTask(
            task_id="codegen_2",
            description=f"Generate code implementation for: {goal}",
            assigned_agent=coder_agent,
            dependencies=["codegen_1"],
            estimated_duration="6 hours",
            priority="high",
            resources_needed=["coding environment", "libraries"]
        ))

        # Code review and testing
        if 'coder' in available_agents:
            subtasks.append(SubTask(
                task_id="codegen_3",
                description="Review generated code for quality and best practices",
                assigned_agent='coder',
                dependencies=["codegen_2"],
                estimated_duration="2 hours",
                priority="medium",
                resources_needed=["code review tools"]
            ))

        # Security check if security agent available
        if 'security' in available_agents:
            subtasks.append(SubTask(
                task_id="codegen_4",
                description="Security audit of generated code",
                assigned_agent='security',
                dependencies=["codegen_2"],
                estimated_duration="3 hours",
                priority="high",
                resources_needed=["security scanning tools"]
            ))

        return subtasks

    def _create_generic_subtasks(
        self,
        goal: str,
        available_agents: List[str]
    ) -> List[SubTask]:
        """Create generic subtasks for unknown goal types"""
        subtasks = []

        # Planning
        subtasks.append(SubTask(
            task_id="task_1",
            description=f"Plan approach for: {goal}",
            assigned_agent=available_agents[0],
            dependencies=[],
            estimated_duration="4 hours",
            priority="high",
            resources_needed=["requirements"]
        ))

        # Execution
        subtasks.append(SubTask(
            task_id="task_2",
            description=f"Execute: {goal}",
            assigned_agent=available_agents[0],
            dependencies=["task_1"],
            estimated_duration="1 day",
            priority="high",
            resources_needed=["resources TBD"]
        ))

        # Validation
        subtasks.append(SubTask(
            task_id="task_3",
            description=f"Validate results for: {goal}",
            assigned_agent=available_agents[0],
            dependencies=["task_2"],
            estimated_duration="2 hours",
            priority="medium",
            resources_needed=["test criteria"]
        ))

        return subtasks

    def _build_dependency_graph(
        self,
        subtasks: List[SubTask]
    ) -> Dict[str, List[str]]:
        """Build dependency graph from subtasks"""
        graph = {}

        for task in subtasks:
            graph[task.task_id] = task.dependencies.copy()

        return graph

    def _estimate_total_duration(
        self,
        subtasks: List[SubTask],
        dependencies: Dict[str, List[str]]
    ) -> str:
        """Estimate total duration considering dependencies"""
        # Simplified estimation - sum of critical path
        # In production, would use proper critical path analysis

        total_hours = 0

        for task in subtasks:
            duration = task.estimated_duration
            hours = self._parse_duration(duration)
            total_hours = max(total_hours, hours)

        if total_hours < 8:
            return f"{total_hours} hours"
        elif total_hours < 40:
            return f"{total_hours / 8:.1f} days"
        else:
            return f"{total_hours / 40:.1f} weeks"

    def _parse_duration(self, duration: str) -> float:
        """Parse duration string to hours"""
        duration = duration.lower()

        if 'hour' in duration:
            return float(duration.split()[0])
        elif 'day' in duration:
            return float(duration.split()[0]) * 8
        elif 'week' in duration:
            return float(duration.split()[0]) * 40
        else:
            return 8.0  # Default

    def _identify_risks(
        self,
        goal: str,
        subtasks: List[SubTask],
        constraints: List[str]
    ) -> List[RiskItem]:
        """Identify potential risks in the plan"""
        risks = []

        # Check for constraint violations
        if 'zero downtime' in str(constraints).lower():
            risks.append(RiskItem(
                risk_id="risk_1",
                description="Potential service disruption during migration",
                severity="critical",
                probability="likely",
                mitigation="Implement blue-green deployment strategy",
                impact="Service unavailability affecting users"
            ))

        # Check for complex dependencies
        task_count = len(subtasks)
        if task_count > 10:
            risks.append(RiskItem(
                risk_id="risk_2",
                description="Complex plan with many interdependent tasks",
                severity="medium",
                probability="possible",
                mitigation="Break into smaller phases, add checkpoints",
                impact="Delays and coordination challenges"
            ))

        # Security risks
        if not any(task.assigned_agent == 'security' for task in subtasks):
            risks.append(RiskItem(
                risk_id="risk_3",
                description="No security review included in plan",
                severity="high",
                probability="very_likely",
                mitigation="Add security audit task",
                impact="Security vulnerabilities in implementation"
            ))

        return risks

    def assess_risk(self, plan: TaskPlan) -> List[RiskItem]:
        """
        Assess risks in execution plan

        Args:
            plan: Task plan to assess

        Returns:
            List of identified risks
        """
        return plan.risks

    def generate_alternatives(
        self,
        problem: str,
        failed_approaches: List[str] = None,
        constraints: List[str] = None
    ) -> List[Alternative]:
        """
        Generate alternative solutions

        Args:
            problem: Problem description
            failed_approaches: Previously failed approaches
            constraints: Constraints to consider

        Returns:
            List of alternative approaches
        """
        alternatives = []
        failed = failed_approaches or []

        # Generate alternatives based on problem type
        if 'migration' in problem.lower():
            alternatives = self._generate_migration_alternatives(failed)
        elif 'performance' in problem.lower() or 'optimization' in problem.lower():
            alternatives = self._generate_optimization_alternatives(failed)
        elif 'integration' in problem.lower():
            alternatives = self._generate_integration_alternatives(failed)
        else:
            alternatives = self._generate_generic_alternatives(problem, failed)

        return alternatives

    def _generate_migration_alternatives(
        self,
        failed: List[str]
    ) -> List[Alternative]:
        """Generate migration alternatives"""
        alternatives = []

        if 'direct dump and restore' not in [f.lower() for f in failed]:
            alternatives.append(Alternative(
                approach_id="alt_1",
                description="Direct dump and restore with minimal downtime",
                pros=["Simple", "Well-understood", "Fast"],
                cons=["Requires downtime", "Risk of data loss"],
                estimated_effort="2 days",
                success_probability="high"
            ))

        alternatives.append(Alternative(
            approach_id="alt_2",
            description="Blue-green deployment with gradual migration",
            pros=["Zero downtime", "Easy rollback", "Lower risk"],
            cons=["More complex", "Requires more resources"],
            estimated_effort="1 week",
            success_probability="very_high"
        ))

        alternatives.append(Alternative(
            approach_id="alt_3",
            description="Incremental migration with dual-write period",
            pros=["Very low risk", "Continuous validation"],
            cons=["Longest duration", "Complex coordination"],
            estimated_effort="2 weeks",
            success_probability="very_high"
        ))

        return alternatives

    def _generate_optimization_alternatives(
        self,
        failed: List[str]
    ) -> List[Alternative]:
        """Generate optimization alternatives"""
        alternatives = []

        alternatives.append(Alternative(
            approach_id="alt_1",
            description="Algorithm optimization and code refactoring",
            pros=["Permanent improvement", "No infrastructure changes"],
            cons=["Requires code changes", "Testing needed"],
            estimated_effort="1 week",
            success_probability="high"
        ))

        alternatives.append(Alternative(
            approach_id="alt_2",
            description="Caching layer implementation",
            pros=["Quick wins", "Easy to implement"],
            cons=["Cache invalidation complexity", "Memory overhead"],
            estimated_effort="3 days",
            success_probability="high"
        ))

        alternatives.append(Alternative(
            approach_id="alt_3",
            description="Horizontal scaling and load balancing",
            pros=["Handles growth", "Improves reliability"],
            cons=["Infrastructure cost", "Operational complexity"],
            estimated_effort="1 week",
            success_probability="very_high"
        ))

        return alternatives

    def _generate_integration_alternatives(
        self,
        failed: List[str]
    ) -> List[Alternative]:
        """Generate integration alternatives"""
        alternatives = []

        alternatives.append(Alternative(
            approach_id="alt_1",
            description="REST API integration with webhooks",
            pros=["Standard approach", "Well-supported"],
            cons=["Potential latency", "Network dependency"],
            estimated_effort="5 days",
            success_probability="high"
        ))

        alternatives.append(Alternative(
            approach_id="alt_2",
            description="Message queue based integration",
            pros=["Asynchronous", "Decoupled", "Reliable"],
            cons=["Additional infrastructure", "Eventual consistency"],
            estimated_effort="1 week",
            success_probability="very_high"
        ))

        alternatives.append(Alternative(
            approach_id="alt_3",
            description="Database-level integration via triggers",
            pros=["Real-time", "No application changes"],
            cons=["Database coupling", "Hard to debug"],
            estimated_effort="3 days",
            success_probability="medium"
        ))

        return alternatives

    def _generate_generic_alternatives(
        self,
        problem: str,
        failed: List[str]
    ) -> List[Alternative]:
        """Generate generic alternatives"""
        alternatives = []

        alternatives.append(Alternative(
            approach_id="alt_1",
            description=f"Standard approach for: {problem}",
            pros=["Well-tested", "Low risk"],
            cons=["May not be optimal"],
            estimated_effort="1 week",
            success_probability="high"
        ))

        alternatives.append(Alternative(
            approach_id="alt_2",
            description=f"Innovative approach for: {problem}",
            pros=["Potentially better results"],
            cons=["Higher risk", "Less proven"],
            estimated_effort="2 weeks",
            success_probability="medium"
        ))

        return alternatives

    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def export_plan(self, plan: TaskPlan, format: str = 'json') -> str:
        """
        Export plan to various formats

        Args:
            plan: Task plan to export
            format: Export format (json, markdown)

        Returns:
            Formatted plan string
        """
        if format == 'json':
            return json.dumps(asdict(plan), indent=2)
        elif format == 'markdown':
            return self._format_plan_markdown(plan)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _format_plan_markdown(self, plan: TaskPlan) -> str:
        """Format plan as markdown"""
        lines = []

        lines.append(f"# Task Plan: {plan.goal}\n")
        lines.append(f"**Plan ID**: {plan.plan_id}")
        lines.append(f"**Created**: {plan.created_at}")
        lines.append(f"**Estimated Duration**: {plan.total_estimated_duration}\n")

        lines.append("## Required Agents\n")
        for agent in plan.required_agents:
            lines.append(f"- {agent}")
        lines.append("")

        lines.append("## Subtasks\n")
        for task in plan.subtasks:
            lines.append(f"### {task.task_id}: {task.description}\n")
            lines.append(f"- **Agent**: {task.assigned_agent}")
            lines.append(f"- **Duration**: {task.estimated_duration}")
            lines.append(f"- **Priority**: {task.priority}")

            if task.dependencies:
                lines.append(f"- **Dependencies**: {', '.join(task.dependencies)}")

            lines.append("")

        if plan.risks:
            lines.append("## Risks\n")
            for risk in plan.risks:
                lines.append(f"### {risk.description}\n")
                lines.append(f"- **Severity**: {risk.severity}")
                lines.append(f"- **Probability**: {risk.probability}")
                lines.append(f"- **Mitigation**: {risk.mitigation}\n")

        return "\n".join(lines)

    def _needs_clarification(self, message: str, conversation_history: list = None) -> Optional[Dict[str, Any]]:
        """
        Check if a request needs clarification before proceeding.

        Detects vague or ambiguous requests that should prompt follow-up questions
        rather than attempting to execute with assumptions.

        Args:
            message: User's message
            conversation_history: Previous messages for context

        Returns:
            Dict with clarification info if needed, None otherwise.
            Format: {"needs_clarification": True, "questions": [...], "reason": "..."}
        """
        msg_lower = message.lower()

        # Check for vague docker container requests
        if 'docker' in msg_lower and ('container' in msg_lower or 'spin up' in msg_lower or 'create' in msg_lower):
            # Check if they specified an image
            image_indicators = ['image', 'ubuntu', 'alpine', 'debian', 'centos', 'nginx', 'redis',
                               'postgres', 'mysql', 'mongo', 'python:', 'node:', 'from ']
            has_image = any(ind in msg_lower for ind in image_indicators)

            # Check if they specified ports
            has_ports = 'port' in msg_lower or '-p ' in msg_lower or re.search(r'\d{2,5}:', msg_lower)

            if not has_image:
                questions = ["Which Docker image should I use? (e.g., ubuntu:latest, nginx:alpine)"]
                if not has_ports:
                    questions.append("Should I expose any ports? If so, which ones?")
                if 'persistent' not in msg_lower and 'volume' not in msg_lower:
                    questions.append("Do you need persistent storage (volume mounts)?")

                return {
                    "needs_clarification": True,
                    "questions": questions,
                    "reason": "Docker container request missing required details",
                    "original_request": message
                }

        # Check for vague setup/configure requests
        setup_keywords = ['setup', 'configure', 'set up', 'config']
        if any(kw in msg_lower for kw in setup_keywords):
            # If very short and vague
            word_count = len(message.split())
            if word_count < 5:
                return {
                    "needs_clarification": True,
                    "questions": [
                        "What specifically would you like me to set up or configure?",
                        "Are there any specific requirements or constraints?"
                    ],
                    "reason": "Setup/configure request too vague",
                    "original_request": message
                }

        # Check for vague "create" requests
        if msg_lower.startswith('create ') or ' create ' in msg_lower:
            create_targets = ['file', 'directory', 'folder', 'script', 'service', 'user',
                            'database', 'table', 'container', 'project', 'repo']
            has_target = any(target in msg_lower for target in create_targets)
            if not has_target and len(message.split()) < 4:
                return {
                    "needs_clarification": True,
                    "questions": ["What would you like me to create? (file, directory, service, etc.)"],
                    "reason": "Create request missing target",
                    "original_request": message
                }

        # Check for ambiguous pronouns without clear context
        pronouns = ['it', 'this', 'that', 'them', 'these', 'those']
        has_pronoun = any(re.search(rf'\b{p}\b', msg_lower) for p in pronouns)
        if has_pronoun:
            # Check if conversation history provides context
            has_context = False
            if conversation_history and len(conversation_history) >= 2:
                # Look for recent concrete references
                recent_messages = ' '.join([m.get('content', '') for m in conversation_history[-4:]])
                # If recent messages have concrete subjects, we have context
                concrete_indicators = ['file', 'docker', 'container', 'server', 'script', 'command',
                                      'service', 'process', 'directory', 'database']
                has_context = any(ind in recent_messages.lower() for ind in concrete_indicators)

            if not has_context:
                # Find which pronoun was used
                used_pronoun = next((p for p in pronouns if re.search(rf'\b{p}\b', msg_lower)), 'it')
                return {
                    "needs_clarification": True,
                    "questions": [f"Could you clarify what '{used_pronoun}' refers to?"],
                    "reason": "Ambiguous pronoun reference",
                    "original_request": message
                }

        return None

    def _should_triage(self, message: str) -> tuple:
        """
        Detect if message should be triaged to specialized agents.

        Returns:
            (should_triage: bool, reason: str, complexity: str)
        """
        import re
        msg_lower = message.lower()

        # System/infra keywords - check these FIRST before conversational
        # Messages about system resources should ALWAYS be triaged
        system_keywords = ['docker', 'container', 'server', 'database', 'port', 'service',
                          'disk', 'memory', 'cpu', 'process', 'file', 'files', 'directory',
                          'directories', 'folder', 'folders', 'path', 'network', 'ssh',
                          'git', 'deploy', 'kubernetes', 'compose', 'tmp', 'log', 'logs',
                          'config', 'configuration', 'environment', 'variable', 'package',
                          'module', 'library', 'dependency', 'dependencies', 'permission',
                          'permissions', 'user', 'group', 'system', 'binary', 'executable',
                          'space', 'storage', 'available', 'usage', 'running', 'status']

        # Action indicators for system queries (including "how much", "how many", "tell me")
        system_query_patterns = [
            r'how much\s+(free\s+)?disk',
            r'how much\s+(free\s+)?space',
            r'how much\s+(free\s+)?memory',
            r'how much\s+(free\s+)?storage',
            r'how many\s+files?',
            r'how many\s+containers?',
            r'how many\s+process',
            r'tell me.*(disk|space|memory|storage|cpu|running)',
            r'check\s+(disk|space|memory|storage|cpu)',
            r'show\s+(disk|space|memory|storage|cpu)',
            r'list\s+(files?|directories?|folders?|containers?|process)',
            r'what.*(is|are)\s+(running|available|free|used)',
            # "what is the (current) memory/disk/cpu usage/status"
            r'what\s+(is|are)\s+.*(memory|disk|cpu|storage|space)\s*(usage|status|utilization|consumption)',
            r'(memory|disk|cpu|storage|space)\s*(usage|status|utilization)\s*\??$',
        ]
        for pattern in system_query_patterns:
            if re.search(pattern, msg_lower):
                return (True, "system query detected", "moderate")

        # Conversational patterns - questions about concepts, advice-seeking
        # These should NOT trigger execution but get helpful answers
        conversational_patterns = [
            r'^what is\s+(a|an|the)\s+\w+(\s+\w+)?\??$',  # "what is a docker?" conceptual
            r'^what are\s+(the\s+)?\w+s?\??$',            # "what are containers?"
            r'^explain\s+(how|what|why)',                  # "explain how docker works"
            r'^tell me about\s+',                          # "tell me about python" (knowledge)
            r'^how does\s+\w+\s+work',                    # "how does git work?"
            r'^why is\s+',                                 # conceptual why
            r'^describe\s+(the|a|an)\s+',                 # "describe the architecture"
            r'^define\s+',                                 # "define a function" (conceptual)
            # Advice-seeking patterns
            r'^what\s+(approach|strategy|technique|method)',   # "what approach should I try"
            r'^how (do|can|should) i\s+',                      # "how do I decode base64"
            r'^what tools?\s+(can|should|do)',                 # "what tools can I use"
            r'^what.*(recommend|suggest)',                     # "what do you recommend"
            r'^(any|some)\s+(ideas?|suggestions?|advice|tips)', # "any ideas for this"
            r'^help me understand',                            # "help me understand"
            r'^can you (explain|help|advise)',                 # "can you explain"
            r'what would you',                                 # "what would you try"
            r'what should i try',                              # "what should I try"
            r'^(give me|provide)\s+(advice|tips|guidance)',    # "give me advice"
        ]
        # Check if this is an advice-seeking question (even with system keywords)
        # Questions like "how do I decode base64?" should be conversational
        advice_indicators = [
            r'^how (do|can|should) i\s+',
            r'^what (approach|strategy|should)',
            r'\?$',  # Ends with question mark
            r'^(can|could|would) you (explain|help|advise|recommend)',
        ]
        is_advice_seeking = any(re.search(p, msg_lower) for p in advice_indicators)

        # Apply conversational patterns if:
        # 1. No system keywords, OR
        # 2. Message is clearly seeking advice/explanation (even with system keywords)
        has_system = any(kw in msg_lower for kw in system_keywords)
        if not has_system or is_advice_seeking:
            for pattern in conversational_patterns:
                if re.search(pattern, msg_lower):
                    return (False, "conversational query", "simple")

        # Actionable overrides (kept for backwards compat)
        actionable_overrides = ['and change', 'and update', 'and run', 'and execute',
                               'then do', 'also', 'please run', 'please execute']

        # Multi-step tasks - definitely triage
        multi_step_indicators = ['and then', 'after that', 'first', 'next', 'also',
                                 'and change', 'and update', 'and modify', 'and run',
                                 'followed by', 'once done']
        if any(ind in msg_lower for ind in multi_step_indicators):
            return (True, "multi-step task detected", "complex")

        # Path detection - messages with file paths are almost always actionable
        # Match /path, ~/path, ./path patterns but EXCLUDE URLs
        path_pattern = r'(?<!:)[/~]\S*(?:/\S+)+'  # Negative lookbehind for :// URLs
        url_pattern = r'https?://|ftp://'
        has_url = bool(re.search(url_pattern, message))
        has_path = bool(re.search(path_pattern, message)) and not has_url
        if has_path:
            # If there's a path (not URL), it's likely a system operation
            return (True, "file path detected", "moderate")

        # URL/web detection
        if has_url:
            web_keywords = ['web', 'recon', 'scan', 'investigate', 'exploit', 'download']
            if any(kw in msg_lower for kw in web_keywords):
                return (True, "web URL detected", "moderate")

        # Shell command detection - common commands indicate execution
        shell_commands = ['ls', 'cd', 'cat', 'grep', 'find', 'rm', 'cp', 'mv', 'mkdir',
                         'chmod', 'chown', 'pip', 'npm', 'yarn', 'python', 'python3',
                         'node', 'cargo', 'go', 'make', 'cmake', 'curl', 'wget',
                         'tar', 'zip', 'unzip', 'apt', 'yum', 'brew', 'systemctl',
                         'journalctl', 'df', 'du', 'top', 'htop', 'ps', 'kill',
                         'docker', 'kubectl', 'terraform', 'ansible',
                         # GPU and hardware monitoring
                         'nvidia-smi', 'nvtop', 'gpustat', 'lspci', 'lsusb', 'lsblk',
                         'free', 'uptime', 'uname', 'hostname', 'whoami', 'id',
                         # Network commands
                         'ssh', 'scp', 'rsync', 'ping', 'netstat', 'ss', 'ifconfig', 'ip',
                         # Other common commands
                         'head', 'tail', 'less', 'more', 'wc', 'sort', 'uniq', 'awk', 'sed']
        # Check if message starts with or contains "run <command>" pattern
        for cmd in shell_commands:
            if re.search(rf'\b{cmd}\b', msg_lower):
                return (True, f"shell command detected ({cmd})", "moderate")

        # Action verbs that indicate execution
        action_verbs = ['check', 'run', 'execute', 'deploy', 'build', 'start', 'stop',
                        'create', 'delete', 'update', 'modify', 'install', 'configure',
                        'analyze', 'scan', 'fix', 'debug', 'test', 'migrate', 'list',
                        'show', 'get', 'set', 'add', 'remove', 'write', 'read',
                        'implement', 'refactor', 'develop', 'design', 'setup', 'init',
                        'initialize', 'generate', 'compile', 'package', 'publish']

        # has_system already defined above, use it with action verbs
        has_action = any(verb in msg_lower for verb in action_verbs)

        if has_action and has_system:
            return (True, "system action detected", "moderate")

        # Code/development tasks - expanded
        dev_keywords = ['code', 'function', 'class', 'implement', 'refactor', 'api',
                       'endpoint', 'bug', 'error', 'test', 'script', 'oauth', 'oauth2',
                       'auth', 'authentication', 'authorization', 'login', 'signup',
                       'register', 'database', 'schema', 'migration', 'model', 'controller',
                       'view', 'component', 'module', 'package', 'library', 'framework',
                       'frontend', 'backend', 'fullstack', 'web', 'mobile', 'app',
                       'application', 'feature', 'functionality', 'integration', 'webhook',
                       'callback', 'handler', 'middleware', 'route', 'routing', 'validation',
                       'jwt', 'token', 'session', 'cookie', 'cache', 'redis', 'mongo',
                       'postgres', 'mysql', 'sqlite', 'rest', 'graphql', 'grpc', 'socket']
        if has_action and any(kw in msg_lower for kw in dev_keywords):
            return (True, "development task detected", "moderate")

        # Direct command patterns - "run X", "execute X", etc.
        direct_cmd_patterns = [
            r'^run\s+\S+',           # "run something"
            r'^execute\s+\S+',       # "execute something"
            r'^please\s+run\s+',     # "please run"
            r'^can you run\s+',      # "can you run"
            r'^start\s+\S+',         # "start something"
            r'^stop\s+\S+',          # "stop something"
            r'^install\s+\S+',       # "install something"
            r'^create\s+\S+',        # "create something"
            r'^delete\s+\S+',        # "delete something"
            r'^build\s+\S+',         # "build something"
        ]
        for pattern in direct_cmd_patterns:
            if re.search(pattern, msg_lower):
                return (True, "direct command pattern", "simple")

        return (False, "simple query", "simple")

    def chat(self, message: str, conversation_history: list = None,
             auto_triage: bool = False, triage_approved: bool = False) -> str:
        """
        Generate a conversational response or trigger triage for complex tasks.

        Args:
            message: User's message
            conversation_history: Previous messages for context
            auto_triage: If True, skip approval and triage directly (fast mode)
            triage_approved: If True, user has approved triage - execute now

        Returns:
            Oracle's response. Special format for triage trigger:
            {"triage_trigger": True, "message": "...", "reason": "...", "complexity": "..."}
        """
        import uuid

        self.logger.info(f"Oracle.chat() received: {message[:50]}...")

        # Apply context resolution BEFORE triage decision
        # This converts "same thing on host1" to "Run 'docker ps' on host1 via SSH"
        resolved_message = message
        if conversation_history and len(conversation_history) > 0:
            resolved_message = self._resolve_context_references(message, conversation_history)
            if resolved_message != message:
                self.logger.info(f"Pre-triage context resolved: '{message}' → '{resolved_message}'")

        # Check if clarification is needed before proceeding
        # This prevents vague requests from being auto-triaged with assumptions
        clarification = self._needs_clarification(resolved_message, conversation_history)
        if clarification and not triage_approved:
            self.logger.info(f"Clarification needed: {clarification['reason']}")
            # Return clarifying questions to the user
            questions = clarification['questions']
            response_lines = ["I need a bit more information to help you effectively:"]
            response_lines.append("")
            for i, q in enumerate(questions, 1):
                response_lines.append(f"{i}. {q}")
            response_lines.append("")
            response_lines.append("Please provide these details so I can proceed.")
            return "\n".join(response_lines)

        # Check if this should be triaged (use resolved message for better keyword detection)
        should_triage, reason, complexity = self._should_triage(resolved_message)
        self.logger.info(f"Should triage: {should_triage} ({reason}, {complexity})")

        # If triage was approved, execute via Triage agent
        if triage_approved or (auto_triage and should_triage):
            self.logger.info("Triage approved, executing via Triage agent...")
            return self._execute_with_triage(resolved_message, conversation_history=conversation_history)

        # If should triage but not approved yet, return triage trigger
        if should_triage:
            self.logger.info("Returning triage trigger for user approval")
            # Return structured response that Console can detect
            # Include both original and resolved message for user clarity
            import json
            trigger_data = {
                "triage_trigger": True,
                "message": resolved_message,  # Use resolved message
                "original_message": message,  # Keep original for reference
                "reason": reason,
                "complexity": complexity,
                "prompt": f"This looks like a {complexity} task ({reason}). Would you like me to triage this to the appropriate agents? [y/n]"
            }
            # Add context note if resolved differently
            if resolved_message != message:
                trigger_data["context_note"] = f"(Understood as: {resolved_message})"
            return json.dumps(trigger_data)

        # Otherwise, generate a conversational response
        history_context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_context += f"{role.capitalize()}: {content}\n"

        prompt = f"""You are Oracle, an AI assistant that helps users accomplish their goals.

The user can also chat directly with these specialist agents via /agent <name>:
- operator: runs shell commands locally or over SSH
- security: suggests commands and reasoning for security-flavored tasks
- coder: generates code from a prompt
- knowledge: RAG-backed Q&A over documents indexed with manage_rag.py
- summarizer: long-document summarization with multiple styles
- triage: lightweight task classification

Do not invent or reference agents that are not in the list above. Other agent
names you may know (e.g. Navigator, Validator, Coordinator, Engineer) are NOT
available in this release — do not mention them.

When asked about tasks, you can execute them directly. For questions or planning,
provide helpful guidance.

{history_context}
User: {message}

Oracle:"""

        try:
            request_id = f"oracle_chat_{uuid.uuid4().hex[:8]}"
            response = self.generate_with_logging(
                prompt=prompt,
                request_id=request_id,
                timeout=60,
                max_tokens=500,
                temperature=0.7
            )

            response = response.strip()
            if response.startswith("Oracle:"):
                response = response[7:].strip()

            return response

        except Exception as e:
            self.logger.error(f"Chat generation failed: {e}")
            return f"I encountered an issue: {str(e)}"

    def _resolve_context_references(self, message: str, conversation_history: list) -> str:
        """
        Resolve references like 'same thing on host1' to explicit commands.

        This pre-processes the message to isolate:
        - Target host (host1, host2, etc.)
        - Action from previous context (docker ps, etc.)

        Returns an explicit message that agents can process without ambiguity.
        """
        import re

        # Known hosts in the environment
        known_hosts = ['host1', 'host2', 'localhost', 'local']

        # Reference words that indicate context is needed (use word boundary matching)
        reference_words = ['this', 'that', 'same', 'again', 'those', 'these', 'there']
        # Note: Removed 'it' - too common as substring (write, with, etc.)

        msg_lower = message.lower()

        # Use word boundary matching to avoid false positives like "it" in "write"
        needs_context = any(re.search(rf'\b{word}\b', msg_lower) for word in reference_words)

        if not needs_context or not conversation_history:
            return message

        # Extract target host from current message - only match known hosts with word boundaries
        target_host = None
        for host in known_hosts:
            if re.search(rf'\b{host}\b', msg_lower):
                target_host = host
                break

        # Also check for SSH patterns like "on host1" - require word boundary before on/to/from
        # This prevents matching "python hello" as "on hello"
        ssh_pattern = re.search(r'\b(?:on|to|from)\s+(\w+)', msg_lower)
        if not target_host and ssh_pattern:
            potential_host = ssh_pattern.group(1)
            # Only accept if it looks like a hostname (not common words)
            common_words = ['the', 'this', 'that', 'it', 'a', 'an', 'my', 'your', 'our',
                           'server', 'machine', 'computer', 'system', 'box', 'host',
                           'python', 'script', 'code', 'file', 'hello', 'world']
            if potential_host not in common_words:
                target_host = potential_host

        # Extract previous action from history
        previous_action = None
        previous_command = None

        for msg in reversed(conversation_history[-6:]):
            content = msg.get('content', '')
            role = msg.get('role', 'user')

            if role == 'user':
                # Look for command keywords in user request
                action_patterns = [
                    (r'(docker\s+\w+)', 'docker'),
                    (r'(df\s+-h)', 'disk'),
                    (r'(ps\s+aux)', 'processes'),
                    (r'(ls\s+-\w+)', 'list'),
                    (r'(cat\s+\S+)', 'view'),
                    (r'(uptime|free|top)', 'system'),
                    (r'(ping\s+\S+)', 'network'),
                    (r'(curl\s+\S+)', 'http'),
                ]

                content_lower = content.lower()
                for pattern, action_type in action_patterns:
                    match = re.search(pattern, content_lower)
                    if match:
                        previous_command = match.group(1)
                        previous_action = action_type
                        break

                # Also extract general action if no specific command found
                if not previous_action:
                    if 'disk' in content_lower or 'space' in content_lower:
                        previous_action = 'disk'
                        previous_command = 'df -h'
                    elif 'docker' in content_lower:
                        previous_action = 'docker'
                        previous_command = 'docker ps'
                    elif 'process' in content_lower:
                        previous_action = 'processes'
                        previous_command = 'ps aux'
                    elif 'memory' in content_lower or 'ram' in content_lower:
                        previous_action = 'memory'
                        previous_command = 'free -h'

                if previous_action:
                    break

            # Also check assistant responses for executed commands
            elif role == 'assistant':
                # Look for command outputs in assistant response
                cmd_match = re.search(r'\$\s*(.+?)(?:\n|$)', content)
                if cmd_match:
                    previous_command = cmd_match.group(1).strip()
                    if 'docker' in previous_command:
                        previous_action = 'docker'
                    elif 'df' in previous_command:
                        previous_action = 'disk'
                    break

        # Build explicit message if we have context
        if target_host and previous_command:
            explicit_msg = f"Run '{previous_command}' on {target_host} via SSH"
            self.logger.info(f"Context resolved: '{message}' → '{explicit_msg}'")
            return explicit_msg
        elif target_host and previous_action:
            action_to_cmd = {
                'docker': 'docker ps',
                'disk': 'df -h',
                'processes': 'ps aux',
                'memory': 'free -h',
                'system': 'uptime',
            }
            cmd = action_to_cmd.get(previous_action, previous_action)
            explicit_msg = f"Run '{cmd}' on {target_host} via SSH"
            self.logger.info(f"Context resolved: '{message}' → '{explicit_msg}'")
            return explicit_msg
        elif target_host and previous_action is None and previous_command is None:
            # We have a host reference but no clear action from history
            # Only transform if the message is purely a context reference like "same on host1"
            # Don't transform if it's a full sentence with its own meaning
            word_count = len(message.split())
            if word_count <= 5:  # Short message likely a reference
                explicit_msg = f"Check status of {target_host} (SSH or ping)"
                self.logger.info(f"Context resolved (host only): '{message}' → '{explicit_msg}'")
                return explicit_msg

        # Fallback: return original message unchanged
        return message

    def _execute_with_triage(self, message: str, plan_approved: bool = False,
                              approved_plan: dict = None, validate: bool = True,
                              conversation_history: list = None) -> str:
        """
        Execute task using Triage agent for classification.

        This is the "fast mode" / "standard mode" - automatic delegation via Triage.
        For COMPLEX tasks, generates a plan requiring user approval before execution.

        Args:
            message: User's task request
            plan_approved: If True, execute the approved plan
            approved_plan: The plan that was approved (for execution)
            validate: If True, validate execution and retry if incomplete
            conversation_history: Previous messages for context (to resolve references like "this")

        Returns:
            Execution result formatted for display, or plan_approval trigger JSON
        """
        import json
        from agents.triage_agent import ModularTriageAgent, TaskCategory, TaskDifficulty

        try:
            # If executing an approved plan
            if plan_approved and approved_plan:
                return self._execute_approved_plan(approved_plan)

            # Build context-enhanced message by resolving references
            enhanced_message = message
            if conversation_history and len(conversation_history) > 0:
                # Use the smart context resolution method
                resolved_message = self._resolve_context_references(message, conversation_history)

                if resolved_message != message:
                    # We resolved context references to an explicit command
                    enhanced_message = resolved_message
                    self.logger.info(f"Context resolved: '{message}' → '{enhanced_message}'")
                else:
                    # No resolution needed - check if we still need raw context appended
                    # Use word boundary matching (not substring) to avoid false positives
                    # Note: 'it' removed - too common as substring (write, with, etc.)
                    reference_words = ['this', 'that', 'same', 'there', 'those', 'these', 'again']
                    msg_lower = message.lower()
                    needs_context = any(re.search(rf'\b{word}\b', msg_lower) for word in reference_words)

                    if needs_context:
                        # Fallback: append raw context if resolution didn't help
                        context_parts = []
                        for msg in conversation_history[-6:]:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')[:200]
                            context_parts.append(f"{role}: {content}")

                        context_summary = "\n".join(context_parts)
                        enhanced_message = f"""[Previous conversation context:]
{context_summary}

[Current request:]
{message}"""
                        self.logger.info(f"Fallback context append (references: {[w for w in reference_words if w in msg_lower]})")

            # Initialize Triage agent
            if not hasattr(self, '_triage_agent'):
                self._triage_agent = ModularTriageAgent(self.model_manager)

            # Pre-triage override for obvious patterns (faster than LLM for clear cases)
            msg_lower = enhanced_message.lower()
            override_category = None

            # Obvious coding/development patterns - bypass triage LLM
            coding_patterns = [
                (r'write\s+(a\s+)?python', TaskCategory.CODING),
                (r'write\s+(a\s+)?script', TaskCategory.CODING),
                (r'create\s+(a\s+)?function', TaskCategory.CODING),
                (r'create\s+(a\s+)?class', TaskCategory.CODING),
                (r'generate\s+code', TaskCategory.CODING),
                (r'hello\s*world', TaskCategory.CODING),  # Classic test
                (r'write\s+(a\s+)?program', TaskCategory.CODING),
                (r'implement\s+a\s+', TaskCategory.CODING),
            ]

            # Summarization patterns - route to summarizer
            summarization_patterns = [
                (r'summarize\s+.*\.(epub|pdf|mobi|docx?|txt)', TaskCategory.SUMMARIZATION),
                (r'summarize\s+chapters?\s+\d+', TaskCategory.SUMMARIZATION),
                (r'summarize\s+~/', TaskCategory.SUMMARIZATION),
                (r'summarize\s+/', TaskCategory.SUMMARIZATION),
                (r'(explanatory|narrative|academic|technical)\s+summar', TaskCategory.SUMMARIZATION),
            ]

            self.logger.info(f"Pre-triage check on: '{msg_lower[:50]}...'")
            # Check coding patterns first
            for pattern, category in coding_patterns:
                match = re.search(pattern, msg_lower)
                if match:
                    override_category = category
                    self.logger.info(f"Pre-triage MATCHED: '{pattern}' found '{match.group()}' → {category.value}")
                    break
            # Check summarization patterns
            if not override_category:
                for pattern, category in summarization_patterns:
                    match = re.search(pattern, msg_lower)
                    if match:
                        override_category = category
                        self.logger.info(f"Pre-triage SUMMARIZATION: '{pattern}' found '{match.group()}' → {category.value}")
                        break
            if not override_category:
                self.logger.info(f"Pre-triage: no patterns matched, falling through to LLM")

            # Classify the request (use enhanced message for context)
            self.logger.info(f"Triaging: {enhanced_message[:100]}...")

            if override_category:
                # Use override instead of LLM triage
                from agents.triage_agent import TriageResult, TaskDifficulty
                from datetime import datetime
                triage_result = TriageResult(
                    category=override_category,
                    difficulty=TaskDifficulty.SIMPLE,
                    confidence=0.95,
                    reasoning="Pre-triage pattern match for obvious coding request",
                    needs_clarification=False,
                    processing_time=0.001,  # Nearly instant since no LLM call
                    timestamp=datetime.now()
                )
            else:
                triage_result = self._triage_agent.analyze(enhanced_message)

            self.logger.info(f"Triage result: {triage_result.category.value} "
                           f"(confidence: {triage_result.confidence:.0%}, "
                           f"difficulty: {triage_result.difficulty.value})")

            # For COMPLEX tasks, generate plan and require approval
            if triage_result.difficulty == TaskDifficulty.COMPLEX:
                return self._generate_plan_for_approval(message, triage_result)

            # Build execution response for SIMPLE/MODERATE tasks
            response_parts = [
                f"📋 Triage: {triage_result.category.value.upper()} task",
                f"   Confidence: {triage_result.confidence:.0%}",
                f"   Difficulty: {triage_result.difficulty.value}",
                f"   Reasoning: {triage_result.reasoning[:100]}...\n"
            ]

            # Route to appropriate agent(s) based on category
            category = triage_result.category

            # Map categories to agent execution
            category_agent_map = {
                TaskCategory.SYSADMIN: ("operator", None),
                TaskCategory.FILEOPS: ("operator", None),
                TaskCategory.NETWORK: ("operator", None),
                TaskCategory.SECURITY: ("security", "operator"),
                TaskCategory.CODING: ("coder", None),
                TaskCategory.DEVELOPMENT: ("coder", None),
                TaskCategory.SUMMARIZATION: ("summarizer", None),
                TaskCategory.KNOWLEDGE_QUERY: ("knowledge", None),
                TaskCategory.CONTENT: ("knowledge", None),
                TaskCategory.UNKNOWN: ("operator", None)
            }

            primary_agent, secondary_agent = category_agent_map.get(
                category, ("operator", None)
            )
            self.logger.info(f"Routing to agent: primary={primary_agent}, secondary={secondary_agent}")

            # Special handling for SUMMARIZATION - validate file path first
            if category == TaskCategory.SUMMARIZATION:
                preprocess = self._preprocess_summarization(enhanced_message)

                if not preprocess['valid']:
                    # File doesn't exist - help user find the right file
                    if preprocess['alternatives']:
                        alt_list = "\n".join(f"  • {f}" for f in preprocess['alternatives'][:15])
                        suggestion_msg = (
                            f"📚 **File not found:** `{preprocess['file_path']}`\n\n"
                            f"**Available books in** `{preprocess['directory']}`:\n{alt_list}\n\n"
                        )

                        # Suggest command with first matching file
                        if preprocess['alternatives']:
                            suggested_file = f"{preprocess['directory']}/{preprocess['alternatives'][0]}"
                            suggested_cmd = self._format_summarization_command(
                                suggested_file,
                                preprocess['style'],
                                preprocess['compression'],
                                preprocess['chapters']
                            )
                            suggestion_msg += f"**Try:** `{suggested_cmd}`"

                        return suggestion_msg
                    elif preprocess['file_path']:
                        return f"📚 **File not found:** `{preprocess['file_path']}`\n\nPlease check the path and try again."
                else:
                    # Valid path - format proper command for summarizer
                    formatted_cmd = self._format_summarization_command(
                        preprocess['file_path'],
                        preprocess['style'],
                        preprocess['compression'],
                        preprocess['chapters']
                    )
                    self.logger.info(f"Formatted summarization command: {formatted_cmd}")
                    enhanced_message = formatted_cmd

            # Execute via appropriate agent
            try:
                from agents.workflow_executor import OracleTaskRunner
            except ImportError:
                # workflow_executor is not part of v0.2 — Oracle delegation
                # falls back to telling the user how to reach the specialist
                # agent directly.
                hint = primary_agent if primary_agent else "operator"
                return (
                    "Oracle's task-runner is not available in this release.\n"
                    f"Use direct-agent mode for this request:  /agent {hint}\n"
                    "(Type / before the command, then describe your task.)"
                )

            if not hasattr(self, '_task_runner'):
                self._task_runner = OracleTaskRunner(self.model_manager)

            if secondary_agent:
                # Two-agent flow (e.g., Security → Operator)
                response_parts.append(f"🔗 Routing: {primary_agent.title()} → {secondary_agent.title()}")
                chain_results = self._task_runner.execute_chain([
                    {"agent": primary_agent, "task": enhanced_message},
                    {"agent": secondary_agent, "task": enhanced_message}
                ])
                # Combine results
                all_success = all(r.get('success', False) for r in chain_results)
                combined_output = "\n\n".join(
                    f"[{r.get('agent', 'agent')}] {r.get('result', r.get('error', 'No result'))}"
                    for r in chain_results
                )
                if all_success:
                    response_parts.append(f"\n✅ Tasks completed\n\n{combined_output}")
                else:
                    response_parts.append(f"\n⚠️ Partial completion\n\n{combined_output}")

                # Aggregate structured results from chain (use last agent's execution results)
                last_result = chain_results[-1] if chain_results else {}
                result = {
                    'success': all_success,
                    'commands_executed': sum(r.get('commands_executed', 0) for r in chain_results),
                    'successful_commands': sum(r.get('successful_commands', 0) for r in chain_results),
                    'failed_commands': sum(r.get('failed_commands', 0) for r in chain_results),
                    'execution_results': last_result.get('execution_results', []),
                    'next_steps': last_result.get('next_steps', ''),
                    'result': combined_output
                }
            else:
                # Single agent execution
                response_parts.append(f"🔗 Routing: {primary_agent.title()}")
                self.logger.info(f"Calling task_runner.execute for {primary_agent}")
                result = self._task_runner.execute(primary_agent, enhanced_message)
                self.logger.info(f"Task runner returned: success={result.get('success')}, commands_executed={result.get('commands_executed', 0)}, error={result.get('error', 'none')}")

                if result.get('success'):
                    output = result.get('result', 'Task completed')
                    cleaned = self._clean_operator_output(str(output)) if primary_agent == 'operator' else str(output)
                    response_parts.append(f"\n✅ {primary_agent.title()} completed\n\n{cleaned}")
                else:
                    response_parts.append(f"\n❌ {primary_agent.title()} failed: {result.get('error', 'Unknown error')}")

            # Build initial response
            response = "\n".join(response_parts)

            # Store structured execution result for validation
            # This preserves the full context from the agent execution
            structured_result = {
                'result': response,
                'success': result.get('success', False),
                'commands_executed': result.get('commands_executed', 0),
                'successful_commands': result.get('successful_commands', 0),
                'failed_commands': result.get('failed_commands', 0),
                'execution_results': result.get('execution_results', []),
                'next_steps': result.get('next_steps', ''),
                'agent_used': primary_agent
            }

            # Validate execution if enabled (and not a RETRY to avoid infinite loops)
            # Skip validation for summarization - it's a text generation task, not command execution
            skip_validation = (
                message.startswith("RETRY:") or
                primary_agent == 'summarizer' or
                category == TaskCategory.SUMMARIZATION
            )
            if validate and not skip_validation:
                try:
                    validation_result = self._validate_execution(
                        user_request=message,
                        execution_result=structured_result,
                        enable_retry=True,
                        max_retries=2
                    )

                    # Append validation status
                    if validation_result.get('validated'):
                        response += "\n\n🔍 **Validation**: ✅ Task verified complete"
                        conf = validation_result.get('validation', {}).get('confidence', 0)
                        if conf > 0:
                            response += f" (confidence: {conf:.0%})"
                    else:
                        response = validation_result.get('result', response)
                        response += "\n\n🔍 **Validation**: ⚠️ Task may be incomplete"

                        # Show feedback if available
                        feedback = validation_result.get('feedback')
                        if feedback:
                            response += f"\n\n💡 **Suggestions**:\n{feedback[:2000]}"

                except Exception as e:
                    self.logger.warning(f"Validation skipped due to error: {e}")
                    # Don't fail the whole execution if validation fails

            return response

        except Exception as e:
            self.logger.error(f"Triage execution failed: {e}", exc_info=True)
            return f"❌ Triage execution failed: {str(e)}"

    def _preprocess_summarization(self, message: str) -> dict:
        """
        Pre-process summarization requests to validate file paths and suggest alternatives.

        This method:
        1. Extracts file path from the message
        2. Validates the path exists
        3. If not found, lists available books in that directory
        4. Returns info for user confirmation or direct execution

        Args:
            message: User's summarization request

        Returns:
            dict with:
                - valid: bool - whether path is valid
                - file_path: str - the file path (expanded)
                - alternatives: list - available files if path invalid
                - directory: str - parent directory
                - style: str - detected style (if any)
                - compression: str - detected compression (if any)
                - chapters: list - detected chapter range (if any)
        """
        from pathlib import Path
        import re

        result = {
            'valid': False,
            'file_path': None,
            'alternatives': [],
            'directory': None,
            'style': None,
            'compression': None,
            'chapters': None,
            'message': message
        }

        # Extract file path from message
        # Patterns: ~/path/to/file.pdf, /absolute/path.epub, "./relative.pdf"
        path_patterns = [
            r'(~[^\s]+\.(?:epub|pdf|mobi|docx?|txt))',  # ~/path/file.ext
            r'(/[^\s]+\.(?:epub|pdf|mobi|docx?|txt))',   # /absolute/path.ext
            r'(\./[^\s]+\.(?:epub|pdf|mobi|docx?|txt))', # ./relative.ext
        ]

        file_path = None
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                file_path = match.group(1)
                break

        if not file_path:
            # Check if there's a path-like pattern without extension
            generic_path = re.search(r'(~/[^\s]+|/[^\s]+)', message)
            if generic_path:
                file_path = generic_path.group(1)

        if not file_path:
            return result

        # Expand and validate path
        expanded_path = Path(file_path).expanduser()
        result['file_path'] = str(expanded_path)

        if expanded_path.exists() and expanded_path.is_file():
            result['valid'] = True
        else:
            # Path doesn't exist - find alternatives
            parent = expanded_path.parent if expanded_path.suffix else expanded_path
            if parent.exists() and parent.is_dir():
                result['directory'] = str(parent)
                # List available books in directory
                book_extensions = {'.epub', '.pdf', '.mobi', '.doc', '.docx', '.txt'}
                alternatives = []
                for f in parent.iterdir():
                    if f.is_file() and f.suffix.lower() in book_extensions:
                        alternatives.append(f.name)
                # Sort by name
                result['alternatives'] = sorted(alternatives)

        # Extract style
        style_match = re.search(r'\b(narrative|academic|technical|explanatory|quick)\b', message, re.IGNORECASE)
        if style_match:
            result['style'] = style_match.group(1).lower()

        # Extract compression
        comp_match = re.search(r'\b(detailed|standard|condensed|outline)\b', message, re.IGNORECASE)
        if comp_match:
            result['compression'] = comp_match.group(1).lower()

        # Extract chapters
        chapter_match = re.search(r'chapters?\s+(\d+)\s*[-–]\s*(\d+)', message, re.IGNORECASE)
        if chapter_match:
            result['chapters'] = [int(chapter_match.group(1)), int(chapter_match.group(2))]
        else:
            single_chapter = re.search(r'chapter\s+(\d+)\b', message, re.IGNORECASE)
            if single_chapter:
                result['chapters'] = [int(single_chapter.group(1))]

        return result

    def _format_summarization_command(self, file_path: str, style: str = None,
                                       compression: str = None, chapters: list = None) -> str:
        """Format a proper summarization command for the summarizer agent."""
        parts = ["summarize"]

        if chapters:
            if len(chapters) == 1:
                parts.append(f"chapter {chapters[0]}")
            else:
                parts.append(f"chapters {chapters[0]}-{chapters[1]}")
            parts.append("from")

        parts.append(file_path)

        if style:
            parts.append(f"--style {style}")

        if compression:
            parts.append(compression)

        return " ".join(parts)

    def _generate_plan_for_approval(self, message: str, triage_result) -> str:
        """
        Generate a plan for COMPLEX tasks requiring user approval.

        Uses TaskAnalyzer for:
        - Input type detection (folder, file, URL, text, etc.)
        - Complexity scoring with coverage %
        - Agent recommendations based on task analysis
        - Risk assessment

        Args:
            message: User's task request
            triage_result: Triage classification result

        Returns:
            JSON string with plan_approval trigger
        """
        import json
        from dataclasses import asdict

        try:
            # Phase D: Use TaskAnalyzer for intelligent analysis (optional)
            task_analyzer = _get_task_analyzer()
            if task_analyzer is None:
                raise ImportError("task_analyzer not available")
            analysis = task_analyzer.analyze(message)

            # Get agents from TaskAnalyzer recommendations + defaults
            recommended_agents = analysis.get_agent_chain()
            available_agents = list(set(
                recommended_agents + ["security", "operator", "navigator", "coder", "knowledge"]
            ))

            # Generate plan using decompose_task with analyzer context
            plan = self.decompose_task(
                goal=message,
                available_agents=available_agents,
                context={
                    "input_type": analysis.input_type.value,
                    "input_path": analysis.input_path,
                    "complexity": analysis.complexity.to_dict() if analysis.complexity else None,
                    "recommended_agents": recommended_agents,
                    "subtasks_suggested": analysis.subtasks
                }
            )

            # Format plan for display
            plan_display = []
            plan_display.append(f"📋 Task Plan: {plan.goal}")

            # Add TaskAnalyzer insights
            if analysis.input_type.value != "text":
                plan_display.append(f"📁 Input Type: {analysis.input_type.value}")
                if analysis.input_path:
                    plan_display.append(f"   Path: {analysis.input_path}")

            if analysis.complexity:
                complexity = analysis.complexity
                plan_display.append(f"📊 Complexity: {complexity.complexity.value.upper()}")
                plan_display.append(f"   Risk Level: {complexity.risk_level.value}")
                if complexity.total_effort > 0:
                    plan_display.append(f"   Effort: {complexity.total_effort:.0%}")
                if complexity.existing_coverage > 0:
                    plan_display.append(f"   Existing Coverage: {complexity.existing_coverage:.0%}")

            plan_display.append(f"⏱️  Estimated Duration: {plan.total_estimated_duration}")
            plan_display.append(f"🤖 Required Agents: {', '.join(plan.required_agents)}")

            # Show recommended agent chain from TaskAnalyzer
            if recommended_agents:
                plan_display.append(f"   (Recommended: {' → '.join(recommended_agents)})")

            plan_display.append("\n📝 Subtasks:")

            for i, task in enumerate(plan.subtasks, 1):
                plan_display.append(f"  {i}. {task.description}")
                plan_display.append(f"     → Agent: {task.assigned_agent}")
                plan_display.append(f"     → Priority: {task.priority}")
                if task.dependencies:
                    plan_display.append(f"     → Depends on: {', '.join(task.dependencies)}")

            if plan.risks:
                plan_display.append("\n⚠️  Risks:")
                for risk in plan.risks[:3]:
                    plan_display.append(f"  • {risk.description} (Severity: {risk.severity})")

            # Add warnings from analysis
            if analysis.warnings:
                plan_display.append("\n⚠️  Warnings:")
                for warning in analysis.warnings[:3]:
                    plan_display.append(f"  • {warning}")

            # Serialize plan for later execution
            plan_data = {
                "plan_id": plan.plan_id,
                "goal": plan.goal,
                "subtasks": [asdict(t) for t in plan.subtasks],
                "required_agents": plan.required_agents,
                "total_estimated_duration": plan.total_estimated_duration,
                # Include analysis for execution context
                "analysis": {
                    "input_type": analysis.input_type.value,
                    "input_path": analysis.input_path,
                    "complexity": analysis.complexity.complexity.value if analysis.complexity else "unknown",
                    "risk_level": analysis.complexity.risk_level.value if analysis.complexity else "low"
                }
            }

            # Return plan approval trigger
            return json.dumps({
                "plan_approval": True,
                "message": message,
                "triage": {
                    "category": triage_result.category.value,
                    "difficulty": triage_result.difficulty.value,
                    "confidence": triage_result.confidence
                },
                "analysis": analysis.to_dict(),
                "plan_display": "\n".join(plan_display),
                "plan_data": plan_data,
                "prompt": "Approve this plan? [y/n/modify]"
            })

        except Exception as e:
            self.logger.error(f"Plan generation failed: {e}", exc_info=True)
            # Fall back to direct execution
            return f"❌ Could not generate plan: {str(e)}. Proceeding with direct execution..."

    def _execute_approved_plan(self, plan: dict) -> str:
        """
        Execute a user-approved plan.

        Args:
            plan: The approved plan data

        Returns:
            Execution results
        """
        try:
            from agents.workflow_executor import OracleTaskRunner
        except ImportError:
            return (
                "Plan execution is not available in this release.\n"
                "Use direct-agent mode (e.g.,  /agent operator) and describe each step."
            )

        try:
            # Initialize components
            if not hasattr(self, '_task_runner'):
                self._task_runner = OracleTaskRunner(self.model_manager)

            response_parts = [
                f"🚀 Executing approved plan: {plan.get('goal', 'Unknown')}",
                f"   Subtasks: {len(plan.get('subtasks', []))}",
                ""
            ]

            # Execute each subtask in order (respecting dependencies)
            subtasks = plan.get('subtasks', [])
            completed = {}

            for i, task in enumerate(subtasks, 1):
                task_id = task.get('task_id', f'task_{i}')
                agent_name = task.get('assigned_agent', 'operator')
                description = task.get('description', 'No description')

                # Check dependencies
                deps = task.get('dependencies', [])
                if deps:
                    unmet = [d for d in deps if d not in completed]
                    if unmet:
                        response_parts.append(f"⏳ [{i}] Waiting for: {', '.join(unmet)}")
                        continue

                response_parts.append(f"▶️  [{i}] {description}")
                response_parts.append(f"   Agent: {agent_name}")

                try:
                    result = self._task_runner.execute(agent_name, description)

                    if result.get('success'):
                        response_parts.append(f"   ✅ Completed")
                        completed[task_id] = result
                    else:
                        response_parts.append(f"   ❌ Failed: {result.get('error', 'Unknown')}")
                except Exception as e:
                    response_parts.append(f"   ❌ Error: {str(e)}")

                response_parts.append("")

            # Summary
            success_count = len(completed)
            total_count = len(subtasks)
            response_parts.append(f"📊 Completed: {success_count}/{total_count} subtasks")

            if success_count == total_count:
                response_parts.append("🎉 Plan executed successfully!")
            else:
                response_parts.append("⚠️  Plan partially completed")

            response = "\n".join(response_parts)

            # Validate the overall plan execution
            goal = plan.get('goal', 'Execute plan')
            try:
                validation_result = self._validate_execution(
                    user_request=goal,
                    execution_result=response,
                    enable_retry=False,  # Don't auto-retry complex plans
                    max_retries=0
                )

                if validation_result.get('validated'):
                    response += "\n\n🔍 **Validation**: ✅ Plan objectives met"
                else:
                    response += "\n\n🔍 **Validation**: ⚠️ Some objectives may not be fully met"
                    feedback = validation_result.get('feedback')
                    if feedback:
                        response += f"\n\n💡 **Review needed**:\n{feedback[:300]}"

            except Exception as e:
                self.logger.warning(f"Plan validation skipped: {e}")

            return response

        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}", exc_info=True)
            return f"❌ Plan execution failed: {str(e)}"

    def _execute_task(self, task: str) -> str:
        """
        Execute a task by delegating to appropriate agents.

        Args:
            task: Task description

        Returns:
            Execution result formatted for display
        """
        try:
            from agents.workflow_executor import OracleTaskRunner

            # Initialize task runner if not already
            if not hasattr(self, '_task_runner'):
                self._task_runner = OracleTaskRunner(self.model_manager)

            # Determine which agent to use based on task
            task_lower = task.lower()
            agent_used = "operator"  # Track which agent handles the task

            # Workflow/coordination tasks -> Coordinator
            if any(kw in task_lower for kw in ['workflow', 'coordinate', 'orchestrate', 'task queue',
                                                'agents status', 'registered agents', 'conflicts']):
                agent_used = "coordinator"
                self.logger.info(f"Delegating to Coordinator: {task[:50]}...")
                result = self._task_runner.execute("coordinator", task)

            # Learning/discovery/exploration -> Navigator
            elif any(kw in task_lower for kw in ['learn', 'how do', 'tutorial', 'guide',
                                                  'documentation', 'resources', 'explore',
                                                  'codebase', 'find code', 'similar', 'pattern']):
                agent_used = "navigator"
                self.logger.info(f"Delegating to Navigator: {task[:50]}...")
                result = self._task_runner.execute("navigator", task)

            # Security tasks -> Security then Operator
            elif any(kw in task_lower for kw in ['security', 'analyze', 'scan', 'binary',
                                                  'suspicious', 'safe', 'risk', 'vulnerability']):
                agent_used = "security → operator"
                self.logger.info(f"Delegating to Security+Operator: {task[:50]}...")
                result = self._task_runner.execute_with_security(task)

            # System tasks -> Operator
            elif any(kw in task_lower for kw in ['disk', 'space', 'process', 'memory', 'cpu',
                                                'file', 'list', 'show', 'check', 'status',
                                                'run', 'execute', 'command']):
                agent_used = "operator"
                self.logger.info(f"Delegating to Operator: {task[:50]}...")
                result = self._task_runner.execute("operator", task)

            # Default to Operator for general tasks
            else:
                agent_used = "operator"
                self.logger.info(f"Delegating to Operator (default): {task[:50]}...")
                result = self._task_runner.execute("operator", task)

            # Add agent info to result
            result['agent_used'] = agent_used

            # Format the response with terminal-friendly output
            if result.get('success'):
                agent_display = result.get('agent_used', 'operator').title()
                response_parts = [f"✓ {agent_display} completed the task\n"]

                # Add security analysis if present
                if result.get('security_analysis'):
                    response_parts.append(f"Security Check:\n{result['security_analysis'][:300]}\n")

                # Add main result - extract clean output from Operator's response
                main_result = result.get('result') or result.get('execution_result', '')
                if main_result:
                    # Try to extract just the command output, not the verbose planning
                    cleaned = self._clean_operator_output(main_result)
                    response_parts.append(cleaned)

                # Add timing
                if result.get('execution_time'):
                    response_parts.append(f"\n({result['execution_time']:.1f}s)")

                return '\n'.join(response_parts)
            else:
                agent_display = result.get('agent_used', 'agent').title()
                return f"✗ {agent_display} failed\n\nError: {result.get('error', 'Unknown error')}"

        except ImportError as e:
            self.logger.warning(f"OracleTaskRunner not available: {e}")
            return (
                "Oracle's task-runner is not available in this release.\n"
                "Use direct-agent mode for this kind of task — e.g.,  /agent operator  or  /agent summarizer."
            )
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return f"Task execution failed: {e}"

    def _clean_operator_output(self, raw_output: str) -> str:
        """
        Clean up Operator output for terminal display.

        Extracts actual command outputs and formats them cleanly.
        """
        lines = raw_output.split('\n')
        cleaned_lines = []
        in_code_block = False
        current_cmd = None

        for line in lines:
            # Skip verbose planning lines
            if line.startswith('**Plan**:') or line.startswith('**Insights**:'):
                continue
            if line.startswith('**Next**:'):
                continue

            # Track code blocks (actual output)
            if line.strip() == '```':
                in_code_block = not in_code_block
                continue

            # Extract command being executed
            if line.strip().startswith('- `') and '`' in line[4:]:
                cmd = line.strip()[3:].split('`')[0]
                current_cmd = cmd
                cleaned_lines.append(f"$ {cmd}")
                continue

            # Include actual output
            if in_code_block or (not line.startswith('**') and line.strip()):
                # Clean up the line
                clean_line = line.rstrip()
                if clean_line:
                    cleaned_lines.append(clean_line)

        return '\n'.join(cleaned_lines) if cleaned_lines else raw_output

    def _validate_execution(
        self,
        user_request: str,
        execution_result: str,
        request_id: str = None,
        enable_retry: bool = True,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Validate task execution and optionally retry with feedback.

        Phase B integration of validation loop into Console/Oracle.

        Args:
            user_request: Original user request
            execution_result: Result from agent execution
            request_id: Request ID for tracking
            enable_retry: Whether to retry on incomplete
            max_retries: Maximum retry attempts

        Returns:
            Dict with:
            - validated: True if complete
            - result: Final execution result
            - validation: Validation details
            - feedback: Feedback suggestions (if incomplete)
            - retries: Number of retries attempted
        """
        if request_id is None:
            request_id = f"val_{uuid_lib.uuid4().hex[:8]}"

        self.logger.info(f"[{request_id}] Validating execution result...")

        try:
            # Get validator agent (optional — not shipped in v0.2)
            validator = _get_validator_agent(self.model_manager)
            if validator is None:
                self.logger.info(f"[{request_id}] validator agent not available — skipping completion check")
                return {
                    "is_complete": True,
                    "validation_skipped": True,
                    "missing_items": [],
                    "feedback": "",
                }

            # Handle both string and dict execution_result formats
            if isinstance(execution_result, str):
                # Legacy format - create minimal dict from string
                execution_dict = {
                    'result': execution_result,
                    'success': '✅' in execution_result or 'completed' in execution_result.lower(),
                    'commands_executed': 1 if '✅' in execution_result else 0,
                    'successful_commands': 1 if '✅' in execution_result else 0,
                    'failed_commands': 1 if '❌' in execution_result else 0,
                    'execution_results': [],
                    'next_steps': ''
                }
            else:
                # Structured format - use as-is
                execution_dict = execution_result

            self.logger.debug(f"[{request_id}] Validator input: commands_executed={execution_dict.get('commands_executed', 0)}, "
                            f"execution_results={len(execution_dict.get('execution_results', []))}")

            # Validate completion
            validation = validator.validate_completion(
                user_request=user_request,
                execution_result=execution_dict,
                request_id=request_id
            )

            self.logger.info(
                f"[{request_id}] Validation: complete={validation.get('is_complete')}, "
                f"confidence={validation.get('confidence', 0):.2f}"
            )

            # If complete, return success
            if validation.get('is_complete', False):
                return {
                    'validated': True,
                    'result': execution_result,
                    'validation': validation,
                    'feedback': None,
                    'retries': 0
                }

            # Task incomplete - get feedback
            self.logger.info(f"[{request_id}] Task incomplete, generating feedback...")

            # Determine if this is a code task or sysadmin task
            agent_used = execution_dict.get('agent_used', '')
            is_sysadmin_task = agent_used in ('operator', 'security', 'knowledge')

            if is_sysadmin_task:
                # For sysadmin tasks, use Validator's suggestions directly
                # Don't use IntelligentFeedbackAgent (designed for code review)
                suggestions = validation.get('suggestions', [])
                missing = validation.get('missing_items', [])
                reasoning = validation.get('reasoning', '')

                feedback_parts = []
                if missing:
                    feedback_parts.append("**Missing items:**")
                    for item in missing:
                        feedback_parts.append(f"- {item}")
                if suggestions:
                    feedback_parts.append("\n**Suggestions:**")
                    for s in suggestions:
                        feedback_parts.append(f"- {s}")
                if reasoning:
                    feedback_parts.append(f"\n**Analysis:** {reasoning[:300]}")

                feedback_text = '\n'.join(feedback_parts) if feedback_parts else "Task appears incomplete."
                self.logger.debug(f"[{request_id}] Using validator feedback for sysadmin task")
            else:
                # For code tasks, use IntelligentFeedbackAgent (optional in v0.2)
                feedback_agent = _get_feedback_agent(self.model_manager)
                if feedback_agent is None:
                    feedback_text = '\n'.join(feedback_parts) if feedback_parts else "Task appears incomplete."
                    self.logger.debug(f"[{request_id}] feedback agent not available — using plain feedback")
                    return feedback_text

                # Build LogAnalysis for feedback agent
                from agents.models import LogAnalysis, Error

                errors = []
                for item in validation.get('missing_items', []):
                    errors.append(Error(
                        type="incomplete",
                        message=item,
                        file="",
                        line=0
                    ))

                analysis = LogAnalysis(
                    status="failed" if errors else "success",
                    errors=errors,
                    warnings=[],
                    raw_output=validation.get('reasoning', '')
                )

                feedback_result = feedback_agent.generate_feedback(
                    analysis=analysis,
                    cycle_number=1,
                    request_id=request_id
                )

                feedback_text = feedback_result.get('feedback', '')

            # If retry is enabled and we haven't exhausted retries
            if enable_retry and max_retries > 0:
                self.logger.info(f"[{request_id}] Attempting retry with feedback context...")

                # Build enhanced request with feedback
                enhanced_request = self._build_retry_request(
                    original_request=user_request,
                    validation=validation,
                    feedback=feedback_text
                )

                # Re-execute with enhanced context
                retry_result = self._execute_with_triage(enhanced_request)

                # Recursively validate (with one less retry)
                return self._validate_execution(
                    user_request=user_request,
                    execution_result=retry_result,
                    request_id=f"{request_id}_r1",
                    enable_retry=enable_retry,
                    max_retries=max_retries - 1
                )

            # No retry - return incomplete result with feedback
            # Add clear message about retry exhaustion
            if max_retries == 0:
                self.logger.info(f"[{request_id}] Max retries exhausted - returning partial results")

            return {
                'validated': False,
                'result': execution_result,
                'validation': validation,
                'feedback': feedback_text,
                'retries': 0,
                'max_retries_exhausted': max_retries == 0
            }

        except Exception as e:
            self.logger.error(f"[{request_id}] Validation failed: {e}", exc_info=True)
            return {
                'validated': False,
                'result': execution_result,
                'validation': {'error': str(e)},
                'feedback': None,
                'retries': 0,
                'error': str(e)
            }

    def _build_retry_request(
        self,
        original_request: str,
        validation: Dict[str, Any],
        feedback: str
    ) -> str:
        """
        Build enhanced request for retry based on validation and feedback.

        Args:
            original_request: Original user request
            validation: Validation result with missing items
            feedback: Feedback suggestions

        Returns:
            Enhanced request string
        """
        missing_items = validation.get('missing_items', [])
        suggestions = validation.get('suggestions', [])

        parts = [
            f"RETRY: {original_request}",
            "",
            "Previous attempt was incomplete. Please address:",
        ]

        if missing_items:
            for item in missing_items[:3]:  # Limit to top 3
                parts.append(f"- {item}")

        if suggestions:
            parts.append("\nSuggestions:")
            for sug in suggestions[:2]:  # Limit to top 2
                parts.append(f"- {sug}")

        if feedback:
            # Extract key points from feedback (first 200 chars)
            parts.append(f"\nFeedback: {feedback[:200]}...")

        return '\n'.join(parts)
