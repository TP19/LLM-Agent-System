#!/usr/bin/env python3
"""
Modular Executor Agent - Fully Migrated

This executor agent reasons freely and executes commands with basic safety filtering.
It collaborates with the security agent through reasoning exchange.
Uses BaseAgent and lazy loading.
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

from core.base_agent import BaseAgent

class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"

@dataclass
class ExecutionResult:
    command: str
    status: ExecutionStatus
    output: str
    exit_code: int
    execution_time: float
    reasoning: str
    timestamp: datetime

@dataclass
class CollaborationCycle:
    user_request: str
    security_suggestion: Dict
    executor_plan: str
    execution_results: List[ExecutionResult]
    next_steps: str
    cycle_number: int

class ModularExecutorAgent(BaseAgent):
    """Modular executor that collaborates with security agent"""

    def __init__(self, model_manager):
        super().__init__("executor", model_manager)

        self.prompt = self._create_reasoning_prompt()
        self.collaboration_history = []
        self.execution_history = []

        # ADD to existing BaseAgent stats, don't replace
        self.stats.update({
            'commands_executed': 0,
            'successful_executions': 0,
            'collaboration_cycles': 0,
            'total_execution_time': 0.0
        })

    def _create_reasoning_prompt(self) -> str:
        """Create free reasoning prompt"""
        return """You are an Executor Agent that helps users accomplish tasks through intelligent command execution.

YOUR APPROACH:
- Think step by step about what the user needs
- Plan a logical sequence of commands to gather info and solve the problem
- Execute commands and analyze results
- Adapt based on what you discover
- Collaborate with the Security Agent when helpful

REASONING PROCESS:
1. Understand what the user really wants to accomplish
2. Plan the most effective approach
3. Execute commands thoughtfully
4. Analyze results and adapt
5. Suggest next steps based on findings

Be practical, intelligent, and results-focused.

FORMAT:
UNDERSTANDING: [what the user wants to accomplish]
PLAN: [your approach to solving this]
COMMANDS: [specific commands to execute now]
REASONING: [why these commands will help]
EXPECTED: [what you expect to learn]

User Request: """

    def execute_task(self, user_request: str, security_suggestion: Dict = None, request_id: str = None) -> Dict[str, Any]:
        """Execute task with free reasoning and optional security collaboration"""

        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        self.logger.info(f"[{request_id}] Executing task: {user_request[:50]}...")

        start_time = time.time()
        collaboration_cycle = CollaborationCycle(
            user_request=user_request,
            security_suggestion=security_suggestion,
            executor_plan="",
            execution_results=[],
            next_steps="",
            cycle_number=len(self.collaboration_history) + 1
        )

        try:
            # 1. Create execution plan
            plan = self._create_execution_plan(user_request, security_suggestion, request_id)
            collaboration_cycle.executor_plan = plan['reasoning']

            # 2. Execute planned commands
            execution_results = self._execute_planned_commands(plan, request_id)
            collaboration_cycle.execution_results = execution_results

            # 3. Analyze results and plan next steps
            analysis = self._analyze_results(execution_results, user_request, request_id)
            collaboration_cycle.next_steps = analysis['next_steps']

            # 4. Store collaboration cycle
            self.collaboration_history.append(collaboration_cycle)

            # 5. Create summary
            processing_time = time.time() - start_time
            summary = self._create_task_summary(collaboration_cycle, processing_time)
            
            # Update stats
            self.update_stats(processing_time)
            self.stats['collaboration_cycles'] += 1

            self.logger.info(f"[{request_id}] Task completed in {processing_time:.2f}s")

            return summary

        except Exception as e:
            self.logger.error(f"[{request_id}] Task execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'user_request': user_request,
                'execution_time': time.time() - start_time
            }

    def _create_execution_plan(self, user_request: str, security_suggestion: Dict = None, request_id: str = None) -> Dict[str, Any]:
        """Create intelligent execution plan"""

        # If we have security suggestions, USE THEM DIRECTLY
        if security_suggestion and security_suggestion.get('commands'):
            self.logger.info(f"[{request_id}] Using security-suggested commands directly")
            
            return {
                'understanding': 'Using security agent suggestions',
                'plan': security_suggestion.get('approach', 'Execute suggested commands'),
                'commands': security_suggestion['commands'],  # Use directly!
                'reasoning': security_suggestion.get('reasoning', 'Security collaboration'),
                'expected': 'Execute security-validated commands'
            }

        # Only generate new plan if no security suggestions
        context = ""
        if self.collaboration_history:
            recent_cycle = self.collaboration_history[-1]
            context += f"\nRecent results: {[r.command for r in recent_cycle.execution_results]}\n"

        full_prompt = self.prompt + context + user_request

        try:
            response = self.generate_with_logging(
                full_prompt,
                request_id,
                max_tokens=600,
                temperature=0.4,
                top_p=0.9
            )

            plan = self._parse_execution_plan(response)
            
            # FALLBACK: If parsing gave us 0 commands, use fallback
            if not plan['commands']:
                self.logger.warning(f"[{request_id}] LLM response parsing failed, using fallback")
                return self._create_fallback_plan(user_request, security_suggestion)

            self.logger.info(f"[{request_id}] Created plan with {len(plan['commands'])} commands")
            self.logger.debug(f"[{request_id}] Commands: {plan['commands']}")

            return plan

        except Exception as e:
            self.logger.error(f"[{request_id}] Failed to create plan: {e}")
            return self._create_fallback_plan(user_request, security_suggestion)

    def _parse_execution_plan(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into execution plan"""
        import re

        try:
            # Extract understanding
            understanding_match = re.search(r'UNDERSTANDING:\s*(.+?)(?=\nPLAN:|$)', response, re.IGNORECASE | re.DOTALL)
            understanding = understanding_match.group(1).strip() if understanding_match else "Task analysis"

            # Extract plan
            plan_match = re.search(r'PLAN:\s*(.+?)(?=\nCOMMANDS:|$)', response, re.IGNORECASE | re.DOTALL)
            plan = plan_match.group(1).strip() if plan_match else "Execute commands to gather information"

            # Extract commands
            commands = []
            commands_match = re.search(r'COMMANDS:\s*(.+?)(?=\nREASONING:|$)', response, re.IGNORECASE | re.DOTALL)
            if commands_match:
                commands_text = commands_match.group(1)
                for line in commands_text.split('\n'):
                    line = line.strip()
                    if line:
                        cleaned_command = self._parse_and_clean_command(line)
                        if cleaned_command:
                            commands.append(cleaned_command)

            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\nEXPECTED:|$)', response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "Logical command sequence"

            # Extract expected outcomes
            expected_match = re.search(r'EXPECTED:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
            expected = expected_match.group(1).strip() if expected_match else "Information gathering"

            return {
                'understanding': understanding,
                'plan': plan,
                'commands': commands,
                'reasoning': reasoning,
                'expected': expected
            }

        except Exception as e:
            self.logger.error(f"Failed to parse plan: {e}")
            return {
                'understanding': 'Plan parsing failed',
                'plan': 'Execute basic commands',
                'commands': ['whoami', 'pwd'],
                'reasoning': 'Fallback due to parsing error',
                'expected': 'Basic system information'
            }

    def _parse_and_clean_command(self, raw_line: str) -> str:
        """Parse and clean a single command line"""
        import re

        # Remove numbering
        cleaned = re.sub(r'^\d+\.\s*', '', raw_line.strip())
        # Remove list markers
        cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
        
        # Extract from quotes/backticks
        quote_match = re.search(r'["\']([^"\']+)["\']', cleaned)
        if quote_match:
            cleaned = quote_match.group(1)
        backtick_match = re.search(r'`([^`]+)`', cleaned)
        if backtick_match:
            cleaned = backtick_match.group(1)

        cleaned = ' '.join(cleaned.split())

        if len(cleaned) < 3:
            return ""

        valid_commands = ['ssh', 'ls', 'df', 'du', 'ps', 'cat', 'grep', 'find', 'whoami', 'pwd', 'sudo', 'curl', 'wget', 'docker']
        if not any(cleaned.lower().startswith(cmd) for cmd in valid_commands):
            return ""

        return cleaned

    def _create_fallback_plan(self, user_request: str, security_suggestion: Dict = None) -> Dict[str, Any]:
        """Create fallback plan when LLM planning fails"""

        commands = []

        if security_suggestion and security_suggestion.get('commands'):
            commands = security_suggestion['commands'][:3]
        else:
            request_lower = user_request.lower()
            if 'disk' in request_lower or 'space' in request_lower:
                commands = ['df -h', 'du -sh /var /tmp']
            elif 'ssh' in request_lower:
                import re
                ssh_match = re.search(r'(\w+@[\d\.]+)', user_request)
                if ssh_match:
                    target = ssh_match.group(1)
                    commands = [f'ssh {target} "whoami"', f'ssh {target} "df -h"']
            else:
                commands = ['whoami', 'pwd', 'ls -la']

        return {
            'understanding': 'Fallback analysis of user request',
            'plan': 'Execute basic investigation commands',
            'commands': commands,
            'reasoning': 'Using fallback command selection',
            'expected': 'Basic system information'
        }

    def _execute_planned_commands(self, plan: Dict[str, Any], request_id: str) -> List[ExecutionResult]:
        """Execute planned commands individually"""

        commands = plan.get('commands', [])
        results = []

        self.logger.info(f"[{request_id}] Executing {len(commands)} commands individually")

        for i, command in enumerate(commands, 1):
            if not command or not command.strip():
                self.logger.warning(f"[{request_id}] Skipping empty command {i}")
                continue

            final_command = self._final_command_cleanup(command)

            if not final_command:
                self.logger.warning(f"[{request_id}] Command {i} failed validation: {command}")
                continue

            self.logger.info(f"[{request_id}] [{i}/{len(commands)}] Executing: {final_command}")

            if self._is_command_safe(final_command):
                result = self._execute_single_command(final_command, plan['reasoning'])
                results.append(result)
                self.execution_history.append(result)

                self.stats['commands_executed'] += 1
                if result.status == ExecutionStatus.SUCCESS:
                    self.stats['successful_executions'] += 1
                    self.logger.info(f"[{request_id}] [{i}/{len(commands)}] Success: {final_command}")
                else:
                    self.logger.warning(f"[{request_id}] [{i}/{len(commands)}] Failed: {final_command}")

                self.stats['total_execution_time'] += result.execution_time
                time.sleep(0.5)

            else:
                result = ExecutionResult(
                    command=final_command,
                    status=ExecutionStatus.BLOCKED,
                    output="Command blocked by safety filter",
                    exit_code=-1,
                    execution_time=0.0,
                    reasoning="Safety filter activation",
                    timestamp=datetime.now()
                )
                results.append(result)
                self.logger.warning(f"[{request_id}] Blocked unsafe command: {final_command}")

        self.logger.info(f"[{request_id}] Execution complete: {len(results)} commands processed")
        return results

    def _final_command_cleanup(self, command: str) -> str:
        """Final cleanup with improved SSH handling"""
        import re

        cleaned = command.strip()

        if cleaned.lower().startswith('ssh'):
            return self._fix_ssh_command_executor(cleaned)

        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]

        cleaned = cleaned.strip('`')
        cleaned = ' '.join(cleaned.split())

        return cleaned

    def _fix_ssh_command_executor(self, ssh_command: str) -> str:
        """Fix SSH command format"""
        import re

        ssh_match = re.match(r'ssh\s+(\w+@[\d\.]+)\s*(.*)$', ssh_command.strip())

        if not ssh_match:
            return ssh_command

        ssh_target = ssh_match.group(1)
        command_part = ssh_match.group(2).strip()

        if not command_part:
            return f"ssh {ssh_target}"

        # Clean quotes
        if command_part.startswith('"') and not command_part.endswith('"'):
            command_part = command_part[1:]
        elif command_part.startswith("'") and not command_part.endswith("'"):
            command_part = command_part[1:]
        elif command_part.endswith('"') and not command_part.startswith('"'):
            command_part = '"' + command_part
        elif command_part.endswith("'") and not command_part.startswith("'"):
            command_part = "'" + command_part

        if (command_part.startswith('"') and command_part.endswith('"')) or \
           (command_part.startswith("'") and command_part.endswith("'")):
            command_part = command_part[1:-1]

        if command_part:
            return f"ssh {ssh_target} '{command_part}'"
        else:
            return f"ssh {ssh_target}"

    def _is_command_safe(self, command: str) -> bool:
        """Enhanced safety check"""

        dangerous_patterns = [
            'rm -rf /',
            'dd if=',
            'mkfs',
            'format',
            'shutdown -h',
            'reboot',
            '> /dev/sd',
            'chmod 777 /',
            'passwd',
            'userdel',
            'deluser'
        ]

        cmd_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern in cmd_lower:
                return False

        if cmd_lower.startswith('ssh'):
            return self._validate_ssh_command(command)

        return True

    def _validate_ssh_command(self, ssh_command: str) -> bool:
        """Validate SSH command structure"""
        import re

        ssh_pattern = r'ssh\s+\w+@[\d\.]+(?:\s+[\'"][^\'"]*[\'"])?$'

        if not re.match(ssh_pattern, ssh_command):
            self.logger.warning(f"SSH command doesn't match expected pattern: {ssh_command}")
            return False

        dangerous_ssh_commands = [
            'rm -rf',
            'dd if=',
            'mkfs',
            'format',
            'shutdown',
            'reboot',
            'passwd',
            'userdel'
        ]

        for dangerous in dangerous_ssh_commands:
            if dangerous in ssh_command.lower():
                self.logger.warning(f"Dangerous SSH command blocked: {ssh_command}")
                return False

        return True

    def _execute_single_command(self, command: str, reasoning: str) -> ExecutionResult:
        """Execute a single command and capture results"""

        start_time = time.time()

        try:
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            execution_time = time.time() - start_time
            status = ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED
            output = process.stdout + process.stderr

            return ExecutionResult(
                command=command,
                status=status,
                output=output,
                exit_code=process.returncode,
                execution_time=execution_time,
                reasoning=reasoning,
                timestamp=datetime.now()
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                command=command,
                status=ExecutionStatus.TIMEOUT,
                output="Command timed out after 30 seconds",
                exit_code=-1,
                execution_time=30.0,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
        except Exception as e:
            return ExecutionResult(
                command=command,
                status=ExecutionStatus.FAILED,
                output=f"Execution error: {str(e)}",
                exit_code=-1,
                execution_time=time.time() - start_time,
                reasoning=reasoning,
                timestamp=datetime.now()
            )

    def _analyze_results(self, results: List[ExecutionResult], user_request: str, request_id: str) -> Dict[str, Any]:
        """Analyze execution results and plan next steps"""

        results_context = "EXECUTION RESULTS:\n"
        for result in results:
            status_icon = "SUCCESS" if result.status == ExecutionStatus.SUCCESS else "FAILED"
            results_context += f"{status_icon} {result.command}\n"
            if result.output:
                results_context += f"   Output: {result.output[:150]}...\n"

        prompt = f"""Analyze these execution results and determine next steps:

ORIGINAL REQUEST: {user_request}

{results_context}

Based on these results, what should we do next to help the user?
Be specific and practical.

FORMAT:
ANALYSIS: [what the results tell us]
INSIGHTS: [key findings from the output]
NEXT_STEPS: [specific actions to take next]
COMPLETION: [are we done, or do we need more work?]
"""

        try:
            response = self.generate_with_logging(
                prompt,
                request_id,
                max_tokens=400,
                temperature=0.3,
                top_p=0.9
            )

            return self._parse_analysis(response)

        except Exception as e:
            self.logger.error(f"[{request_id}] Failed to analyze results: {e}")
            return {
                'analysis': 'Analysis failed',
                'insights': 'Unable to process results',
                'next_steps': 'Review outputs manually',
                'completion': 'incomplete'
            }

    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """Parse analysis response"""
        import re

        try:
            analysis_match = re.search(r'ANALYSIS:\s*(.+?)(?=\nINSIGHTS:|$)', response, re.IGNORECASE | re.DOTALL)
            analysis = analysis_match.group(1).strip() if analysis_match else "Results processed"

            insights_match = re.search(r'INSIGHTS:\s*(.+?)(?=\nNEXT_STEPS:|$)', response, re.IGNORECASE | re.DOTALL)
            insights = insights_match.group(1).strip() if insights_match else "Information gathered"

            next_steps_match = re.search(r'NEXT_STEPS:\s*(.+?)(?=\nCOMPLETION:|$)', response, re.IGNORECASE | re.DOTALL)
            next_steps = next_steps_match.group(1).strip() if next_steps_match else "Continue investigation"

            completion_match = re.search(r'COMPLETION:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
            completion = completion_match.group(1).strip() if completion_match else "in_progress"

            return {
                'analysis': analysis,
                'insights': insights,
                'next_steps': next_steps,
                'completion': completion
            }

        except Exception as e:
            self.logger.error(f"Failed to parse analysis: {e}")
            return {
                'analysis': 'Parse error',
                'insights': 'Could not process',
                'next_steps': 'Manual review needed',
                'completion': 'error'
            }

    def _create_task_summary(self, cycle: CollaborationCycle, total_time: float) -> Dict[str, Any]:
        """Create comprehensive task summary"""

        successful_commands = [r for r in cycle.execution_results if r.status == ExecutionStatus.SUCCESS]
        failed_commands = [r for r in cycle.execution_results if r.status == ExecutionStatus.FAILED]

        return {
            'status': 'completed',
            'user_request': cycle.user_request,
            'cycle_number': cycle.cycle_number,
            'total_time': total_time,
            'executor_plan': cycle.executor_plan,
            'commands_executed': len(cycle.execution_results),
            'successful_commands': len(successful_commands),
            'failed_commands': len(failed_commands),
            'execution_results': [
                {
                    'command': r.command,
                    'status': r.status.value,
                    'output': r.output,
                    'execution_time': r.execution_time
                } for r in cycle.execution_results
            ],
            'next_steps': cycle.next_steps,
            'security_collaboration': cycle.security_suggestion is not None,
            'insights': self._extract_key_insights(cycle.execution_results),
            'stats': self.get_stats()
        }

    def _extract_key_insights(self, results: List[ExecutionResult]) -> List[str]:
        """Extract key insights from execution results"""
        insights = []

        for result in results:
            if result.status == ExecutionStatus.SUCCESS and result.output:
                output = result.output.strip()
                if output:
                    first_line = output.split('\n')[0]
                    if len(first_line) > 10:
                        insights.append(f"{result.command}: {first_line[:100]}")

        return insights

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        base_stats = super().get_stats()
        return {
            **base_stats,
            'commands_executed': self.stats['commands_executed'],
            'successful_executions': self.stats['successful_executions'],
            'collaboration_cycles': self.stats['collaboration_cycles'],
            'total_execution_time': self.stats['total_execution_time'],
            'executor_success_rate': self.stats['successful_executions'] / max(1, self.stats['commands_executed']),
            'avg_command_time': self.stats['total_execution_time'] / max(1, self.stats['commands_executed'])
        }