#!/usr/bin/env python3
"""
Modular Executor Agent - Fully Migrated

This operator agent reasons freely and executes commands with basic safety filtering.
It collaborates with the security agent through reasoning exchange.
Uses BaseAgent and lazy loading.
"""

import subprocess
import json
import time
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

from core.base_agent import BaseAgent
from core.retry import retry

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

class ModularOperatorAgent(BaseAgent):
    """Operator agent that collaborates with Security agent"""

    def __init__(self, model_manager, interactive: bool = False, checkpoint_callback = None):
        super().__init__("operator", model_manager, interactive=interactive, checkpoint_callback=checkpoint_callback)

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
        """Create free reasoning prompt with host context"""
        # Get host context for better remote command understanding
        try:
            hostname = socket.gethostname()
            # Get primary IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            host_ip = s.getsockname()[0]
            s.close()
        except Exception:
            hostname = "unknown"
            host_ip = "unknown"

        return f"""You are an Executor Agent that helps users accomplish tasks through intelligent command execution.

HOST CONTEXT:
You are running on: {hostname} ({host_ip})
Commands you execute run LOCALLY on this machine unless you use SSH.
- If user says "on host1" or "on host2" etc., use SSH to run commands on that host
- Example: "check disk on host1" -> ssh host1 df -h
- Direct commands without host reference run locally on {hostname}

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

    def execute_task_interactive(self, user_request: str, security_result: Dict, max_cycles: int = 5) -> Dict[str, Any]:
        """
        Interactive mode execution - normalizes security_result keys and executes commands

        Args:
            user_request: The user's request
            security_result: Results from Security agent (with 'suggested_commands' key)
            max_cycles: Maximum execution cycles (currently ignored, uses single cycle)

        Returns:
            Dict with execution results in interactive mode format
        """
        # Normalize keys: suggested_commands -> commands for backward compatibility
        security_suggestion = dict(security_result)  # Copy to avoid modifying original
        if 'suggested_commands' in security_suggestion:
            security_suggestion['commands'] = security_suggestion.pop('suggested_commands')

        # Execute using the standard execute_task method
        result = self.execute_task(user_request, security_suggestion=security_suggestion)

        # Transform result to interactive mode format
        # Count total commands executed across all cycles
        execution_results = result.get('execution_results', [])
        total_commands = len(execution_results)

        # Extract command strings for downstream consumers
        executed_commands = [r.command if hasattr(r, 'command') else r.get('command', '')
                           for r in execution_results]

        # Convert ExecutionResult objects to dicts for command_results
        command_results = []
        for r in execution_results:
            if hasattr(r, 'command'):
                command_results.append({
                    'command': r.command,
                    'status': r.status.value if hasattr(r.status, 'value') else str(r.status),
                    'output': r.output,
                    'execution_time': r.execution_time
                })
            else:
                command_results.append(r)

        # Determine actual completion status based on execution results
        # Count failed commands to inform completion check
        failed_count = sum(1 for r in command_results
                          if r.get('status') in ('failed', 'FAILED', 'error', 'ERROR'))
        success_count = sum(1 for r in command_results
                           if r.get('status') in ('success', 'SUCCESS', 'completed', 'COMPLETED'))

        # Caller decides is_complete based on these counts
        # We just report execution status, not completion status
        has_errors = failed_count > 0 or result.get('status') == 'error'

        return {
            'success': not has_errors and success_count > 0,
            'cycles_completed': 1,  # Currently single cycle
            'total_commands': total_commands,
            'commands_executed': total_commands,
            'successful_commands': success_count,
            'failed_commands': failed_count,
            'execution_results': execution_results,
            'executed_commands': executed_commands,
            'command_results': command_results,
            'analysis': result.get('analysis', {}),
            'next_steps': result.get('next_steps', ''),
            'execution_time': result.get('execution_time', 0)
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

        # Check if user request looks like a direct command
        direct_command = self._detect_direct_command(user_request)
        if direct_command:
            self.logger.info(f"[{request_id}] Detected direct command, executing verbatim")
            return {
                'understanding': 'Direct command detected',
                'plan': 'Execute user-specified command directly',
                'commands': [direct_command],
                'reasoning': 'User provided an explicit command to execute',
                'expected': 'Command output'
            }

        # Only generate new plan if no direct command detected
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
            # Helper to create flexible section pattern
            # Handles: SECTION:, **SECTION**, # SECTION, SECTION -
            def section_pattern(name: str, next_sections: list) -> str:
                # Match section header with optional markdown formatting
                header = rf'(?:\*\*{name}\*\*|#{1,2}\s*{name}|{name})\s*[:—\-]?\s*'
                # Match until next section or end
                if next_sections:
                    next_pattern = '|'.join(rf'\*\*{s}\*\*|#{1,2}\s*{s}|{s}\s*[:—\-]' for s in next_sections)
                    return rf'{header}(.+?)(?=\n(?:{next_pattern})|$)'
                else:
                    # No next sections - match until end
                    return rf'{header}(.+?)$'

            # Extract understanding
            understanding_match = re.search(
                section_pattern('UNDERSTANDING', ['PLAN', 'COMMANDS']),
                response, re.IGNORECASE | re.DOTALL
            )
            understanding = understanding_match.group(1).strip() if understanding_match else "Task analysis"

            # Extract plan
            plan_match = re.search(
                section_pattern('PLAN', ['COMMANDS', 'REASONING']),
                response, re.IGNORECASE | re.DOTALL
            )
            plan = plan_match.group(1).strip() if plan_match else "Execute commands to gather information"

            # Extract commands - try multiple methods
            commands = []
            commands_match = re.search(
                section_pattern('COMMANDS', ['REASONING', 'EXPECTED']),
                response, re.IGNORECASE | re.DOTALL
            )
            if commands_match:
                commands_text = commands_match.group(1)
                for line in commands_text.split('\n'):
                    line = line.strip()
                    if line:
                        cleaned_command = self._parse_and_clean_command(line)
                        if cleaned_command:
                            commands.append(cleaned_command)

            # FALLBACK: If no commands found via section headers, try extracting from raw response
            if not commands:
                commands = self._extract_commands_from_raw(response)
                if commands:
                    self.logger.info(f"Extracted {len(commands)} commands from raw LLM response")

            # Extract reasoning
            reasoning_match = re.search(
                section_pattern('REASONING', ['EXPECTED']),
                response, re.IGNORECASE | re.DOTALL
            )
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "Logical command sequence"

            # Extract expected outcomes
            expected_match = re.search(
                section_pattern('EXPECTED', []),
                response, re.IGNORECASE | re.DOTALL
            )
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
            # Return empty commands to trigger fallback in caller
            return {
                'understanding': 'Plan parsing failed',
                'plan': 'Use fallback',
                'commands': [],  # Empty to trigger _create_fallback_plan
                'reasoning': 'Parsing error',
                'expected': 'Basic information'
            }

    def _extract_commands_from_raw(self, response: str) -> List[str]:
        """
        Extract commands from raw LLM response when section parsing fails.

        Looks for:
        - Backtick code blocks: ```bash ... ``` or `command`
        - Shell prefixes: $ command or > command
        - Numbered/bulleted lists with commands
        - SSH patterns: ssh host command
        - Common command patterns
        """
        import re
        commands = []

        # 1. Extract from code blocks (```bash ... ```)
        code_blocks = re.findall(r'```(?:bash|shell|sh)?\s*\n?(.*?)```', response, re.DOTALL | re.IGNORECASE)
        for block in code_blocks:
            for line in block.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    cleaned = self._parse_and_clean_command(line)
                    if cleaned:
                        commands.append(cleaned)

        # 2. Extract inline backtick commands (`command`)
        inline_commands = re.findall(r'`([^`]+)`', response)
        for cmd in inline_commands:
            cmd = cmd.strip()
            # Only take if it looks like a command (starts with known command or contains command chars)
            if self._looks_like_command(cmd):
                cleaned = self._parse_and_clean_command(cmd)
                if cleaned:
                    commands.append(cleaned)

        # 3. Extract shell-prefixed commands ($ command or > command)
        shell_prefixed = re.findall(r'^[\$>]\s*(.+)$', response, re.MULTILINE)
        for cmd in shell_prefixed:
            cleaned = self._parse_and_clean_command(cmd)
            if cleaned:
                commands.append(cleaned)

        # 4. Extract SSH commands mentioned in text
        ssh_patterns = re.findall(r'\bssh\s+[\w@\.\-]+(?:\s+[^\n]+)?', response, re.IGNORECASE)
        for cmd in ssh_patterns:
            cleaned = self._parse_and_clean_command(cmd)
            if cleaned and cleaned not in commands:
                commands.append(cleaned)

        # 5. Extract commands from numbered/bulleted lists
        list_items = re.findall(r'^(?:\d+[\.\)]\s*|\-\s+|\*\s+)(.+)$', response, re.MULTILINE)
        for item in list_items:
            if self._looks_like_command(item):
                cleaned = self._parse_and_clean_command(item)
                if cleaned and cleaned not in commands:
                    commands.append(cleaned)

        # Deduplicate while preserving order
        seen = set()
        unique_commands = []
        for cmd in commands:
            if cmd not in seen:
                seen.add(cmd)
                unique_commands.append(cmd)

        return unique_commands[:5]  # Limit to 5 commands max

    def _looks_like_command(self, text: str) -> bool:
        """Check if text looks like a shell command"""
        import re

        # Known command prefixes
        command_starters = [
            'ssh', 'scp', 'rsync', 'ls', 'cat', 'head', 'tail', 'df', 'du', 'ps',
            'docker', 'git', 'python', 'pip', 'npm', 'node', 'curl', 'wget',
            'grep', 'find', 'awk', 'sed', 'chmod', 'chown', 'mkdir', 'rm', 'cp', 'mv',
            'nvidia-smi', 'nvtop', 'top', 'htop', 'free', 'uptime', 'uname',
            'systemctl', 'journalctl', 'kubectl', 'terraform', 'ansible'
        ]

        text_lower = text.lower().strip()

        # Check if starts with known command
        for cmd in command_starters:
            if text_lower.startswith(cmd + ' ') or text_lower == cmd:
                return True

        # Check for path-like patterns or flag patterns
        if re.match(r'^[a-z_][a-z0-9_-]*\s+', text_lower):
            if '-' in text or '/' in text or '|' in text:
                return True

        return False

    def _detect_direct_command(self, user_request: str) -> str:
        """
        Detect if the user request is a direct command to execute.

        Returns the command string if detected, otherwise None.

        Examples that should be detected as direct commands:
        - "ssh host1 docker ps"
        - "ssh host1 docker logs container_name --tail 20"
        - "df -h"
        - "ls -la /tmp"
        - "docker ps -a"
        - "git status"

        Examples that should NOT be detected (natural language):
        - "check disk space"
        - "show me the running containers"
        - "what's in /tmp?"
        - "debug why malguard is restarting"
        """
        import re

        request = user_request.strip()

        # Skip empty or very short requests
        if len(request) < 2:
            return None

        # Skip retry prefixes - these need LLM interpretation
        if request.upper().startswith('RETRY:'):
            return None

        # Known command prefixes that indicate a direct command
        direct_command_prefixes = [
            'ssh', 'scp', 'rsync', 'sftp',  # Remote commands
            'ls', 'cat', 'head', 'tail', 'less', 'more',  # File viewing
            'df', 'du', 'ps', 'top', 'htop', 'free', 'uptime',  # System monitoring
            'grep', 'find', 'locate', 'which', 'whereis',  # Search commands
            'docker', 'git', 'npm', 'python', 'pip', 'cd', 'pwd', 'whoami',
            'mkdir', 'rm', 'cp', 'mv', 'touch', 'chmod', 'chown',
            'systemctl', 'service', 'journalctl',
            'curl', 'wget', 'ping', 'nslookup', 'dig', 'traceroute',
            'uname', 'hostname', 'id', 'groups',
            'tar', 'zip', 'unzip', 'gzip', 'gunzip',
            'awk', 'sed', 'sort', 'uniq', 'wc', 'cut',
            'netstat', 'ss', 'ifconfig', 'ip',
            'apt', 'apt-get', 'yum', 'dnf', 'pacman',
            'nvidia-smi', 'lspci', 'lsusb', 'lsblk',
            'echo', 'printf', 'env', 'export', 'source',
            'crontab', 'at', 'nohup', 'screen', 'tmux',
            'file', 'stat', 'type', 'man', 'info',
            'kubectl', 'helm', 'podman',  # Container orchestration
        ]

        # Conversational indicators that mean this is NOT a direct command
        conversational_indicators = [
            ' please ', ' can you ', ' would you ', ' could you ',
            ' show me ', ' tell me ', ' help me ', ' i want ', ' i need ',
            ' what is ', ' what are ', ' how do ', ' how to ',
            ' why is ', ' why does ', ' debug ', ' investigate ',
            ' fix ', ' resolve ', ' troubleshoot ',
            '?',  # Questions are usually not direct commands
        ]

        request_lower = ' ' + request.lower() + ' '  # Add spaces for word boundary matching

        # Check for conversational indicators first
        for indicator in conversational_indicators:
            if indicator in request_lower:
                self.logger.debug(f"Conversational indicator '{indicator.strip()}' found, not a direct command")
                return None

        # Check if starts with a known command
        first_word = request.split()[0].lower() if request.split() else ""

        if first_word in direct_command_prefixes:
            # Special handling for SSH commands with remote commands
            # ssh <host> <remote_command> should execute verbatim
            if first_word == 'ssh':
                # Check if this looks like: ssh [options] host [command]
                # Match patterns like: ssh host cmd, ssh -p 22 host cmd, ssh user@host cmd
                ssh_with_cmd = re.match(
                    r'^ssh\s+(?:-\w+\s+)*[\w@\.\-]+\s+.+',
                    request,
                    re.IGNORECASE
                )
                if ssh_with_cmd:
                    self.logger.info(f"Detected SSH command with remote command: {request}")
                    return request

            # For non-SSH commands, just return as direct command
            self.logger.debug(f"Detected direct command: {request}")
            return request

        # Also check for commands with path prefixes
        # e.g., /usr/bin/python, ./script.sh, ~/bin/tool
        if re.match(r'^[./~]', request) and ' ' in request:
            # Starts with path indicator and has arguments
            self.logger.debug(f"Detected path-based command: {request}")
            return request

        # Not a direct command
        return None

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

        # BLACKLIST: Block truly dangerous commands instead of whitelist
        # This gives agents freedom while maintaining safety
        dangerous_patterns = [
            r'^rm\s+-rf\s+/\s*$',              # rm -rf / (root deletion)
            r'^rm\s+-rf\s+/[^/\s]*\s*$',       # rm -rf /anything (top-level deletion)
            r':\(\)\{.*\}',                     # Fork bomb
            r'>\s*/dev/sd[a-z]',                # Direct disk write
            r'dd\s+if=.*of=/dev/(sd|hd|nvme)', # DD to physical disk
            r'^mkfs',                           # Format filesystem
            r'^fdisk',                          # Partition manipulation
            r'^parted',                         # Partition manipulation
            r'curl.*\|\s*(bash|sh)',            # Pipe curl to shell (common attack)
            r'wget.*\|\s*(bash|sh)',            # Pipe wget to shell
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                self.logger.warning(f"🛡️ Blocked potentially dangerous command: {cleaned}")
                return ""

        # Log non-standard commands for monitoring (informational only)
        common_commands = ['ssh', 'ls', 'df', 'du', 'ps', 'cat', 'grep', 'find', 'whoami', 'pwd', 'docker', 'git']
        if not any(cleaned.lower().startswith(cmd) for cmd in common_commands):
            self.logger.info(f"🔍 Allowing non-standard command: {cleaned}")

        return cleaned

    def _create_fallback_plan(self, user_request: str, security_suggestion: Dict = None) -> Dict[str, Any]:
        """Create fallback plan when LLM planning fails"""
        import re
        import socket

        commands = []
        remote_hosts = []
        include_local = False

        # Get current hostname for "this system" detection
        try:
            current_host = socket.gethostname().lower()
        except Exception:
            current_host = "localhost"

        # Detect ALL remote host references in natural language
        # Start by detecting IP addresses and user@IP patterns (highest priority)
        ssh_targets: Dict[str, str] = {}  # hostname -> ssh_target (user@host or just host)

        # Pattern 0a: user@IP format (e.g., "user@192.0.2.10")
        user_ip_pattern = r'(\w+)@((?:\d{1,3}\.){3}\d{1,3})'
        user_ip_matches = re.findall(user_ip_pattern, user_request)
        for user, ip in user_ip_matches:
            ssh_target = f'{user}@{ip}'
            # Use IP as key to avoid duplicate hosts
            if ip not in ssh_targets:
                remote_hosts.append(ip)
                ssh_targets[ip] = ssh_target
                self.logger.info(f"Detected user@IP target: {ssh_target}")

        # Pattern 0b: standalone IP addresses (e.g., "192.0.2.10")
        ip_pattern = r'(?<!\d)((?:\d{1,3}\.){3}\d{1,3})(?!\d)'
        ip_matches = re.findall(ip_pattern, user_request)
        for ip in ip_matches:
            if ip not in ssh_targets:
                remote_hosts.append(ip)
                ssh_targets[ip] = ip  # Just IP, no user
                self.logger.info(f"Detected IP address: {ip}")

        # Known valid hostnames (whitelist approach to prevent false positives).
        # Extend this set with your own SSH hosts, or override via _known_hosts.
        known_hosts = {'localhost', 'host1', 'host2', 'remote'}

        # Pattern 1: "on host1", "on host2", "from host1", "at host2"
        host_pattern = r'\b(?:on|from|at)\s+([a-zA-Z][a-zA-Z0-9_-]*)\b'
        host_matches = re.findall(host_pattern, user_request, re.IGNORECASE)

        # Pattern 2: "and host1", "and host2" (for "this system and host1" patterns)
        and_pattern = r'\band\s+([a-zA-Z][a-zA-Z0-9_-]*)\b'
        and_matches = re.findall(and_pattern, user_request, re.IGNORECASE)
        host_matches.extend(and_matches)

        # Only accept known hostnames to prevent false positives from common words
        for potential_host in host_matches:
            potential_host_lower = potential_host.lower()
            if potential_host_lower in known_hosts:
                if potential_host_lower not in remote_hosts and potential_host_lower not in ssh_targets:
                    remote_hosts.append(potential_host_lower)
                    ssh_targets[potential_host_lower] = potential_host_lower

        # Check for "this system" or local references
        if re.search(r'\b(this\s+system|locally|local|here)\b', user_request, re.IGNORECASE):
            include_local = True

        # Log detected hosts
        if remote_hosts:
            self.logger.info(f"Detected remote host references: {remote_hosts}")
        if include_local:
            self.logger.info("Detected local/this system reference")

        # Determine base commands based on request type
        request_lower = user_request.lower()
        base_cmds = []

        # Detect action verbs to determine appropriate commands
        create_actions = ['spin up', 'create', 'run', 'start', 'launch', 'deploy']
        check_actions = ['check', 'list', 'show', 'status', 'see', 'view', 'get']
        stop_actions = ['stop', 'kill', 'terminate', 'shutdown', 'halt']
        remove_actions = ['remove', 'delete', 'destroy', 'rm']

        # Detect which action type
        has_create_action = any(action in request_lower for action in create_actions)
        has_check_action = any(action in request_lower for action in check_actions)
        has_stop_action = any(action in request_lower for action in stop_actions)
        has_remove_action = any(action in request_lower for action in remove_actions)

        # Web URL detection - use curl/wget
        url_match = re.search(r'(https?://[^\s]+)', user_request)
        if url_match:
            target_url = url_match.group(1)
            self.logger.info(f"Web URL detected: {target_url}")
            # Extract hostname from URL to exclude from SSH targets
            url_host_match = re.search(r'https?://([^:/\s]+)', target_url)
            if url_host_match:
                url_hostname = url_host_match.group(1).split('.')[-2] if '.' in url_host_match.group(1) else url_host_match.group(1)
                # Remove URL hostname from remote_hosts to prevent SSH attempts
                remote_hosts = [h for h in remote_hosts if h != url_hostname]
                self.logger.info(f"Excluded {url_hostname} from SSH targets (is URL)")

            # Fetch the URL
            base_cmds = [f'curl -s {target_url}']
            remote_hosts = []
            include_local = True
        elif 'disk' in request_lower or 'space' in request_lower:
            base_cmds = ['df -h']
        elif 'resource' in request_lower or 'memory' in request_lower or 'cpu' in request_lower:
            base_cmds = ['free -h', 'uptime', 'df -h']
        elif 'gpu' in request_lower or 'nvidia' in request_lower:
            base_cmds = ['nvidia-smi']
        elif 'process' in request_lower or 'running' in request_lower:
            base_cmds = ['ps aux | head -20']
        elif 'docker' in request_lower or 'container' in request_lower:
            # Action-based docker command selection
            if has_create_action:
                # Extract image name if provided
                image_match = re.search(r'(?:image|with|using)\s+(\S+)', request_lower)
                image = image_match.group(1) if image_match else 'ubuntu:latest'
                # Build docker run command
                docker_cmd = f'docker run -d --name auto_container {image}'
                # Add common flags if mentioned
                if 'persistent' in request_lower or 'volume' in request_lower:
                    docker_cmd = docker_cmd.replace('docker run', 'docker run -v /tmp/data:/data')
                if 'interactive' in request_lower:
                    docker_cmd = docker_cmd.replace('-d', '-it')
                base_cmds = [docker_cmd]
                self.logger.info(f"Docker CREATE action detected, command: {docker_cmd}")
            elif has_stop_action:
                # Try to extract container name
                container_match = re.search(r'(?:container|named?)\s+(\S+)', request_lower)
                container = container_match.group(1) if container_match else ''
                if container:
                    base_cmds = [f'docker stop {container}']
                else:
                    base_cmds = ['docker ps', 'echo "Specify container name to stop"']
                self.logger.info(f"Docker STOP action detected")
            elif has_remove_action:
                container_match = re.search(r'(?:container|named?)\s+(\S+)', request_lower)
                container = container_match.group(1) if container_match else ''
                if container:
                    base_cmds = [f'docker rm -f {container}']
                else:
                    base_cmds = ['docker ps -a', 'echo "Specify container name to remove"']
                self.logger.info(f"Docker REMOVE action detected")
            else:
                # Default to check/list action
                base_cmds = ['docker ps']
                if 'all' in request_lower or 'stopped' in request_lower:
                    base_cmds = ['docker ps -a']
                self.logger.info(f"Docker CHECK action detected (default)")
        else:
            base_cmds = ['uptime', 'df -h']

        if security_suggestion and security_suggestion.get('commands'):
            base_cmds = security_suggestion['commands'][:3]

        # Generate commands for all hosts
        if remote_hosts or include_local:
            # Add local commands if requested
            if include_local:
                for cmd in base_cmds:
                    commands.append(cmd)

            # Add remote commands for each host (use ssh_targets for proper user@host format)
            for host in remote_hosts:
                ssh_target = ssh_targets.get(host, host)  # Use user@IP if available, else just host
                for cmd in base_cmds:
                    if not cmd.startswith('ssh '):
                        # Quote commands with special chars for remote execution
                        safe_cmd = cmd.replace('"', '\\"')
                        commands.append(f'ssh {ssh_target} "{safe_cmd}"')
                    else:
                        commands.append(cmd)
        elif not remote_hosts and not include_local:
            # No hosts mentioned - run locally
            commands = base_cmds

        # Explicit SSH command handling
        if 'ssh' in request_lower and not commands:
            ssh_match = re.search(r'(\w+@[\d\.]+)', user_request)
            if ssh_match:
                target = ssh_match.group(1)
                commands = [f'ssh {target} "whoami"', f'ssh {target} "df -h"']

        # Ultimate fallback
        if not commands:
            commands = ['whoami', 'pwd', 'uptime']

        # Generate plan description
        if len(remote_hosts) > 1:
            plan_desc = f'Execute commands on {", ".join(remote_hosts)}'
        elif len(remote_hosts) == 1:
            plan_desc = f'Execute commands on {remote_hosts[0]}'
        elif include_local:
            plan_desc = 'Execute commands locally'
        else:
            plan_desc = 'Execute basic commands'

        return {
            'understanding': 'Fallback analysis of user request',
            'plan': plan_desc,
            'commands': commands,
            'reasoning': 'Using fallback command selection',
            'expected': 'System information from specified hosts'
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

        # More flexible SSH pattern that allows:
        # - Hostnames (host1, host2) or user@host
        # - IP addresses
        # - Commands with or without quotes
        # - Complex command arguments
        ssh_patterns = [
            r'^ssh\s+[\w@\.\-]+(?:\s+.+)?$',  # ssh host [command]
            r'^ssh\s+-\w+\s+[\w@\.\-]+(?:\s+.+)?$',  # ssh -options host [command]
        ]

        if not any(re.match(p, ssh_command) for p in ssh_patterns):
            self.logger.warning(f"SSH command doesn't match expected pattern: {ssh_command}")
            return False

        dangerous_ssh_commands = [
            'rm -rf /',      # Only block root deletion
            'rm -rf /*',
            'dd if=',
            'mkfs',
            'format c:',
            'shutdown -h now',
            'reboot',
            'passwd',
            'userdel',
            ':(){ :|:& };:',  # Fork bomb
        ]

        cmd_lower = ssh_command.lower()
        for dangerous in dangerous_ssh_commands:
            if dangerous in cmd_lower:
                self.logger.warning(f"Dangerous SSH command blocked: {ssh_command}")
                return False

        return True

    def _execute_single_command(self, command: str, reasoning: str) -> ExecutionResult:
        """Execute a single command and capture results"""

        start_time = time.time()
        is_ssh_command = command.strip().startswith('ssh ')

        try:
            # SSH commands get retry logic for network resilience
            if is_ssh_command:
                process = self._run_ssh_command_with_retry(command)
            else:
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

    @retry(
        max_attempts=2,
        backoff_strategy="linear",
        base_delay=1.0,
        max_delay=5.0,
        exceptions=(subprocess.SubprocessError, ConnectionError, OSError)
    )
    def _run_ssh_command_with_retry(self, command: str):
        """Run SSH command with automatic retry on network failures"""
        return subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
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

        # Determine actual status based on execution results
        # Caller decides completion from these results
        if len(failed_commands) > 0:
            status = 'partial'  # Some commands failed
        elif len(successful_commands) == 0:
            status = 'no_commands'  # No commands were executed
        else:
            status = 'executed'  # All commands executed successfully

        return {
            'status': status,
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
    def chat(self, message: str, history: list = None) -> str:
        """
        Chat interface for Operator Agent.

        Handles conversational requests, executing commands when appropriate.

        Args:
            message: User message
            history: Conversation history (optional)

        Returns:
            Response string with execution results or analysis
        """
        # Check if this looks like a task/command request
        task_keywords = ['run', 'execute', 'check', 'show', 'list', 'find', 'get', 'ssh',
                        'ping', 'look', 'scan', 'analyze', 'tell me', 'what is', 'how much']
        is_task = any(kw in message.lower() for kw in task_keywords)

        if is_task:
            # Execute as a task
            try:
                result = self.execute_task(message)

                if result.get('status') == 'error':
                    return f"Error: {result.get('error', 'Unknown error')}"

                # Format the response
                response_parts = []

                # Add plan if available
                if result.get('executor_plan'):
                    response_parts.append(f"**Plan**: {result['executor_plan'][:200]}")

                # Add command results
                exec_results = result.get('execution_results', [])
                if exec_results:
                    response_parts.append(f"\n**Executed {len(exec_results)} command(s)**:")
                    for r in exec_results[:5]:  # Limit to first 5
                        cmd = r.get('command', r.command if hasattr(r, 'command') else 'unknown')
                        status = r.get('status', r.status.value if hasattr(r, 'status') else 'unknown')
                        output = r.get('output', r.output if hasattr(r, 'output') else '')
                        if len(output) > 200:
                            output = output[:200] + "..."
                        response_parts.append(f"  - `{cmd}` [{status}]")
                        if output.strip():
                            response_parts.append(f"    ```\n{output}\n    ```")

                # Add insights
                if result.get('insights'):
                    response_parts.append(f"\n**Insights**: {', '.join(result['insights'][:3])}")

                # Add next steps
                if result.get('next_steps'):
                    response_parts.append(f"\n**Next**: {result['next_steps']}")

                return '\n'.join(response_parts) if response_parts else "Task executed successfully."

            except Exception as e:
                self.logger.error(f"Operator chat error: {e}")
                return f"Error executing task: {e}"
        else:
            # Use base agent chat for general conversation
            return super().chat(message, history)


# Backward compatibility alias
OperatorAgent = ModularOperatorAgent
