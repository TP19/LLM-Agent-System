#!/usr/bin/env python3
"""
Modular Security Agent - Lazy Loading Version

This security agent helps drive the process by suggesting commands and reasoning,
rather than blocking execution. Focus on collaboration, not safety theater.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import uuid

# Import base agent for modular architecture
from core.base_agent import BaseAgent

@dataclass
class SecuritySuggestion:
    """Security agent's collaborative suggestion"""
    commands: List[str]
    reasoning: str
    approach: str
    next_steps: List[str]
    confidence: float
    timestamp: datetime

class TaskCategory(Enum):
    SYSADMIN = "sysadmin"
    FILEOPS = "fileops" 
    NETWORK = "network"
    DEVELOPMENT = "development"
    CONTENT = "content"
    SECURITY = "security"
    CODING = "coding"     
    UNKNOWN = "unknown"

class TaskDifficulty(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class ModularSecurityAgent(BaseAgent):
    """Modular security agent focused on helping, not blocking"""

    def __init__(self, model_manager):
        super().__init__("security", model_manager)

        # Collaborative prompt - focused on helping
        self.prompt = self._create_collaborative_prompt()

        self.stats.update({
            'suggestions_made': 0,
            'commands_suggested': 0,
            'avg_confidence': 0.0
        })

        self.logger.info("✅ Modular security agent initialized")

    def _create_collaborative_prompt(self) -> str:
        """Create collaborative prompt with better SSH command handling"""
        return """You are a Security Agent working WITH an Executor Agent to help users accomplish their goals safely and effectively.

YOUR ROLE: Collaborative partner, not gatekeeper
- Suggest specific commands that will help accomplish the user's goal
- Provide reasoning for your suggestions
- Think step-by-step about the best approach
- Focus on getting work done, not blocking it

SSH COMMAND RULES:
- For SSH commands, always use single quotes around the remote command
- Format: ssh user@host 'command here'
- This avoids quote escaping issues

COMMAND FORMAT:
- Generate clean, executable commands
- No numbering, bullet points, or extra formatting
- Commands ready to execute directly
- When sshing into servers, and needing to retrieve information about ssh status, try first to retrieve information via direct commands as ssh username@host 'df -h' and not sshing directly as for example username@host
- Remember, One command per line

FORMAT:
COMMANDS:
ls -la /tmp

REASONING: [why these commands help achieve the goal]
APPROACH: [overall strategy]  
NEXT_STEPS: [what to do after these commands]
CONFIDENCE: [0.0-1.0]

User Request: """

    def suggest_approach(self, user_request: str, triage_info: Dict = None, request_id: str = None) -> SecuritySuggestion:
        """Suggest collaborative approach for user request"""

        if not request_id:
            request_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"[{request_id}] Creating security suggestion for: {user_request[:50]}...")

        self.logger.info(f"Creating suggestion for: {user_request[:50]}...")

        start_time = time.time()

        try:
            # Build context
            context = ""
            if triage_info:
                context = f"\nTriage Info: Category={triage_info.get('category', 'unknown')}, Difficulty={triage_info.get('difficulty', 'unknown')}\n"

            # Generate suggestion using enhanced base class
            full_prompt = self.prompt + context + user_request
            response = self.generate_with_logging(
                full_prompt,
                request_id,
                max_tokens=600,
                temperature=0.4,
                top_p=0.9
            )

            suggestion = self._parse_suggestion(response, user_request)

            # Update stats
            self._update_stats(suggestion)

            self.logger.info(f"✅ Suggested {len(suggestion.commands)} commands in {time.time() - start_time:.2f}s")

            return suggestion

        except Exception as e:
            self.logger.error(f"Failed to create suggestion: {e}")
            return self._create_fallback_suggestion(user_request)

    def _parse_suggestion(self, response: str, user_request: str) -> SecuritySuggestion:
        """Parse LLM response into structured suggestion"""
        import re

        try:
            # Extract commands with better parsing
            commands = []
            commands_match = re.search(r'COMMANDS:\s*(.+?)(?=\nREASONING:|$)', response, re.IGNORECASE | re.DOTALL)
            if commands_match:
                commands_text = commands_match.group(1)

                # Parse commands line by line, cleaning each one
                for line in commands_text.split('\n'):
                    line = line.strip()
                    if line:
                        # Clean the command properly
                        cleaned_command = self._clean_command_string(line)
                        if cleaned_command:
                            commands.append(cleaned_command)

            # If no commands found, try extracting from anywhere in response
            if not commands:
                # Look for common command patterns
                ssh_commands = re.findall(r'ssh\s+\w+@[\d\.]+\s+"[^"]+"', response)
                for cmd in ssh_commands:
                    cleaned = self._clean_command_string(cmd)
                    if cleaned:
                        commands.append(cleaned)

            # Extract other fields
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\nAPPROACH:|$)', response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "Collaborative analysis"

            approach_match = re.search(r'APPROACH:\s*(.+?)(?=\nNEXT_STEPS:|$)', response, re.IGNORECASE | re.DOTALL)
            approach = approach_match.group(1).strip() if approach_match else "Step-by-step investigation"

            next_steps = []
            next_steps_match = re.search(r'NEXT_STEPS:\s*(.+?)(?=\nCONFIDENCE:|$)', response, re.IGNORECASE | re.DOTALL)
            if next_steps_match:
                next_steps_text = next_steps_match.group(1)
                for line in next_steps_text.split('\n'):
                    line = line.strip()
                    if line and len(line) > 5:  # Skip very short lines
                        step = self._clean_step_string(line)
                        if step:
                            next_steps.append(step)

            confidence = 0.8
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass

            return SecuritySuggestion(
                commands=commands,
                reasoning=reasoning,
                approach=approach,
                next_steps=next_steps,
                confidence=confidence,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Failed to parse suggestion: {e}")
            return self._create_fallback_suggestion(user_request)

    def _clean_command_string(self, raw_command: str) -> str:
        """Clean command string with SSH handling"""
        import re

        # Remove numbering (1., 2., etc.)
        cleaned = re.sub(r'^\d+\.\s*', '', raw_command.strip())

        # Remove bullet points and list markers
        cleaned = re.sub(r'^[-*•]\s*', '', cleaned)

        # Remove markdown backticks
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)

        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())

        # Special handling for SSH commands
        if cleaned.lower().startswith('ssh'):
            cleaned = self._fix_ssh_command(cleaned)
        else:
            # For non-SSH commands, remove extra quotes
           cleaned = cleaned.strip("'\"`)")

        # Basic validation
        if len(cleaned) < 3:
            return ""

        # Must start with a valid command pattern
        valid_starts = ['ssh', 'ls', 'df', 'du', 'ps', 'cat', 'grep', 'find', 'whoami', 'pwd', 'sudo']
        if not any(cleaned.lower().startswith(cmd) for cmd in valid_starts):
            return ""

        return cleaned

    def _clean_step_string(self, raw_step: str) -> str:
        """Clean a next step string"""
        import re

        # Remove numbering
        cleaned = re.sub(r'^\d+\.\s*', '', raw_step.strip())

        # Remove bullet points
        cleaned = re.sub(r'^[-*•]\s*', '', cleaned)

        # Clean up quotes and formatting
        cleaned = cleaned.strip("'\"`)")

        # Clean up whitespace
        cleaned = ' '.join(cleaned.split())

        return cleaned if len(cleaned) > 10 else ""


    def _create_fallback_suggestion(self, user_request: str) -> SecuritySuggestion:
        """Create fallback suggestion when parsing fails"""

        # Create basic commands based on request content
        commands = []
        request_lower = user_request.lower()

        if 'disk' in request_lower or 'space' in request_lower:
            commands.extend(['df -h', 'du -sh /var /tmp /home'])

        if 'ssh' in request_lower:
            # Extract target if possible
            import re
            ssh_match = re.search(r'(\w+@[\d\.]+)', user_request)
            if ssh_match:
                target = ssh_match.group(1)
                commands.extend([
                    f'ssh {target} "whoami"',
                    f'ssh {target} "df -h"'
                ])

        if 'log' in request_lower:
            commands.extend(['ls -la /var/log', 'du -sh /var/log/*'])

        if not commands:
            commands = ['whoami', 'pwd', 'ls -la']

        return SecuritySuggestion(
            commands=commands,
            reasoning="Fallback analysis - basic investigation commands",
            approach="Gather information to understand the situation",
            next_steps=["Review command outputs", "Determine next actions"],
            confidence=0.6,
            timestamp=datetime.now()
        )

    def _fix_ssh_command(self, command: str) -> str:
        """Fix SSH command quote issues"""
        import re

        # Check if it's an SSH command
        if not command.lower().startswith('ssh'):
            return command

        # Pattern to match SSH commands with potential quote issues
        ssh_pattern = r'ssh\s+(\w+@[\d\.]+)\s*["\']?([^"\']*)["\']?'
        match = re.match(ssh_pattern, command)

        if match:
            ssh_target = match.group(1)
            remote_command = match.group(2).strip()

            # Rebuild with single quotes (safer for shell)
            if remote_command:
                fixed_command = f"ssh {ssh_target} '{remote_command}'"
                return fixed_command

        return command

    def analyze_results(self, executed_commands: List[Dict], user_request: str, request_id: str = None) -> SecuritySuggestion:
        """Analyze execution results and suggest next steps"""

        self.logger.info(f"Analyzing results from {len(executed_commands)} commands")

        # Build results context
        results_context = "EXECUTION RESULTS:\n"
        for cmd_result in executed_commands:
            cmd = cmd_result.get('command', 'unknown')
            success = cmd_result.get('success', False)
            output = cmd_result.get('output', '')[:200]  # First 200 chars
            results_context += f"- {cmd}: {'✅' if success else '❌'} {output}\n"

        prompt = f"""Based on these command execution results, suggest what to do next:

ORIGINAL REQUEST: {user_request}

{results_context}

What should we do next to help the user accomplish their goal?
Suggest specific follow-up commands based on what we learned.

FORMAT:
COMMANDS: [list of specific follow-up commands]
REASONING: [analysis of results and why these next commands]
APPROACH: [strategy for next phase]
NEXT_STEPS: [what to do after these commands]
CONFIDENCE: [0.0-1.0]
"""

        try:
            response = self.generate_with_logging(
                prompt,
                request_id,
                max_tokens=500,
                temperature=0.4,
                top_p=0.9
            )

            suggestion = self._parse_suggestion(response, user_request)
            self._update_stats(suggestion)

            return suggestion

        except Exception as e:
            self.logger.error(f"Failed to analyze results: {e}")
            return self._create_fallback_suggestion(user_request)

    def _update_stats(self, suggestion: SecuritySuggestion):
        """Update internal statistics"""
        self.stats['suggestions_made'] += 1
        self.stats['commands_suggested'] += len(suggestion.commands)

        # Update average confidence
        total_confidence = self.stats['avg_confidence'] * (self.stats['suggestions_made'] - 1)
        self.stats['avg_confidence'] = (total_confidence + suggestion.confidence) / self.stats['suggestions_made']

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            **self.stats,
            'avg_commands_per_suggestion': self.stats['commands_suggested'] / max(1, self.stats['suggestions_made'])
        }

    # Aggresive Collaboration when executor is stuck in "planning paralysis" and not WORKING!
    def execute_commands_directly(self, commands: List[str], user_request: str) -> Dict[str, Any]:
        """Security agent executes commands directly when executor stalls"""

        self.logger.info(f"🚀 SECURITY AGENT TAKING CONTROL - Executing {len(commands)} commands directly")

        results = []

        for i, command in enumerate(commands, 1):
            self.logger.info(f"🔧 [{i}/{len(commands)}] SECURITY EXECUTING: {command}")

            try:
                # Execute the command directly
                import subprocess
                import time

                start_time = time.time()

                # Basic safety check
                if self._is_command_safe_for_direct_execution(command):
                    process = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    execution_time = time.time() - start_time
                    success = process.returncode == 0

                    result = {
                        'command': command,
                        'success': success,
                        'output': process.stdout + process.stderr,
                        'exit_code': process.returncode,
                        'execution_time': execution_time,
                        'executed_by': 'security_agent'
                    }

                    results.append(result)

                    if success:
                        self.logger.info(f"✅ [{i}/{len(commands)}] SECURITY SUCCESS: {command}")
                        print(f"🔧 EXECUTED: {command}")
                        print(f"📄 OUTPUT: {result['output'][:200]}...")
                    else:
                        self.logger.warning(f"❌ [{i}/{len(commands)}] SECURITY FAILED: {command}")
                        print(f"❌ FAILED: {command} - {result['output'][:100]}")

                else:
                    result = {
                        'command': command,
                        'success': False,
                        'output': 'Command blocked by security agent safety check',
                        'exit_code': -1,
                        'execution_time': 0,
                        'executed_by': 'security_agent'
                    }
                    results.append(result)
                    self.logger.warning(f"🚫 [{i}/{len(commands)}] SECURITY BLOCKED: {command}")

                # Small delay between commands
                time.sleep(1)

            except Exception as e:
                result = {
                    'command': command,
                    'success': False,
                    'output': f'Security agent execution error: {str(e)}',
                    'exit_code': -1,
                    'execution_time': 0,
                    'executed_by': 'security_agent'
                }
                results.append(result)
                self.logger.error(f"💥 [{i}/{len(commands)}] SECURITY ERROR: {command} - {e}")

        # Analyze what we accomplished
        successful_results = [r for r in results if r['success']]

        summary = {
            'status': 'security_agent_execution_complete',
            'commands_executed': len(results),
            'successful_commands': len(successful_results),
            'failed_commands': len(results) - len(successful_results),
            'execution_results': results,
            'analysis': self._analyze_direct_execution_results(results, user_request),
            'next_actions': self._suggest_next_actions_from_results(results, user_request)
        }

        return summary

    def _is_command_safe_for_direct_execution(self, command: str) -> bool:
        """Check if command is safe for security agent to execute directly"""

        # Allow most read-only operations
        safe_patterns = [
            r'^ssh\s+\w+@[\d\.]+\s+\'docker ps',
            r'^ssh\s+\w+@[\d\.]+\s+\'docker start \w+',
            r'^ssh\s+\w+@[\d\.]+\s+\'docker stop \w+',
            r'^ssh\s+\w+@[\d\.]+\s+\'docker restart \w+',
            r'^ssh\s+\w+@[\d\.]+\s+\'docker logs \w+',
            r'^ssh\s+\w+@[\d\.]+\s+\'df -h',
            r'^ssh\s+\w+@[\d\.]+\s+\'du -sh',
            r'^ssh\s+\w+@[\d\.]+\s+\'ls -',
            r'^ssh\s+\w+@[\d\.]+\s+\'ps aux',
            r'^df\s+-h',
            r'^du\s+-sh',
            r'^ls\s+-',
            r'^ps\s+aux',
            r'^whoami',
            r'^pwd'
        ]

        import re
        cmd_lower = command.lower()

        # Block obviously dangerous commands
        dangerous_patterns = [
            'rm -rf /',
            'dd if=',
            'mkfs',
            'format',
            'shutdown',
            'reboot',
            'passwd',
            'userdel',
            'deluser'
        ]

        for dangerous in dangerous_patterns:
            if dangerous in cmd_lower:
                return False

        # Allow commands that match safe patterns
        for pattern in safe_patterns:
            if re.match(pattern, command, re.IGNORECASE):
                return True

        # Default: allow if it looks safe
        return len(command) < 100 and not any(char in command for char in ['&', '|', ';', '`', '$'])

    def _analyze_direct_execution_results(self, results: List[Dict], user_request: str) -> str:
        """Analyze results of direct execution"""

        if not results:
            return "No commands were executed"

        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        analysis_parts = []
        analysis_parts.append(f"SECURITY AGENT EXECUTION SUMMARY:")
        analysis_parts.append(f"✅ Successful: {len(successful)}")
        analysis_parts.append(f"❌ Failed: {len(failed)}")

        # Analyze specific outputs
        for result in successful:
            if 'docker ps' in result['command']:
                analysis_parts.append(f"🐳 Container status: Found running containers in output")
            elif 'docker start' in result['command']:
                analysis_parts.append(f"🚀 Container start: Attempted to start container")
            elif 'df -h' in result['command']:
                analysis_parts.append(f"💾 Disk space: Retrieved disk usage information")

        return "\n".join(analysis_parts)

    def _suggest_next_actions_from_results(self, results: List[Dict], user_request: str) -> List[str]:
        """Suggest next actions based on execution results"""

        actions = []

        # Look for patterns in results
        container_found = False
        container_started = False

        for result in results:
            if result['success'] and 'docker ps' in result['command']:
                if 'air' in result['output'].lower():
                    container_found = True
                    actions.append("Container with 'air' found in docker ps output")

            if result['success'] and 'docker start' in result['command']:
                container_started = True
                actions.append("Container start command executed")

        if not container_found and any('docker' in r['command'] for r in results):
            actions.append("Search for containers with different name patterns")

        if not actions:
            actions.append("Review command outputs and continue investigation")

        return actions
