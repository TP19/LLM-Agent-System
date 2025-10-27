#!/usr/bin/env python3
"""
Workflow Validator - Anti-Hallucination Guards

Validates agent outputs to prevent hallucinations and ensure
proper workflow execution.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    confidence: float
    issues: List[str]
    warnings: List[str]
    corrected_data: Optional[Dict[str, Any]] = None


class WorkflowValidator:
    """
    Validates agent outputs and workflow decisions
    
    Prevents:
    - Hallucinated file paths
    - Invalid commands
    - Incorrect workflow routing
    - Fabricated results
    - Inconsistent state
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Known safe command patterns
        self.safe_commands = {
            'ls', 'pwd', 'whoami', 'date', 'uptime',
            'docker ps', 'docker logs', 'docker inspect',
            'systemctl status', 'journalctl',
            'cat', 'head', 'tail', 'grep', 'find',
            'df', 'du', 'free', 'top'
        }
        
        # Dangerous patterns to watch for
        self.danger_patterns = [
            r'rm\s+-rf',
            r'dd\s+if=',
            r'mkfs',
            r':\(\)\{',  # Fork bomb
            r'>/dev/sd',
            r'chmod\s+777',
            r'chmod\s+-R\s+777'
        ]
    
    def validate_triage_result(self, triage_result, original_request: str) -> ValidationResult:
        """
        Validate triage classification - HANDLES BOTH DICT AND OBJECT
        
        Args:
            triage_result: TriageResult object OR dict with classification results
            original_request: Original user request
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        
        # CRITICAL FIX: Handle both dict and object
        if isinstance(triage_result, dict):
            # It's a dict from interactive mode
            category = triage_result.get('classification', 'unknown')
            confidence = triage_result.get('confidence', 0.0)
            recommended_agent = triage_result.get('recommended_agent', 'executor')
        else:
            # It's a TriageResult object from base agent
            category = triage_result.category.value if hasattr(triage_result.category, 'value') else str(triage_result.category)
            confidence = triage_result.confidence
            recommended_agent = triage_result.recommended_agent if hasattr(triage_result, 'recommended_agent') else 'executor'
        
        # Check 1: Category makes sense for request
        request_lower = original_request.lower()
        
        # Terminal categories should not continue to security/executor
        terminal_categories = {'summarization', 'coding', "knowledge_query"}
        
        if category in terminal_categories:
            if 'execute' in request_lower or 'run command' in request_lower:
                warnings.append(
                    f"Request contains 'execute/run' but classified as terminal category '{category}'"
                )
        
        # Check 2: Confidence is reasonable
        if confidence < 0.3:
            warnings.append(
                f"Very low confidence ({confidence:.2f}) - may need user clarification"
            )
        
        # Check 3: Category matches common patterns
        category_patterns = {
            'sysadmin': ['check', 'status', 'disk', 'memory', 'process', 'docker'],
            'fileops': ['find', 'search', 'backup', 'copy', 'move'],
            'network': ['ping', 'download', 'curl', 'wget', 'api'],
            'summarization': ['summarize', 'analyze file', 'overview of', 'extract from'],
            'coding': ['build', 'create module', 'generate code', 'implement']
        }
        
        if category in category_patterns:
            patterns = category_patterns[category]
            has_pattern = any(pattern in request_lower for pattern in patterns)
            
            if not has_pattern and confidence > 0.7:
                warnings.append(
                    f"High confidence ({confidence:.2f}) but no typical '{category}' keywords found"
                )
        
        # Check 4: Terminal categories should have high confidence
        if category in terminal_categories and confidence < 0.7:
            warnings.append(
                f"Terminal category '{category}' should have higher confidence (currently {confidence:.2f})"
            )
        
        # Determine validation result
        is_valid = len(issues) == 0
        validation_confidence = confidence if is_valid else confidence * 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=validation_confidence,
            issues=issues,
            warnings=warnings
        )
    
    def validate_security_suggestions(self, security_result, 
                                     request: str) -> ValidationResult:
        """
        Validate security agent suggestions
        
        Args:
            security_result: Security suggestions
            request: Original request
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        
        # Check 1: Commands should be related to request
        suggested_commands = security_result.get('suggested_commands', [])
        
        if not suggested_commands:
            issues.append("No commands suggested when execution expected")
        
        # Check 2: Validate each command
        for cmd in suggested_commands:
            cmd_validation = self._validate_command(cmd, request)
            
            if not cmd_validation.is_valid:
                issues.extend(cmd_validation.issues)
            
            warnings.extend(cmd_validation.warnings)
        
        # Check 3: Risk level should be reasonable
        risk_level = security_result.get('risk_level', 'unknown')
        
        if risk_level == 'critical':
            # Check if actually dangerous
            has_danger = any(
                re.search(pattern, ' '.join(suggested_commands))
                for pattern in self.danger_patterns
            )
            
            if not has_danger:
                warnings.append("Risk marked critical but no dangerous patterns found")
        
        is_valid = len(issues) == 0
        confidence = 1.0 - (len(issues) * 0.3 + len(warnings) * 0.1)
        confidence = max(0.0, min(1.0, confidence))
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            warnings=warnings
        )
    
    def _validate_command(self, command: str, context: str) -> ValidationResult:
        """
        Validate a single command
        
        Args:
            command: Command to validate
            context: Request context
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        
        # Check for dangerous patterns
        for pattern in self.danger_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                issues.append(f"Dangerous pattern detected: {pattern}")
        
        # Check for hallucinated paths
        path_matches = re.findall(r'(/[a-zA-Z0-9_/.-]+)', command)
        
        for path_str in path_matches:
            # Skip common system paths
            if path_str.startswith(('/usr/', '/bin/', '/etc/', '/var/', '/tmp/')):
                continue
            
            # Check if path mentioned in context
            if path_str not in context:
                warnings.append(f"Path not in request context: {path_str}")
        
        # Check for IP addresses not in request
        ip_matches = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', command)
        
        for ip in ip_matches:
            if ip not in context:
                warnings.append(f"IP address not in request: {ip}")
        
        is_valid = len(issues) == 0
        confidence = 1.0 if is_valid else 0.0
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            warnings=warnings
        )
    
    def validate_execution_result(self, execution_result: Dict,
                                  expected_commands: List[str]) -> ValidationResult:
        """
        Validate execution results
        
        Args:
            execution_result: Execution result dict
            expected_commands: Commands that should have been executed
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        
        # Check 1: Verify commands were actually executed
        executed_commands = execution_result.get('executed_commands', [])
        
        if not executed_commands:
            issues.append("No commands executed when expected")
        
        # Check 2: Results should have outputs
        command_results = execution_result.get('command_results', [])
        
        if executed_commands and not command_results:
            warnings.append("Commands executed but no results captured")
        
        # Check 3: Validate completion claim
        is_complete = execution_result.get('is_complete', False)
        completion_reasoning = execution_result.get('completion_reasoning', '')
        
        if is_complete and not completion_reasoning:
            warnings.append("Marked complete but no reasoning provided")
        
        # Check 4: Verify output consistency
        for cmd_result in command_results:
            if cmd_result.get('success') and not cmd_result.get('output'):
                warnings.append(
                    f"Command marked successful but no output: "
                    f"{cmd_result.get('command', 'unknown')}"
                )
        
        is_valid = len(issues) == 0
        confidence = 0.9 - (len(issues) * 0.3 + len(warnings) * 0.1)
        confidence = max(0.0, min(1.0, confidence))
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            warnings=warnings
        )
    
    def validate_workflow_routing(self, triage_category: str,
                                  next_stage: str) -> ValidationResult:
        """
        Validate workflow routing decisions
        
        Args:
            triage_category: Category from triage
            next_stage: Next planned stage
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        
        # Terminal categories should not route to execution
        terminal_categories = {'summarization', 'coding', 'knowledge_query'}
        execution_stages = {'security', 'executor', 'execution'}
        
        if triage_category in terminal_categories and next_stage in execution_stages:
            issues.append(
                f"Invalid routing: {triage_category} is terminal but "
                f"routing to {next_stage}"
            )
        
        # Valid routing patterns
        valid_routes = {
            'summarization': {'completion', 'qa'},
            'coding': {'completion', 'qa'},
            'knowledge_query': {'completion', 'qa'},
            'sysadmin': {'security', 'execution'},
            'fileops': {'security', 'execution'},
            'network': {'security', 'execution'},
            'development': {'security', 'execution'},
            'security': {'security', 'execution'}
        }
        
        if triage_category in valid_routes:
            if next_stage not in valid_routes[triage_category]:
                warnings.append(
                    f"Unusual routing: {triage_category} → {next_stage}"
                )
        
        is_valid = len(issues) == 0
        confidence = 1.0 if is_valid else 0.0
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            warnings=warnings
        )
    
    def validate_summarization_result(self, summary_result: Dict,
                                     source_content: str) -> ValidationResult:
        """
        Validate summarization results
        
        Args:
            summary_result: Summary result dict
            source_content: Original content
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        
        summary_text = summary_result.get('summary', '')
        
        # Check 1: Summary exists
        if not summary_text:
            issues.append("Empty summary generated")
        
        # Check 2: Summary is shorter than source
        if len(summary_text) > len(source_content):
            issues.append("Summary longer than source (possible hallucination)")
        
        # Check 3: Summary references source concepts
        # Extract key terms from source
        source_words = set(
            word.lower() for word in re.findall(r'\w+', source_content)
            if len(word) > 5
        )
        
        summary_words = set(
            word.lower() for word in re.findall(r'\w+', summary_text)
            if len(word) > 5
        )
        
        if source_words:
            overlap = len(source_words & summary_words) / len(source_words)
            
            if overlap < 0.1:
                warnings.append(
                    f"Low term overlap ({overlap:.1%}) - possible hallucination"
                )
        
        is_valid = len(issues) == 0
        confidence = 0.9 - (len(warnings) * 0.1)
        confidence = max(0.0, min(1.0, confidence))
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            warnings=warnings
        )