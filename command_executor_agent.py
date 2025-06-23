#!/usr/bin/env python3
"""
Command Executor Agent - Conversation-Aware Command Execution with llama.cpp

This agent executes approved commands while maintaining conversation context using
llama.cpp models loaded on-demand for optimal resource management.

Key features:
- On-demand model loading for conversation management
- Stateful conversation management over stateless LLM API
- Security whitelist integration
- Comprehensive execution logging
- Error handling and recovery
- Performance monitoring
- Resource-efficient model lifecycle
"""

import subprocess
import json
import re
import os
import logging
import time
import yaml
import gc
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError as e:
    print("Error: llama-cpp-python not installed.")
    print("Install with: pip install llama-cpp-python")
    raise e

class MessageType(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    COMMAND_OUTPUT = "command"

@dataclass
class ConversationMessage:
    """Represents a single message in conversation history"""
    type: MessageType
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_prompt_format(self) -> str:
        """Convert to LLM prompt format"""
        if self.type == MessageType.SYSTEM:
            return f"System: {self.content}"
        elif self.type == MessageType.USER:
            return f"Human: {self.content}"
        elif self.type == MessageType.ASSISTANT:
            return f"Assistant: {self.content}"
        elif self.type == MessageType.COMMAND_OUTPUT:
            return f"Command Result: {self.content}"
        return self.content

@dataclass
class CommandExecution:
    """Results of command execution"""
    command: str
    success: bool
    output: str
    exit_code: int
    execution_time: float
    timestamp: datetime
    whitelist_entry: Optional[str] = None

class ExecutorModelManager:
    """
    Manages llama.cpp model lifecycle for command execution and conversation
    
    This component handles model loading/unloading specifically optimized
    for conversational command execution tasks.
    """
    
    def __init__(self, model_path: str, n_gpu_layers: int = 40, n_ctx: int = 2048, verbose: bool = False):
        self.model_path = Path(model_path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.llm = None
        self.load_time = None
        
        # Validate model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Executor model file not found: {model_path}")
        
        self.logger = logging.getLogger("ExecutorModelManager")
    
    def load_model(self) -> bool:
        """Load the executor model into memory"""
        if self.llm is not None:
            self.logger.debug("Executor model already loaded")
            return True
        
        try:
            start_time = time.time()
            self.logger.info(f"Loading executor model: {self.model_path.name}")
            
            self.llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=self.verbose
            )
            
            self.load_time = time.time() - start_time
            self.logger.info(f"Executor model loaded successfully in {self.load_time:.1f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load executor model: {e}")
            self.llm = None
            return False
    
    def unload_model(self):
        """Unload model and free memory"""
        if self.llm is not None:
            self.logger.info("Unloading executor model")
            del self.llm
            self.llm = None
            
            # Force garbage collection to free memory
            gc.collect()
            
            self.logger.debug("Executor model unloaded and memory freed")
    
    def is_loaded(self) -> bool:
        """Check if model is currently loaded"""
        return self.llm is not None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate conversation response using the loaded model"""
        if self.llm is None:
            raise RuntimeError("Executor model not loaded. Call load_model() first.")
        
        try:
            # Set default generation parameters optimized for conversation
            generation_params = {
                'temperature': kwargs.get('temperature', 0.3),
                'max_tokens': kwargs.get('max_tokens', 250),
                'top_p': kwargs.get('top_p', 0.9),
                'stop': kwargs.get('stop_sequence', ["\nHuman:", "\nUser:", "Human:", "User:"]),
                'stream': False
            }
            
            # Generate response
            response = self.llm(prompt, **generation_params)
            
            # Extract generated text
            if isinstance(response, dict) and 'choices' in response:
                generated_text = response['choices'][0]['text'].strip()
            else:
                generated_text = str(response).strip()
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Conversation generation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded executor model"""
        return {
            'model_path': str(self.model_path),
            'model_name': self.model_path.name,
            'n_gpu_layers': self.n_gpu_layers,
            'n_ctx': self.n_ctx,
            'load_time': self.load_time,
            'is_loaded': self.is_loaded(),
            'model_type': 'command_execution'
        }

class ConversationManager:
    """
    Manages conversation state across multiple LLM interactions
    
    This component maintains the complete conversation history, allowing
    the LLM to make context-aware decisions based on previous interactions.
    """
    
    def __init__(self, max_context_length: int = 6000):
        self.messages: List[ConversationMessage] = []
        self.max_context_length = max_context_length
        self.logger = logging.getLogger("ConversationManager")
    
    def add_message(self, msg_type: MessageType, content: str, metadata: Dict[str, Any] = None):
        """Add message to conversation history"""
        message = ConversationMessage(
            type=msg_type,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.logger.debug(f"Added {msg_type.value} message: {content[:50]}...")
    
    def build_prompt_context(self) -> str:
        """Build complete conversation context for LLM"""
        formatted_messages = [msg.to_prompt_format() for msg in self.messages]
        full_context = "\n\n".join(formatted_messages)
        
        # Intelligent truncation if context too long
        if len(full_context) > self.max_context_length:
            # Keep system message and recent messages
            system_msg = formatted_messages[0] if self.messages and self.messages[0].type == MessageType.SYSTEM else ""
            recent_msgs = formatted_messages[-8:]  # Keep last 8 messages
            
            truncated_context = system_msg + "\n\n[...conversation history truncated...]\n\n" + "\n\n".join(recent_msgs)
            self.logger.warning(f"Context truncated from {len(full_context)} to {len(truncated_context)} chars")
            return truncated_context
        
        return full_context
    
    def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary for monitoring"""
        return {
            'total_messages': len(self.messages),
            'message_types': {
                msg_type.value: sum(1 for msg in self.messages if msg.type == msg_type) 
                for msg_type in MessageType
            },
            'context_length': len(self.build_prompt_context()),
            'last_message_time': self.messages[-1].timestamp.isoformat() if self.messages else None
        }
    
    def clear_conversation(self):
        """Clear conversation history (except system message)"""
        if self.messages and self.messages[0].type == MessageType.SYSTEM:
            system_msg = self.messages[0]
            self.messages = [system_msg]
        else:
            self.messages = []
        self.logger.info("Conversation history cleared")

class CommandExecutor:
    """
    Handles secure command execution with whitelist integration
    
    This component executes only whitelisted commands and provides
    comprehensive logging and error handling.
    """
    
    def __init__(self, whitelist_file: str = "config/command_whitelist.json"):
        self.logger = logging.getLogger("CommandExecutor")
        self.whitelist_file = whitelist_file
        
        # Load configuration
        self.config = self._load_config()
        
        # Load whitelist
        self.whitelist = self._load_whitelist()
        
        # Execution tracking
        self.execution_history: List[CommandExecution] = []
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'security_violations': 0,
            'avg_execution_time': 0.0
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load executor configuration"""
        config_file = Path("config/executor_config.yaml")
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
        
        # Default configuration
        return {
            'command_timeout': 30,
            'max_output_length': 3000,
            'environment_variables': {
                'PYTHONUNBUFFERED': '1',
                'PATH': os.environ.get('PATH', '')
            }
        }
    
    def _load_whitelist(self) -> Dict[str, Any]:
        """Load command whitelist from security agent"""
        try:
            with open(self.whitelist_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Whitelist file {self.whitelist_file} not found")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading whitelist: {e}")
            return {}
    
    def reload_whitelist(self):
        """Reload whitelist (called after security agent updates)"""
        old_size = len(self.whitelist)
        self.whitelist = self._load_whitelist()
        new_size = len(self.whitelist)
        
        if new_size != old_size:
            self.logger.info(f"Whitelist reloaded: {old_size} -> {new_size} entries")
    
    def is_command_allowed(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Check if command is allowed by whitelist with regex support
        
        Returns: (is_allowed, whitelist_entry_key)
        """
        command = command.strip()
        
        for entry_key, entry_data in self.whitelist.items():
            pattern = entry_data['command_pattern']
            pattern_type = entry_data.get('pattern_type', 'exact')
            
            # Exact match first (for performance)
            if pattern_type == "exact" and command == pattern:
                return True, entry_key
            
            # Pattern matching (exact or regex)
            if self._matches_pattern(command, pattern, pattern_type):
                return True, entry_key
        
        return False, None
    
    def _matches_pattern(self, command: str, pattern: str, pattern_type: str = "exact") -> bool:
        """Check if command matches whitelist pattern with regex support"""
        if pattern_type == "regex":
            try:
                return bool(re.match(pattern, command, re.IGNORECASE))
            except re.error as e:
                self.logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                return False
        
        # Exact matching (legacy and default)
        if pattern_type == "exact" or pattern_type is None:
            # For python commands
            if pattern.startswith('python'):
                return command.startswith(pattern)
            
            # For commands with parameters
            if ' ' in pattern:
                pattern_parts = pattern.split()
                command_parts = command.split()
                
                if len(command_parts) >= len(pattern_parts):
                    for i, pattern_part in enumerate(pattern_parts):
                        if pattern_part != command_parts[i]:
                            return False
                    return True
            
            return command == pattern
        
        # Unknown pattern type, default to exact
        return command == pattern
    
    def execute_command(self, command: str) -> CommandExecution:
        """Execute a single command with comprehensive error handling"""
        start_time = time.time()
        
        # Security check
        is_allowed, whitelist_entry = self.is_command_allowed(command)
        if not is_allowed:
            self.stats['security_violations'] += 1
            error_msg = f"SECURITY: Command '{command}' not in whitelist"
            self.logger.warning(error_msg)
            
            return CommandExecution(
                command=command,
                success=False,
                output=error_msg,
                exit_code=-1,
                execution_time=0.0,
                timestamp=datetime.now(),
                whitelist_entry=None
            )
        
        try:
            self.logger.info(f"Executing: {command}")
            
            # Prepare environment
            env = dict(os.environ)
            env.update(self.config.get('environment_variables', {}))
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config['command_timeout'],
                env=env
            )
            
            execution_time = time.time() - start_time
            
            # Format output
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            
            # Truncate if too long
            max_length = self.config['max_output_length']
            if len(output) > max_length:
                output = output[:max_length] + f"\n... (truncated after {max_length} chars)"
            
            # Add execution metadata
            output += f"\n[Completed in {execution_time:.2f}s, exit code: {result.returncode}]"
            
            success = (result.returncode == 0)
            
            # Create execution record
            execution = CommandExecution(
                command=command,
                success=success,
                output=output,
                exit_code=result.returncode,
                execution_time=execution_time,
                timestamp=datetime.now(),
                whitelist_entry=whitelist_entry
            )
            
            # Update statistics
            self._update_stats(execution)
            
            # Store in history
            self.execution_history.append(execution)
            
            # Log result
            status = "SUCCESS" if success else "FAILED"
            self.logger.info(f"{status}: {command} (exit: {result.returncode}, time: {execution_time:.2f}s)")
            
            return execution
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            error_msg = f"Command timed out after {self.config['command_timeout']}s"
            self.logger.error(f"TIMEOUT: {command}")
            
            execution = CommandExecution(
                command=command,
                success=False,
                output=error_msg,
                exit_code=-2,
                execution_time=execution_time,
                timestamp=datetime.now(),
                whitelist_entry=whitelist_entry
            )
            
            self._update_stats(execution)
            self.execution_history.append(execution)
            return execution
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Execution error: {str(e)}"
            self.logger.error(f"ERROR: {command} - {e}")
            
            execution = CommandExecution(
                command=command,
                success=False,
                output=error_msg,
                exit_code=-3,
                execution_time=execution_time,
                timestamp=datetime.now(),
                whitelist_entry=whitelist_entry
            )
            
            self._update_stats(execution)
            self.execution_history.append(execution)
            return execution
    
    def _update_stats(self, execution: CommandExecution):
        """Update execution statistics"""
        self.stats['total_executions'] += 1
        
        if execution.success:
            self.stats['successful_executions'] += 1
        else:
            self.stats['failed_executions'] += 1
        
        # Update average execution time
        total_time = self.stats['avg_execution_time'] * (self.stats['total_executions'] - 1)
        self.stats['avg_execution_time'] = (total_time + execution.execution_time) / self.stats['total_executions']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total = self.stats['total_executions']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'success_rate': self.stats['successful_executions'] / total,
            'failure_rate': self.stats['failed_executions'] / total,
            'security_violation_rate': self.stats['security_violations'] / total,
            'whitelist_size': len(self.whitelist)
        }

class CommandExecutorAgent:
    """
    Main agent that combines conversation management with command execution using llama.cpp
    
    This orchestrates the complete flow with on-demand model loading:
    - Maintains conversation state
    - Communicates with LLM using llama.cpp
    - Parses commands from responses
    - Executes commands securely
    - Continues conversation flow
    """
    
    def __init__(self, 
                 model_path: str,
                 n_gpu_layers: int = 40,
                 n_ctx: int = 2048,
                 model_verbose: bool = False,
                 max_iterations: int = 20):
        
        # Set up logging
        self.logger = logging.getLogger("CommandExecutorAgent")
        
        # Initialize model manager
        self.model_manager = ExecutorModelManager(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=model_verbose
        )
        
        # Initialize components
        self.max_iterations = max_iterations
        self.conversation = ConversationManager()
        self.executor = CommandExecutor()
        
        # LLM parameters
        self.llm_params = {
            'temperature': 0.3,
            'max_tokens': 250,
            'top_p': 0.9,
            'stop_sequence': ["\nHuman:", "\nUser:", "Human:", "User:"]
        }
        
        # Set up system prompt
        self.system_prompt = self._create_system_prompt()
        self.conversation.add_message(MessageType.SYSTEM, self.system_prompt)
        
        # Statistics
        self.session_stats = {
            'tasks_processed': 0,
            'tasks_completed': 0,
            'llm_calls': 0,
            'total_processing_time': 0.0,
            'model_load_count': 0,
            'total_model_load_time': 0.0
        }
        
        self.logger.info(f"Command Executor Agent initialized with model: {self.model_manager.model_path.name}")
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for command execution"""
        return """You are a helpful AI assistant that can execute system commands safely.

EXECUTION RULES:
1. To execute a command, use: EXECUTE: [command]
2. Execute only ONE command per response
3. Wait for command results before deciding next steps
4. When task is complete, use: TASK_COMPLETE: [summary]
5. If you need clarification, use: NEED_CLARIFICATION: [question]

AVAILABLE COMMANDS (whitelisted):
- Network: curl ifconfig.co, ping -c 4 google.com
- System Info: whoami, pwd, date, df -h, free -h, uptime
- File Operations: ls -la, ls, cat /etc/os-release
- Python: python3 -c "code_here"

WORKFLOW:
1. Understand the user's request
2. Plan the necessary commands
3. Execute commands one by one
4. Analyze results and continue if needed
5. Provide final summary when complete

EXAMPLE:
Human: What's my IP address?
Assistant: I'll check your public IP address.
EXECUTE: curl ifconfig.co

Remember: Always explain what you're doing and why. Be helpful and thorough."""
    
    def send_to_llm(self, additional_context: str = "") -> str:
        """Send conversation context to LLM and get response with on-demand loading"""
        try:
            # Load model if not already loaded
            if not self.model_manager.is_loaded():
                self.logger.info("Loading executor model for conversation...")
                model_load_start = time.time()
                
                if not self.model_manager.load_model():
                    raise RuntimeError("Failed to load executor model")
                
                model_load_time = time.time() - model_load_start
                self.session_stats['model_load_count'] += 1
                self.session_stats['total_model_load_time'] += model_load_time
            
            # Build complete prompt
            context = self.conversation.build_prompt_context()
            if additional_context:
                prompt = context + "\n\n" + additional_context
            else:
                prompt = context + "\n\nAssistant:"
            
            # Call LLM
            self.session_stats['llm_calls'] += 1
            
            generated_text = self.model_manager.generate(prompt, **self.llm_params)
            
            if len(generated_text) < 3:
                return "I need to think about this more. Could you clarify your request?"
            
            self.logger.debug(f"LLM response: {generated_text[:100]}...")
            return generated_text
            
        except Exception as e:
            error_msg = f"LLM communication error: {e}"
            self.logger.error(error_msg)
            return "I'm having trouble processing your request. Please try again."
    
    def parse_llm_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to determine action"""
        response = response.strip()
        
        # Look for command execution
        execute_patterns = [
            r'EXECUTE:\s*(.+)',
            r'RUN:\s*(.+)',
            r'COMMAND:\s*(.+)'
        ]
        
        for pattern in execute_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                command = match.group(1).strip()
                # Clean up command
                command = re.sub(r'["`\']*$', '', command)
                return 'execute', command
        
        # Look for task completion
        complete_patterns = [
            r'TASK_COMPLETE:\s*(.+)',
            r'COMPLETE:\s*(.+)',
            r'DONE:\s*(.+)'
        ]
        
        for pattern in complete_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return 'complete', match.group(1).strip()
        
        # Look for clarification
        clarification_patterns = [
            r'NEED_CLARIFICATION:\s*(.+)',
            r'CLARIFICATION:\s*(.+)'
        ]
        
        for pattern in clarification_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return 'clarification', match.group(1).strip()
        
        # Default: continue conversation
        return 'continue', response
    
    def execute_task(self, user_request: str) -> Dict[str, Any]:
        """Execute a complete user task with on-demand model loading"""
        start_time = time.time()
        self.logger.info(f"Starting task execution: {user_request[:80]}...")
        
        # Initialize task tracking
        self.session_stats['tasks_processed'] += 1
        iterations = 0
        commands_executed = 0
        model_info = None
        
        # Add user request to conversation
        self.conversation.add_message(MessageType.USER, user_request)
        
        # Reload whitelist in case security agent updated it
        self.executor.reload_whitelist()
        
        try:
            while iterations < self.max_iterations:
                iterations += 1
                self.logger.debug(f"Iteration {iterations}/{self.max_iterations}")
                
                # Get LLM response (model loads on-demand)
                llm_response = self.send_to_llm()
                
                # Get model info after first successful load
                if model_info is None and self.model_manager.is_loaded():
                    model_info = self.model_manager.get_model_info()
                
                # Add response to conversation
                self.conversation.add_message(MessageType.ASSISTANT, llm_response)
                
                # Parse response for action
                action_type, content = self.parse_llm_response(llm_response)
                
                if action_type == 'execute':
                    # Execute command
                    self.logger.info(f"Executing: {content}")
                    execution_result = self.executor.execute_command(content)
                    commands_executed += 1
                    
                    # Add execution result to conversation
                    result_msg = f"Command: {execution_result.command}\nOutput:\n{execution_result.output}"
                    self.conversation.add_message(MessageType.COMMAND_OUTPUT, result_msg)
                    
                    # Log execution
                    status = "SUCCESS" if execution_result.success else "FAILED"
                    self.logger.info(f"Command {status}: {content}")
                
                elif action_type == 'complete':
                    # Task completed
                    total_time = time.time() - start_time
                    self.session_stats['tasks_completed'] += 1
                    self.session_stats['total_processing_time'] += total_time
                    
                    return {
                        'status': 'completed',
                        'result': content,
                        'metrics': {
                            'iterations': iterations,
                            'commands_executed': commands_executed,
                            'total_time': total_time,
                            'llm_calls': self.session_stats['llm_calls'],
                            'model_load_count': self.session_stats['model_load_count'],
                            'avg_model_load_time': (
                                self.session_stats['total_model_load_time'] / self.session_stats['model_load_count']
                                if self.session_stats['model_load_count'] > 0 else 0.0
                            )
                        },
                        'conversation_summary': self.conversation.get_summary(),
                        'execution_history': [
                            {
                                'command': exec.command,
                                'success': exec.success,
                                'exit_code': exec.exit_code,
                                'execution_time': exec.execution_time,
                                'whitelist_entry': exec.whitelist_entry
                            }
                            for exec in self.executor.execution_history
                        ],
                        'model_info': model_info,
                        'timestamp': datetime.now().isoformat()
                    }
                
                elif action_type == 'clarification':
                    # Need clarification
                    return {
                        'status': 'needs_clarification',
                        'message': content,
                        'metrics': {
                            'iterations': iterations,
                            'commands_executed': commands_executed
                        },
                        'conversation_summary': self.conversation.get_summary(),
                        'model_info': model_info,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Continue to next iteration for 'continue' action
            
            # Maximum iterations reached
            total_time = time.time() - start_time
            self.session_stats['total_processing_time'] += total_time
            
            return {
                'status': 'max_iterations_reached',
                'message': 'Task did not complete within maximum iterations',
                'last_response': llm_response,
                'metrics': {
                    'iterations': iterations,
                    'commands_executed': commands_executed,
                    'total_time': total_time
                },
                'conversation_summary': self.conversation.get_summary(),
                'model_info': model_info,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error during task execution: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'metrics': {
                    'iterations': iterations,
                    'commands_executed': commands_executed
                },
                'conversation_summary': self.conversation.get_summary(),
                'model_info': model_info,
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            # Always unload model to free resources
            self.model_manager.unload_model()
            self.logger.debug("Executor model unloaded after task completion")
    
    def update_whitelist(self, updates: List[Dict[str, Any]]):
        """Update whitelist (called by security agent)"""
        if updates:
            self.executor.reload_whitelist()
            self.logger.info(f"Whitelist updated with {len(updates)} changes")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including model performance"""
        executor_stats = self.executor.get_statistics()
        conversation_stats = self.conversation.get_summary()
        
        enhanced_session_stats = {
            **self.session_stats,
            'avg_model_load_time': (
                self.session_stats['total_model_load_time'] / self.session_stats['model_load_count']
                if self.session_stats['model_load_count'] > 0 else 0.0
            )
        }
        
        return {
            'session': enhanced_session_stats,
            'executor': executor_stats,
            'conversation': conversation_stats,
            'runtime_info': {
                'max_iterations': self.max_iterations,
                'model_info': self.model_manager.get_model_info()
            }
        }
    
    def test_model_loading(self) -> bool:
        """Test executor model loading capability"""
        try:
            self.logger.info("Testing executor model loading...")
            
            if not self.model_manager.load_model():
                return False
            
            # Try a simple generation
            test_response = self.model_manager.generate("Test executor: ", max_tokens=5)
            
            # Unload model
            self.model_manager.unload_model()
            
            self.logger.info("✅ Executor model loading test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Executor model loading test failed: {e}")
            # Ensure cleanup
            self.model_manager.unload_model()
            return False
    
    def save_execution_log(self, output_dir: str = "logs") -> str:
        """Save execution log"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"execution_log_{timestamp}.json")
        
        log_data = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'statistics': self.get_statistics()
            },
            'conversation_history': [
                {
                    'type': msg.type.value,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat(),
                    'metadata': msg.metadata
                }
                for msg in self.conversation.messages
            ],
            'execution_history': [
                {
                    'command': exec.command,
                    'success': exec.success,
                    'output': exec.output,
                    'exit_code': exec.exit_code,
                    'execution_time': exec.execution_time,
                    'timestamp': exec.timestamp.isoformat(),
                    'whitelist_entry': exec.whitelist_entry
                }
                for exec in self.executor.execution_history
            ]
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Execution log saved: {log_file}")
        return log_file
    
    def cleanup(self):
        """Cleanup resources (call when agent is no longer needed)"""
        self.model_manager.unload_model()
        self.conversation.clear_conversation()
        self.logger.info("Command executor agent cleanup completed")


def main():
    """Standalone testing interface for the llama.cpp executor agent"""
    print("Command Executor Agent - llama.cpp Integration Test")
    print("=" * 50)
    
    # Configuration
    model_path = input("Enter path to your executor GGUF model file: ").strip()
    if not model_path:
        print("No model path provided. Exiting.")
        return
    
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        return
    
    # Optional parameters
    try:
        n_gpu_layers = int(input("GPU layers to offload (default 40): ") or "40")
        n_ctx = int(input("Context size (default 2048): ") or "2048")
    except ValueError:
        n_gpu_layers, n_ctx = 40, 2048
    
    # Initialize agent
    try:
        agent = CommandExecutorAgent(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            model_verbose=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize executor agent: {e}")
        return
    
    # Test model loading
    if not agent.test_model_loading():
        print("❌ Executor model loading test failed. Check your model file and parameters.")
        return
    
    # Available test tasks
    example_tasks = [
        "What is my public IP address?",
        "Check disk space and memory usage",
        "Show me information about this system",
        "Test connectivity to google.com",
        "Calculate 2^10 using Python",
        "List files in current directory"
    ]
    
    print("Available test tasks:")
    for i, task in enumerate(example_tasks, 1):
        print(f"{i}. {task}")
    
    print("\nSelect task (1-6) or enter custom request:")
    user_input = input("> ").strip()
    
    if user_input.isdigit() and 1 <= int(user_input) <= len(example_tasks):
        task = example_tasks[int(user_input) - 1]
    else:
        task = user_input
    
    print(f"\nExecuting: {task}")
    print("-" * 40)
    
    # Execute task
    result = agent.execute_task(task)
    
    # Display results
    print(f"\nStatus: {result['status']}")
    if result['status'] == 'completed':
        print(f"Result: {result['result']}")
    elif result['status'] == 'needs_clarification':
        print(f"Clarification: {result['message']}")
    elif result['status'] == 'error':
        print(f"Error: {result['error']}")
    
    # Show metrics
    metrics = result.get('metrics', {})
    print(f"\nMetrics:")
    print(f"  Iterations: {metrics.get('iterations', 'N/A')}")
    print(f"  Commands: {metrics.get('commands_executed', 'N/A')}")
    print(f"  Time: {metrics.get('total_time', 0):.2f}s")
    print(f"  LLM Calls: {metrics.get('llm_calls', 'N/A')}")
    print(f"  Model Loads: {metrics.get('model_load_count', 'N/A')}")
    if metrics.get('avg_model_load_time'):
        print(f"  Avg Load Time: {metrics['avg_model_load_time']:.2f}s")
    
    # Show model info
    if result.get('model_info'):
        model_info = result['model_info']
        print(f"\nModel Info:")
        print(f"  Model: {model_info['model_name']}")
        print(f"  GPU Layers: {model_info['n_gpu_layers']}")
        print(f"  Context: {model_info['n_ctx']}")
    
    # Save log
    log_file = agent.save_execution_log()
    print(f"\nLog saved: {log_file}")
    
    # Show statistics
    print(f"\nAgent Statistics:")
    stats = agent.get_statistics()
    session_stats = stats['session']
    print(f"  Tasks processed: {session_stats['tasks_processed']}")
    print(f"  Tasks completed: {session_stats['tasks_completed']}")
    print(f"  LLM calls: {session_stats['llm_calls']}")
    print(f"  Model loads: {session_stats['model_load_count']}")
    
    executor_stats = stats['executor']
    print(f"  Successful executions: {executor_stats['successful_executions']}")
    print(f"  Security violations: {executor_stats['security_violations']}")
    
    # Cleanup
    agent.cleanup()
    print("\n✅ Executor agent cleanup completed")


if __name__ == "__main__":
    main()