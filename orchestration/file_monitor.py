#!/usr/bin/env python3
"""
Clean File Monitor - Simple Agent Orchestration

This file monitor orchestrates the three agents with simple, effective communication.
Focus: Get agents to collaborate and complete tasks, not safety theater.
"""

import os
import json
import logging
import time
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Dict, Any, Optional, List
import uuid

# Import our clean agents (updated paths)
import sys

project_root = Path(__file__).parent.parent  # Adjust based on your structure
sys.path.insert(0, str(project_root))

# Import modular agents  
from agents.triage_agent import ModularTriageAgent, TriageResult  
from agents.security_agent import ModularSecurityAgent, SecuritySuggestion  
from agents.executor_agent import ModularExecutorAgent  
from agents.coder_agent import ModularCoderAgent  
from agents.summarization_agent import ModularSummarizationAgent, SummaryDepth

# # Import original agents (will be migrated one by one)
# from test_b.clean_triage_agent import CleanTriageAgent
# from test_b.clean_security_agent import CleanSecurityAgent, SecuritySuggestion  
# from test_b.clean_executor_agent import CleanExecutorAgent
# from test_b.clean_coder_agent import CleanCoderAgent

class CleanRequestProcessor(FileSystemEventHandler):
    """Simplified request processor with clean agent collaboration"""
    
    def __init__(self, 
                 watch_dir: str = "requests",
                 logs_dir: str = "logs", 
                 archive_dir: str = "processed",
                 pending_dir: str = "pending_approval",
                 config_dir: str = "config"):
        
        # Setup directories
        self.watch_dir = Path(watch_dir)
        self.logs_dir = Path(logs_dir)
        self.archive_dir = Path(archive_dir)
        self.pending_dir = Path(pending_dir)
        self.config_dir = Path(config_dir)
        
        for directory in [self.watch_dir, self.logs_dir, self.archive_dir, self.pending_dir, self.config_dir]:
            directory.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize agents
        self.agents_initialized = False
        self._initialize_agents()
        
        # Simple statistics
        self.stats = {
            'requests_processed': 0,
            'successful_completions': 0,
            'collaboration_sessions': 0,
            'avg_processing_time': 0.0,
            'start_time': datetime.now()
        }
        
        self.logger.info("🚀 Clean file monitor initialized")

    def on_created(self, event):
        """Handle new file creation"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process .txt files, ignore temporary/hidden files
        if (file_path.suffix.lower() == '.txt' and 
            not file_path.name.startswith('.') and
            not file_path.name.startswith('~')):
            
            self.logger.info(f"Detected new file: {file_path.name}")
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            self.process_request(file_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        runtime_hours = (datetime.now() - self.stats['start_time']).total_seconds() / 3600
        
        return {
            **self.stats,
            'success_rate': self.stats['successful_completions'] / max(1, self.stats['requests_processed']),
            'runtime_hours': runtime_hours,
            'agents_initialized': self.agents_initialized
        }
    
    def _setup_logging(self):
        """Setup simple logging without overriding agent logging"""
        
        log_file = self.logs_dir / f"clean_monitor_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Don't use basicConfig - it overrides everything
        # Instead, set up just the monitor logger
        self.logger = logging.getLogger("CleanMonitor")
        self.logger.setLevel(logging.INFO)
        
        # File handler for monitor
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler for monitor  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers only if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        config_file = self.config_dir / "models.yaml"
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"✅ Loaded config: {config_file}")
                return config
        except FileNotFoundError:
            self.logger.error(f"❌ Config not found: {config_file}")
            self._create_default_config(config_file)
            return self._load_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            return {}
    
    def _create_default_config(self, config_file: Path):
        """Create default configuration file"""
        default_config = {
            'models': {
                'triage': {
                    'model_path': '/path/to/your/triage_model.gguf',
                    'n_gpu_layers': 30,
                    'n_ctx': 4096
                },
                'security': {
                    'model_path': '/path/to/your/security_model.gguf', 
                    'n_gpu_layers': 30,
                    'n_ctx': 4096
                },
                'executor': {
                    'model_path': '/path/to/your/executor_model.gguf',
                    'n_gpu_layers': 30,
                    'n_ctx': 4096
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"📝 Created default config: {config_file}")
        print(f"⚠️  Please update model paths in {config_file}")
    
    def _initialize_agents(self):
        try:
            from core.model_manager import LazyModelManager
            from agents.triage_agent import ModularTriageAgent
            
            models_config = self.config.get('models', {})
            
            # Initialize lazy model manager
            self.model_manager = LazyModelManager(models_config)
            
            # Test model availability
            self.logger.info("🔍 Testing model availability...")
            for agent_name, config in models_config.items():
                model_path = config.get('model_path')
                if not model_path or not Path(model_path).exists():
                    raise RuntimeError(f"Model not found for {agent_name}: {model_path}")
                self.logger.info(f"✅ Model available for {agent_name}: {Path(model_path).name}")
            
            # Initialize NEW modular triage agent
            self.triage_agent = ModularTriageAgent(self.model_manager)
            
            # Initialize OLD agents (to be migrated one by one)
            self._initialize_legacy_agents(models_config)
            
            self.agents_initialized = True
            self.logger.info("🎉 Hybrid agents initialized - NEW triage, OLD others")
            
        except Exception as e:
            self.logger.error(f"❌ Agent initialization failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.agents_initialized = False

    def _initialize_legacy_agents(self, models_config):
        """Initialize remaining agents - TEMPORARY until migration complete"""

        # Don't load models directly - let them use lazy loading when called
        self.logger.info("⚠️ Using legacy agents without model pre-loading")

        # Create agents without loading models
        self.security_agent = None   # Will initialize on first use
        self.executor_agent = None   # Will initialize on first use  
        self.coder_agent = None      # Will initialize on first use
        self.summarization_agent = None

        # Store config for lazy initialization
        self.legacy_config = models_config

    def _get_security_agent(self):
        """Lazy initialize security agent"""
        if self.security_agent is None:
            from test_c.agents.security_agent import ModularSecurityAgent
            config = self.legacy_config.get('security', {})
            model_path = config.get('model_path')

            if model_path and Path(model_path).exists():
                self.security_agent = ModularSecurityAgent(self.model_manager)
                self.logger.info("⚡ Security agent lazy-loaded")
            else:
                raise RuntimeError(f"Security model not found: {model_path}")

        return self.security_agent

    def _get_executor_agent(self):
        """Lazy initialize executor agent"""
        if self.executor_agent is None:
            from test_c.agents.executor_agent import ModularExecutorAgent
            self.executor_agent = ModularExecutorAgent(self.model_manager)
            self.logger.info("Executor agent lazy-loaded")
        return self.executor_agent

    def _get_coder_agent(self):
        """Lazy initialize coder agent"""
        if self.coder_agent is None:
            from test_b.clean_coder_agent import CleanCoderAgent
            config = self.legacy_config.get('coder', {})
            model_path = config.get('model_path')

            if model_path and Path(model_path).exists():
                self.coder_agent = CleanCoderAgent(
                    model_path=model_path,
                    n_gpu_layers=config.get('n_gpu_layers', 30),
                    n_ctx=config.get('n_ctx', 8192)
                )
                self.logger.info("⚡ Coder agent lazy-loaded")
            else:
                self.coder_agent = None

        return self.coder_agent
    
    def _get_summarization_agent(self):
        """Lazy initialize summarization agent"""
        if self.summarization_agent is None:
            from test_c.agents.summarization_agent import ModularSummarizationAgent
            
            self.summarization_agent = ModularSummarizationAgent(self.model_manager)
            self.logger.info("⚡ Summarization agent lazy-loaded")
        
        return self.summarization_agent
        
    def process_request(self, file_path: Path):
        """Enhanced request processing with proper routing and hard stops"""
        
        if not self.agents_initialized:
            self.logger.error("❌ Agents not initialized, cannot process request")
            return
        
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        self.logger.info(f"[{request_id}] 🔄 Processing: {file_path.name}")
        
        try:
            # 1. Read request
            request_content = self._read_request_file(file_path)
            if not request_content:
                self._archive_file(file_path, "empty", request_id)
                return
            
            # 2. Triage analysis  
            self.logger.info(f"[{request_id}] 📋 Step 1: Triage analysis")
            triage_result = self.triage_agent.analyze(request_content)
            
            self.logger.info(f"[{request_id}] ✅ Triage: {triage_result.category.value} "
                        f"(conf: {triage_result.confidence:.2f})")
            
            # Handle clarification if needed
            if triage_result.needs_clarification:
                self.logger.info(f"[{request_id}] ❓ Needs clarification")
                self._move_to_pending(file_path, request_id, triage_result, "clarification")
                return  # Hard stop
            
            # 3. Route based on category - with proper returns for each path
            result = None
            
            if triage_result.category.value == 'summarization':
                # Handle summarization requests
                self.logger.info(f"[{request_id}] 📚 Routing to summarization")
                result = self._handle_summarization_request(request_content, request_id, triage_result)
                
                # Archive and complete
                processing_time = time.time() - start_time
                self._archive_successful_completion(file_path, request_id, {
                    'request_content': request_content,
                    'triage_result': self._serialize_triage_result(triage_result),
                    'result': result,
                    'processing_time': processing_time
                })
                
                # Update stats
                self._update_stats(processing_time, result.get('status') == 'completed')
                
                # Print summary
                self._print_completion_summary(request_id, result, processing_time)
                return  # Hard stop - don't continue to other handlers
            
            elif triage_result.category.value == 'coding':
                # Handle coding requests
                self.logger.info(f"[{request_id}] 💻 Routing to coding")
                result = self._handle_coding_request(request_content, request_id, triage_result)
                
                # Archive and complete
                processing_time = time.time() - start_time
                self._archive_successful_completion(file_path, request_id, {
                    'request_content': request_content,
                    'triage_result': self._serialize_triage_result(triage_result),
                    'result': result,
                    'processing_time': processing_time
                })
                
                # Update stats
                self._update_stats(processing_time, result.get('status') == 'completed')
                
                # Print summary
                self._print_completion_summary(request_id, result, processing_time)
                return  # Hard stop
            
            else:
                # Handle all other regular requests (sysadmin, fileops, network, etc.)
                self.logger.info(f"[{request_id}] ⚙️ Routing to regular handler")
                result = self._handle_regular_request(request_content, request_id, triage_result)
                
                # Archive and complete
                processing_time = time.time() - start_time
                self._archive_successful_completion(file_path, request_id, {
                    'request_content': request_content,
                    'triage_result': self._serialize_triage_result(triage_result),
                    'result': result,
                    'processing_time': processing_time
                })
                
                # Update stats
                self._update_stats(processing_time, result.get('status') == 'completed')
                
                # Print summary
                self._print_completion_summary(request_id, result, processing_time)
                return  # Hard stop
            
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            self._archive_file(file_path, "error", request_id, {'error': str(e)})
            self._update_stats(time.time() - start_time, False)
# COMPLETE FIX for clean_file_monitor.py

# 1. First, update your triage agent prompt in clean_triage_agent.py
    def _create_practical_prompt(self) -> str:
        """Create a practical, less strict prompt"""
        return """You are a Triage Agent that quickly classifies user requests to route them to the right agent.

BE PRACTICAL, NOT OVERLY CAUTIOUS. If a request is reasonably clear, classify it confidently.

CATEGORIES:
- sysadmin: System tasks (check disk, memory, processes, SSH operations, server maintenance)
- fileops: File operations (find files, backup, directory operations) 
- network: Network tasks (ping, connectivity, downloads, API calls)
- development: Code tasks (git, scripts, building, debugging, CREATE MODULAR SYSTEMS, BUILD FROM SPECIFICATIONS)
- content: Writing, documentation, text generation
- security: High-risk operations (user management, permissions, dangerous commands)
- unknown: Truly unclear requests only

DEVELOPMENT requests include:
- "Create the modular system from specification"
- "Generate Python modules for X"
- "Build directory structure and files"
- "Create agents/classes/modules from design"
- "Build from paste.txt specification"
- Any request about creating code, modules, or project structures

DIFFICULTY:
- simple: Straightforward task, clear command
- moderate: Multi-step or requires some investigation  
- complex: Complex problem-solving needed

CONFIDENCE: Be realistic but not overly strict
- 0.8-1.0: Clear and straightforward
- 0.6-0.8: Minor ambiguity but intent is clear
- 0.4-0.6: Some uncertainty but can make reasonable guess
- 0.0-0.4: Genuinely unclear

Only request clarification if the request is genuinely confusing or dangerous.

FORMAT:
CATEGORY: [category]
DIFFICULTY: [difficulty]  
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
CLARIFICATION: [yes/no]

Request: """

# 2. Then, fix the process_request method in clean_file_monitor.py
# 2. Then, fix the process_request method in clean_file_monitor.py
    def process_request(self, file_path: Path):
        """Process request with stall detection and security takeover - UPDATED"""
        
        if not self.agents_initialized:
            self.logger.error("❌ Agents not initialized, cannot process request")
            return
        
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        self.logger.info(f"[{request_id}] 🔄 Processing: {file_path.name}")
        
        try:
            # 1. Read request
            request_content = self._read_request_file(file_path)
            if not request_content:
                self._archive_file(file_path, "empty", request_id)
                return
            
            # 2. Triage analysis
            self.logger.info(f"[{request_id}] 📋 Step 1: Triage analysis")
            triage_result = self.triage_agent.analyze(request_content)
            
            self.logger.info(f"[{request_id}] ✅ Triage: {triage_result.category.value} "
                        f"(conf: {triage_result.confidence:.2f})")
            
            # Handle clarification if needed
            if triage_result.needs_clarification:
                self.logger.info(f"[{request_id}] ❓ Needs clarification")
                self._move_to_pending(file_path, request_id, triage_result, "clarification")
                return
            
            if triage_result.category.value == 'summarization':
                self.logger.info(f"[{request_id}] 📚 Routing to summarization handler")
                result = self._handle_summarization_request(request_content, request_id, triage_result)
                    
            # 3. Route based on category
            if triage_result.category.value == 'development':
                # Check if this is specifically a coding/building request
                if any(keyword in request_content.lower() for keyword in 
                    ['modular', 'specification', 'create system', 'build structure', 'paste.txt']):
                    result = self._handle_coding_request(request_content, request_id, triage_result)
                else:
                    result = self._handle_regular_request(request_content, request_id, triage_result)
            else:
                # Handle regular requests (existing flow)
                result = self._handle_regular_request(request_content, request_id, triage_result)
            
            # 4. Archive with results
            processing_time = time.time() - start_time
            self._archive_successful_completion(file_path, request_id, {
                'request_content': request_content,
                'triage_result': self._serialize_triage_result(triage_result),
                'result': result,
                'processing_time': processing_time
            })
            
            # Update stats
            self._update_stats(processing_time, True)
            
            # Print summary
            self._print_completion_summary(request_id, result, processing_time)
            
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            self._archive_file(file_path, "error", request_id, {'error': str(e)})
            self._update_stats(time.time() - start_time, False)

    # 3. Add the missing _handle_regular_request method
    def _handle_regular_request(self, request_content: str, request_id: str, triage_result) -> Dict[str, Any]:
        """Handle regular (non-coding) requests with existing flow"""
        
        self.logger.info(f"[{request_id}] ⚙️ Handling regular request")
        
        try:
            # 3. Security collaboration
            self.logger.info(f"[{request_id}] 🛡️ Step 2: Security collaboration")
            security_suggestion = self._get_security_agent().suggest_approach(
                request_content, 
                {
                    'category': triage_result.category.value,
                    'difficulty': triage_result.difficulty.value,
                    'confidence': triage_result.confidence
                }
            )
            
            self.logger.info(f"[{request_id}] ✅ Security suggested {len(security_suggestion.commands)} commands")
            
            # 4. Executor collaboration
            self.logger.info(f"[{request_id}] ⚡ Step 3: Execution with collaboration")
            
            security_dict = {
                'commands': security_suggestion.commands,
                'reasoning': security_suggestion.reasoning,
                'approach': security_suggestion.approach,
                'next_steps': security_suggestion.next_steps,
                'confidence': security_suggestion.confidence
            }
            
            execution_result = self._get_executor_agent().execute_task(request_content, security_dict)
            
            # 5. CHECK FOR STALL - Security takeover logic
            commands_executed = execution_result.get('commands_executed', 0)
            
            if commands_executed == 0 and security_suggestion.commands:
                # EXECUTOR STALLED - SECURITY TAKES CONTROL
                self.logger.warning(f"[{request_id}] 🚨 Executor stalled with 0 commands executed")
                
                takeover_result = self._handle_executor_stall(
                    request_id, request_content, security_suggestion, execution_result
                )
                
                # Merge takeover results into execution result
                execution_result['security_takeover'] = takeover_result
                execution_result['commands_executed'] = takeover_result.get('commands_executed', 0)
                execution_result['successful_commands'] = takeover_result.get('successful_commands', 0)
                execution_result['execution_results'] = takeover_result.get('execution_results', [])
                execution_result['status'] = 'completed_with_security_takeover'
            
            # 6. Check if more collaboration is needed
            elif self._needs_more_collaboration(execution_result):
                self.logger.info(f"[{request_id}] 🔄 Step 4: Follow-up collaboration")
                follow_up_result = self._continue_collaboration(request_id, request_content, execution_result)
                execution_result['follow_up'] = follow_up_result
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Regular request failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'user_request': request_content
            }

    # 4. Add the _handle_coding_request method (the one I provided earlier)
    def _handle_coding_request(self, request_content: str, request_id: str, triage_result) -> Dict[str, Any]:
        """Handle coding/development requests with coder agent - FIXED PATH PARSING"""
        
        self.logger.info(f"[{request_id}] 💻 Handling coding request")
        
        if not hasattr(self, 'coder_agent') or not self.coder_agent:
            self.logger.warning(f"[{request_id}] ⚠️ Coder agent not available, falling back to regular processing")
            return self._handle_regular_request(request_content, request_id, triage_result)
        
        try:
            # 1. Security review of coding task
            self.logger.info(f"[{request_id}] 🛡️ Security review of coding task")
            security_suggestion = self.security_agent.suggest_approach(
                request_content,
                {
                    'category': triage_result.category.value,
                    'difficulty': triage_result.difficulty.value,
                    'confidence': triage_result.confidence
                }
            )
            
            # 2. Parse paths from request - NEW LOGIC
            spec_file_path, base_directory = self._parse_paths_from_request(request_content, request_id)
            
            # 3. Coder agent processes the request
            self.logger.info(f"[{request_id}] ⚡ Coder agent processing specification")
            self.logger.info(f"[{request_id}] 📁 Base directory: {base_directory}")
            self.logger.info(f"[{request_id}] 📄 Spec file: {spec_file_path}")
            
            # Check if this is a specification-based request
            if any(keyword in request_content.lower() for keyword in 
                ['specification', 'modular', 'create system', 'build structure', 'paste.txt']):
                
                # Load the specification content
                spec_content = self._load_specification_content(spec_file_path, request_content, request_id)
                
                # Process the specification with the correct base directory
                coder_result = self._get_coder_agent().process_specification(
                    spec_content,
                    base_path=base_directory
                )
                
                # 4. Security verification of what was created
                if coder_result.success:
                    self.logger.info(f"[{request_id}] 🔍 Security verification of generated code")
                    verification = self._verify_generated_files(coder_result.files_created, request_id)
                    
                    return {
                        'status': 'completed',
                        'type': 'coding_project',
                        'user_request': request_content,
                        'commands_executed': len(coder_result.files_created),
                        'successful_commands': len(coder_result.files_created) if coder_result.success else 0,
                        'failed_commands': len(coder_result.errors),
                        'base_directory': base_directory,
                        'spec_file_path': spec_file_path,
                        'coder_result': {
                            'files_created': coder_result.files_created,
                            'directories_created': coder_result.directories_created,
                            'processing_time': coder_result.processing_time,
                            'success': coder_result.success,
                            'errors': coder_result.errors,
                            'warnings': coder_result.warnings
                        },
                        'security_verification': verification,
                        'insights': [
                            f"Generated {len(coder_result.files_created)} code files",
                            f"Created {len(coder_result.directories_created)} directories",
                            f"Project structure established in {base_directory}"
                        ]
                    }
                else:
                    return {
                        'status': 'failed',
                        'type': 'coding_project',
                        'user_request': request_content,
                        'commands_executed': 0,
                        'successful_commands': 0,
                        'failed_commands': len(coder_result.errors),
                        'base_directory': base_directory,
                        'errors': coder_result.errors,
                        'warnings': coder_result.warnings
                    }
            
            else:
                # Handle other coding requests (not specification-based)
                self.logger.info(f"[{request_id}] ℹ️ Non-specification coding request, using regular flow")
                return self._handle_regular_request(request_content, request_id, triage_result)
                
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Coding request failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'error': str(e),
                'user_request': request_content,
                'commands_executed': 0,
                'successful_commands': 0,
                'failed_commands': 1
            }
        
    def _handle_summarization_request(self, request_content: str, request_id: str, 
                                    triage_result) -> Dict[str, Any]:
        """Handle summarization requests"""
        
        self.logger.info(f"[{request_id}] 📚 Handling summarization request")
        
        try:
            # Extract file path from request
            file_path = self._extract_file_path_from_request(request_content, request_id)
            
            if not file_path:
                return {
                    'status': 'error',
                    'error': 'Could not find file path in request',
                    'user_request': request_content,
                    'suggestion': 'Please provide a file path like: "Summarize /path/to/file.txt"'
                }
            
            # Determine depth based on request
            depth = SummaryDepth.STANDARD  # Default
            if 'quick' in request_content.lower() or 'overview' in request_content.lower():
                depth = SummaryDepth.QUICK
            elif 'deep' in request_content.lower() or 'detailed' in request_content.lower():
                depth = SummaryDepth.DEEP
            
            # Get summarization agent and process
            summarization_agent = self._get_summarization_agent()
            
            self.logger.info(f"[{request_id}] 📄 Summarizing: {file_path}")
            summary_result = summarization_agent.summarize_file(
                file_path=file_path,
                depth=depth,
                force_refresh=False,
                request_id=request_id
            )
            
            # Convert to standard result format
            if summary_result.success:
                return {
                    'status': 'completed',
                    'type': 'summarization',
                    'user_request': request_content,
                    'file_path': file_path,
                    'commands_executed': 1,  # Summarization counts as 1 operation
                    'successful_commands': 1,
                    'failed_commands': 0,
                    'summary_result': {
                        'summary': summary_result.summary,
                        'key_insights': summary_result.key_insights,
                        'total_tokens': summary_result.total_tokens,
                        'chunk_count': summary_result.chunk_count,
                        'processing_time': summary_result.processing_time,
                        'depth': summary_result.depth.value,
                        'cached': False  # TODO: Add cache detection
                    },
                    'insights': summary_result.key_insights,
                    'stats': summarization_agent.get_stats()
                }
            else:
                return {
                    'status': 'failed',
                    'type': 'summarization',
                    'user_request': request_content,
                    'file_path': file_path,
                    'commands_executed': 0,
                    'successful_commands': 0,
                    'failed_commands': 1,
                    'errors': summary_result.errors
                }
                
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Summarization failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'type': 'summarization',
                'error': str(e),
                'user_request': request_content
            }

    def _extract_file_path_from_request(self, request_content: str, request_id: str) -> Optional[str]:
        """Extract file path from summarization request"""
        import re
        
        # Look for file paths in various formats
        patterns = [
            r'/[^\s]+\.(?:txt|md|markdown)',  # Absolute paths
            r'~/[^\s]+\.(?:txt|md)',          # Home directory paths
            r'\.{1,2}/[^\s]+\.(?:txt|md)',    # Relative paths
            r'(?:file|document|path):\s*([^\s]+\.(?:txt|md))',  # "file: /path"
            r'"([^"]+\.(?:txt|md))"',         # Quoted paths
            r"'([^']+\.(?:txt|md))'"          # Single quoted paths
        ]
        
        for pattern in patterns:
            match = re.search(pattern, request_content)
            if match:
                path = match.group(1) if match.lastindex else match.group(0)
                path = path.strip('"\'')
                self.logger.info(f"[{request_id}] 📁 Extracted file path: {path}")
                return path
        
        self.logger.warning(f"[{request_id}] ⚠️ Could not extract file path from request")
        return None

    def _parse_paths_from_request(self, request_content: str, request_id: str) -> tuple[str, str]:
        """Parse file paths and base directory from user request - NEW METHOD"""
        import re
        
        # Look for file paths in the request
        # Pattern: /path/to/file.txt or /path/to/directory/
        path_patterns = [
            r'(/[^\s]+\.txt)',  # Absolute paths ending in .txt
            r'(/[^\s]+/)',      # Absolute directory paths
            r'in\s+([^\s]+\.txt)',  # "in /path/file.txt"
            r'from\s+([^\s]+\.txt)',  # "from /path/file.txt"
            r'use\s+([^\s]+/)\s+as.*project',  # "use /path/ as project directory"
            r'directory\s+([^\s]+/)',  # "directory /path/"
        ]
        
        spec_file_path = None
        base_directory = "."  # Default
        
        # Extract specification file path
        for pattern in path_patterns[:4]:  # First 4 are for files
            match = re.search(pattern, request_content)
            if match:
                potential_path = match.group(1)
                if potential_path.endswith('.txt'):
                    spec_file_path = potential_path
                    self.logger.info(f"[{request_id}] 📄 Found spec file path: {spec_file_path}")
                    break
        
        # Extract base directory
        for pattern in path_patterns[4:]:  # Last patterns are for directories
            match = re.search(pattern, request_content)
            if match:
                potential_dir = match.group(1)
                if potential_dir.endswith('/'):
                    base_directory = potential_dir.rstrip('/')
                    self.logger.info(f"[{request_id}] 📁 Found base directory: {base_directory}")
                    break
        
        # If we found a spec file but no explicit base directory, 
        # use the directory containing the spec file
        if spec_file_path and base_directory == ".":
            base_directory = str(Path(spec_file_path).parent)
            self.logger.info(f"[{request_id}] 📁 Using spec file directory as base: {base_directory}")
        
        # Fallback: look for any absolute path
        if not spec_file_path:
            abs_path_match = re.search(r'(/[^\s]+)', request_content)
            if abs_path_match:
                potential_path = abs_path_match.group(1)
                if potential_path.endswith('.txt'):
                    spec_file_path = potential_path
                    base_directory = str(Path(spec_file_path).parent)
                    self.logger.info(f"[{request_id}] 📄 Fallback spec file: {spec_file_path}")
                    self.logger.info(f"[{request_id}] 📁 Fallback base directory: {base_directory}")
        
        return spec_file_path, base_directory

    def _load_specification_content(self, spec_file_path: str, request_content: str, request_id: str) -> str:
        """Load specification content from file or use request content - NEW METHOD"""
        
        if spec_file_path:
            try:
                spec_path = Path(spec_file_path)
                if spec_path.exists():
                    with open(spec_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.logger.info(f"[{request_id}] ✅ Loaded specification from {spec_file_path}")
                    self.logger.info(f"[{request_id}] 📊 Specification length: {len(content)} characters")
                    return content
                else:
                    self.logger.warning(f"[{request_id}] ⚠️ Specification file not found: {spec_file_path}")
            except Exception as e:
                self.logger.error(f"[{request_id}] ❌ Error loading specification: {e}")
        
        # Fallback to request content
        self.logger.info(f"[{request_id}] 📄 Using request content as specification")
        return request_content

    # 5. Add the verification method
    def _verify_generated_files(self, files_created: List[str], request_id: str) -> Dict[str, Any]:
        """Security verification of generated files"""
        
        verification = {
            'files_checked': len(files_created),
            'safe_files': 0,
            'suspicious_files': 0,
            'issues': []
        }
        
        for file_path in files_created:
            try:
                if Path(file_path).exists():
                    # Basic file safety check
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for obvious security issues
                    if any(danger in content.lower() for danger in 
                        ['rm -rf', 'os.system(', 'eval(', 'exec(', 'subprocess.call']):
                        verification['suspicious_files'] += 1
                        verification['issues'].append(f"Potential security issue in {file_path}")
                    else:
                        verification['safe_files'] += 1
                else:
                    verification['issues'].append(f"File not found: {file_path}")
                    
            except Exception as e:
                verification['issues'].append(f"Error checking {file_path}: {str(e)}")
        
        verification['verification_passed'] = verification['suspicious_files'] == 0
        
        self.logger.info(f"[{request_id}] 🔍 Verification: {verification['safe_files']} safe, "
                        f"{verification['suspicious_files']} suspicious")
        
        return verification
        
    def _read_request_file(self, file_path: Path) -> Optional[str]:
        """Read and validate request file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                self.logger.warning(f"Empty request file: {file_path.name}")
                return None
            
            # Basic validation
            if len(content) > 2000:
                self.logger.warning(f"Request too long, truncating: {file_path.name}")
                content = content[:2000] + "...[truncated]"
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def _needs_more_collaboration(self, execution_result: Dict[str, Any]) -> bool:
        """Determine if more collaboration cycles are needed"""
        
        # Simple heuristic: continue if we have partial success and next steps suggest more work
        if execution_result.get('status') == 'completed':
            next_steps = execution_result.get('next_steps', '').lower()
            
            # Continue if next steps suggest more work
            continue_indicators = ['continue', 'next', 'follow', 'more', 'additional', 'investigate']
            if any(indicator in next_steps for indicator in continue_indicators):
                return True
            
            # Continue if we had some failures that might be recoverable
            if execution_result.get('failed_commands', 0) > 0:
                return True
        
        return False
    
    def _continue_collaboration(self, request_id: str, original_request: str, 
                              execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Continue collaboration for follow-up work"""
        
        self.logger.info(f"[{request_id}] 🤝 Continuing collaboration")
        
        try:
            # Get security analysis of results
            executed_commands = execution_result.get('execution_results', [])
            follow_up_suggestion = self.security_agent.analyze_results(executed_commands, original_request)
            
            # If security suggests follow-up commands, execute them
            if follow_up_suggestion.commands:
                self.logger.info(f"[{request_id}] 🔄 Executing {len(follow_up_suggestion.commands)} follow-up commands")
                
                follow_up_dict = {
                    'commands': follow_up_suggestion.commands,
                    'reasoning': follow_up_suggestion.reasoning,
                    'approach': follow_up_suggestion.approach,
                    'next_steps': follow_up_suggestion.next_steps,
                    'confidence': follow_up_suggestion.confidence
                }
                
                follow_up_result = self.executor_agent.execute_task(original_request, follow_up_dict)
                
                self.stats['collaboration_sessions'] += 1
                
                return follow_up_result
            else:
                return {'status': 'no_follow_up_needed'}
                
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Follow-up collaboration failed: {e}")
            return {'status': 'follow_up_error', 'error': str(e)}
    
    def _move_to_pending(self, file_path: Path, request_id: str, triage_result: TriageResult, reason: str):
        """Move request to pending for human review"""
        
        pending_file = self.pending_dir / f"{reason}_{request_id}_{file_path.name}"
        
        metadata = {
            'request_id': request_id,
            'reason': reason,
            'original_file': file_path.name,
            'triage_result': self._serialize_triage_result(triage_result),
            'timestamp': datetime.now().isoformat(),
            'instructions': f"This request needs {reason}. Review and resubmit if appropriate."
        }
        
        metadata_file = self.pending_dir / f"{reason}_{request_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        shutil.move(str(file_path), str(pending_file))
        self.logger.info(f"[{request_id}] 📋 Moved to pending: {reason}")
    
    def _archive_successful_completion(self, file_path: Path, request_id: str, results: Dict[str, Any]):
        """Archive successfully completed request with results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{timestamp}_completed_{request_id}_{file_path.name}"
        archive_path = self.archive_dir / archive_name
        
        # Save detailed results
        results_file = self.archive_dir / f"{timestamp}_completed_{request_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Move original file
        shutil.move(str(file_path), str(archive_path))
        
        self.logger.info(f"[{request_id}] ✅ Archived as completed: {archive_name}")
    
    def _archive_file(self, file_path: Path, status: str, request_id: str, metadata: Dict = None):
        """Archive file with status"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{timestamp}_{status}_{request_id}_{file_path.name}"
        archive_path = self.archive_dir / archive_name
        
        if metadata:
            metadata_file = self.archive_dir / f"{timestamp}_{status}_{request_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        shutil.move(str(file_path), str(archive_path))
        self.logger.info(f"[{request_id}] 📦 Archived as {status}: {archive_name}")
    
    def _serialize_triage_result(self, result: TriageResult) -> Dict[str, Any]:
        """Convert triage result to serializable dict"""
        return {
            'category': result.category.value,
            'difficulty': result.difficulty.value,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'needs_clarification': result.needs_clarification,
            'processing_time': result.processing_time,
            'timestamp': result.timestamp.isoformat()
        }
    
    def _serialize_security_suggestion(self, suggestion: SecuritySuggestion) -> Dict[str, Any]:
        """Convert security suggestion to serializable dict"""
        return {
            'commands': suggestion.commands,
            'reasoning': suggestion.reasoning,
            'approach': suggestion.approach,
            'next_steps': suggestion.next_steps,
            'confidence': suggestion.confidence,
            'timestamp': suggestion.timestamp.isoformat()
        }
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update processing statistics with takeover tracking - UPDATED"""
        self.stats['requests_processed'] += 1
        
        if success:
            self.stats['successful_completions'] += 1
        
        # Track security takeovers
        if not hasattr(self, 'security_takeovers'):
            self.stats['security_takeovers'] = 0
        
        # Update average processing time
        total_time = self.stats['avg_processing_time'] * (self.stats['requests_processed'] - 1)
        self.stats['avg_processing_time'] = (total_time + processing_time) / self.stats['requests_processed']

    def _print_completion_summary(self, request_id: str, execution_result: Dict[str, Any], processing_time: float):
        """Print human-readable completion summary - UPDATED WITH PATHS"""
        
        print(f"\n{'='*60}")
        print(f"TASK COMPLETED: {request_id}")
        print(f"{'='*60}")
        print(f"Request: {execution_result.get('user_request', 'Unknown')[:60]}...")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Commands executed: {execution_result.get('commands_executed', 0)}")
        print(f"Successful: {execution_result.get('successful_commands', 0)}")
        print(f"Failed: {execution_result.get('failed_commands', 0)}")
        
        # SHOW ACTUAL OUTPUT
        if execution_result.get('execution_results'):
            print(f"\n{'='*60}")
            print("COMMAND OUTPUTS:")
            print(f"{'='*60}")
            for result in execution_result['execution_results']:
                status = "SUCCESS" if result.get('status') == 'success' else "FAILED"
                print(f"\n[{status}] {result.get('command')}")
                if result.get('output'):
                    print(f"Output:\n{result['output']}")
                print("-" * 60)
        
        # Show security takeover if it happened
        if execution_result.get('security_takeover'):
            print(f"\nSecurity Takeover: YES")
            takeover = execution_result['security_takeover']
            if takeover.get('execution_results'):
                print(f"\nSecurity Executed Commands:")
                for result in takeover['execution_results']:
                    status = "SUCCESS" if result.get('success') else "FAILED"
                    print(f"  [{status}] {result.get('command')}")
                    if result.get('output'):
                        # Show first 200 chars of output
                        output = result['output'][:200]
                        print(f"    Output: {output}...")
        
        print(f"\n{'='*60}\n")
        
        if execution_result.get('type') == 'coding_project':
            # Special display for coding projects
            coder_result = execution_result.get('coder_result', {})
            print(f"💻 Type: Code Generation Project")
            
            # Show paths
            if execution_result.get('base_directory'):
                print(f"📁 Base directory: {execution_result['base_directory']}")
            if execution_result.get('spec_file_path'):
                print(f"📄 Specification file: {execution_result['spec_file_path']}")
                
            print(f"📁 Directories created: {len(coder_result.get('directories_created', []))}")
            print(f"📄 Files generated: {len(coder_result.get('files_created', []))}")
            
            if coder_result.get('errors'):
                print(f"❌ Errors: {len(coder_result.get('errors', []))}")
                for error in coder_result.get('errors', [])[:3]:  # Show first 3
                    print(f"   ❌ {error}")
            
            verification = execution_result.get('security_verification', {})
            if verification.get('verification_passed'):
                print(f"🛡️ Security verification: ✅ Passed")
            else:
                print(f"🛡️ Security verification: ⚠️ Issues found")
                for issue in verification.get('issues', [])[:3]:  # Show first 3
                    print(f"   ⚠️ {issue}")
                    
            if execution_result.get('insights'):
                print(f"💡 Key insights:")
                for insight in execution_result['insights']:
                    print(f"   • {insight}")
            
            # Show some created files
            if coder_result.get('files_created'):
                print(f"📋 Sample files created:")
                for file_path in coder_result['files_created'][:5]:  # Show first 5
                    print(f"   ✅ {file_path}")
                if len(coder_result['files_created']) > 5:
                    print(f"   ... and {len(coder_result['files_created']) - 5} more files")

        elif execution_result.get('type') == 'summarization':
            # Special display for summarization
            summary_data = execution_result.get('summary_result', {})
            print(f"📚 Type: Document Summarization")
            print(f"📄 File: {execution_result.get('file_path', 'Unknown')}")
            print(f"📊 Tokens processed: {summary_data.get('total_tokens', 0):,}")
            print(f"📦 Chunks: {summary_data.get('chunk_count', 0)}")
            print(f"⏱️ Processing time: {summary_data.get('processing_time', 0):.2f}s")
            print(f"🎯 Depth: {summary_data.get('depth', 'standard')}")
            
            print(f"\n📋 SUMMARY:")
            print("=" * 60)
            summary_text = summary_data.get('summary', 'No summary generated')
            # Print summary with word wrap
            import textwrap
            for line in textwrap.wrap(summary_text, width=60):
                print(line)
            
            if summary_data.get('key_insights'):
                print(f"\n💡 KEY INSIGHTS:")
                print("=" * 60)
                for insight in summary_data['key_insights']:
                    print(f"  • {insight}")
                                
        else:
            # Regular display for other requests
            print(f"🔧 Commands executed: {execution_result.get('commands_executed', 0)}")
            print(f"✅ Successful: {execution_result.get('successful_commands', 0)}")
            print(f"❌ Failed: {execution_result.get('failed_commands', 0)}")
            
            if execution_result.get('insights'):
                print(f"💡 Key insights:")
                for insight in execution_result['insights'][:3]:  # Show first 3
                    print(f"   • {insight}")
            
            if execution_result.get('next_steps'):
                print(f"📋 Next steps: {execution_result['next_steps']}")
            
            if execution_result.get('follow_up'):
                print(f"🔄 Follow-up collaboration: Yes")
            
            if execution_result.get('security_takeover'):
                print(f"🚨 Security takeover: Yes")
        
        print()

    
    def cleanup(self):
        """Cleanup all agents"""
        try:
            if hasattr(self, 'triage_agent') and self.triage_agent:
                self.triage_agent.cleanup()
            if hasattr(self, 'security_agent') and self.security_agent:
                self.security_agent.cleanup()
            if hasattr(self, 'executor_agent') and self.executor_agent:
                self.executor_agent.cleanup()
            if hasattr(self, 'coder_agent') and self.coder_agent:
                self.coder_agent.cleanup()
            if hasattr(self, 'summarization_agent') and self.summarization_agent:
                self.summarization_agent.cleanup()

            self.logger.info("🧹 Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _handle_executor_stall(self, request_id: str, request_content: str, security_suggestion: SecuritySuggestion, execution_result: Dict) -> Dict[str, Any]:
        """Handle when executor stalls and security agent takes control - NEW METHOD"""
        
        self.logger.warning(f"[{request_id}] 🚨 EXECUTOR STALLED - Security agent taking control")
        
        print(f"\n🚨 EXECUTOR STALLED on {request_id}")
        print("🛡️ SECURITY AGENT TAKING CONTROL")
        print("⚡ Executing commands directly...")
        print()
        
        try:
            # Security agent executes the commands directly
            takeover_result = self.security_agent.execute_commands_directly(
                security_suggestion.commands, 
                request_content
            )
            
            # Update collaboration stats
            self.stats['collaboration_sessions'] += 1
            self.stats['security_takeovers'] = self.stats.get('security_takeovers', 0) + 1
            
            # Print what happened
            print(f"🎯 SECURITY TAKEOVER COMPLETE:")
            print(f"   Commands executed: {takeover_result['commands_executed']}")
            print(f"   Successful: {takeover_result['successful_commands']}")
            print(f"   Failed: {takeover_result['failed_commands']}")
            print()
            
            if takeover_result['execution_results']:
                print(f"📋 RESULTS:")
                for result in takeover_result['execution_results'][:3]:  # Show first 3
                    status = "✅" if result['success'] else "❌"
                    print(f"   {status} {result['command']}")
                    if result['success'] and result['output']:
                        print(f"      Output: {result['output'][:100]}...")
                print()
            
            return takeover_result
            
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Security takeover failed: {e}")
            return {
                'status': 'security_takeover_failed',
                'error': str(e),
                'commands_executed': 0
            }

def main():
    """Main function to run the clean file monitor"""
    
    print("🚀 CLEAN AGENT COLLABORATION SYSTEM")
    print("=" * 50)
    print("Simple, effective agent collaboration without safety theater")
    print("Drop .txt files in 'requests/' directory to get started")
    print()
    
    # Create processor
    processor = CleanRequestProcessor()
    
    if not processor.agents_initialized:
        print("❌ Agents not initialized. Check your config/models.yaml file.")
        return
    
    # Setup file watcher
    observer = Observer()
    observer.schedule(processor, processor.watch_dir, recursive=False)
    
    try:
        observer.start()
        processor.logger.info("👀 File monitor started, watching for requests...")
        
        print(f"👀 Watching: {processor.watch_dir}")
        print("Press Ctrl+C to stop")
        print()
        print(f"📂 Current working directory: {os.getcwd()}")
        print(f"📂 Watching directory: {processor.watch_dir}")
        print(f"📂 Absolute watch path: {processor.watch_dir.absolute()}")
        
        # Keep running
        while True:
            time.sleep(1)
            
            # Print stats periodically
            if int(time.time()) % 30 == 0:  # Every 30 seconds
                stats = processor.get_stats()
                print(f"\n📊 Stats: {stats['requests_processed']} processed, "
                      f"{stats['successful_completions']} completed, "
                      f"{stats['collaboration_sessions']} collaborations, "
                      f"{stats['runtime_hours']:.1f}h runtime")
    
    except KeyboardInterrupt:
        processor.logger.info("👋 Stopping file monitor...")
        observer.stop()
        processor.cleanup()
    
    observer.join()

if __name__ == "__main__":
    main()