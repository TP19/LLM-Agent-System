#!/usr/bin/env python3
"""
File Monitor - Multi-Agent Request Orchestrator with llama.cpp Integration

This is the main orchestrator that watches for new requests and coordinates
the complete flow using llama.cpp models loaded on-demand.

Key features:
- Real-time file monitoring with watchdog
- On-demand model loading for each agent
- Complete A-to-B workflow orchestration
- Human approval workflow management
- Comprehensive logging and statistics
- Automatic archiving and cleanup
- Resource-efficient model management
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
from typing import Dict, Any, Optional
import uuid

# Import all agents with llama.cpp support
from triage_agent import TriageAgent, TriageAnalysis
from security_agent import SecurityAgent, SecurityAnalysis
from command_executor_agent import CommandExecutorAgent

class ModelConfiguration:
    """
    Manages model configuration for all agents
    
    This centralized configuration allows easy management of model paths,
    GPU settings, and other parameters across all agents.
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Validate model files exist
        self._validate_model_files()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load model config: {e}")
        
        # Create default configuration
        return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default model configuration"""
        config = {
            'models': {
                'triage': {
                    'model_path': 'models/triage_model.gguf',
                    'n_gpu_layers': 40,
                    'n_ctx': 2048,
                    'verbose': False
                },
                'security': {
                    'model_path': 'models/security_model.gguf',
                    'n_gpu_layers': 40,
                    'n_ctx': 2048,
                    'verbose': False
                },
                'executor': {
                    'model_path': 'models/executor_model.gguf',
                    'n_gpu_layers': 40,
                    'n_ctx': 2048,
                    'verbose': False
                }
            },
            'global_settings': {
                'auto_detect_gpu_layers': True,
                'fallback_to_cpu': True,
                'model_timeout': 300,  # 5 minutes max for model operations
                'cleanup_interval': 3600  # Cleanup every hour
            }
        }
        
        # Save default config
        self._save_config(config)
        return config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def _validate_model_files(self):
        """Validate that configured model files exist"""
        missing_models = []
        
        for agent_name, model_config in self.config.get('models', {}).items():
            model_path = Path(model_config.get('model_path', ''))
            if not model_path.exists():
                missing_models.append(f"{agent_name}: {model_path}")
        
        if missing_models:
            print("⚠️  Warning: Missing model files:")
            for missing in missing_models:
                print(f"   - {missing}")
            print("\nUpdate config/models.yaml with correct paths or download models.")
    
    def get_model_config(self, agent_name: str) -> Dict[str, Any]:
        """Get model configuration for specific agent"""
        return self.config.get('models', {}).get(agent_name, {})
    
    def get_global_settings(self) -> Dict[str, Any]:
        """Get global model settings"""
        return self.config.get('global_settings', {})

class RequestProcessor(FileSystemEventHandler):
    """
    Handles new request files and orchestrates the complete processing flow
    
    This is the heart of the multi-agent system - it coordinates all the agents
    using llama.cpp models loaded on-demand for optimal resource usage.
    """
    
    def __init__(self, 
                 watch_dir: str = "requests", 
                 logs_dir: str = "logs",
                 archive_dir: str = "processed",
                 pending_dir: str = "pending_approval",
                 config_dir: str = "config"):
        
        # Set up directories
        self.watch_dir = Path(watch_dir)
        self.logs_dir = Path(logs_dir)
        self.archive_dir = Path(archive_dir)
        self.pending_approval_dir = Path(pending_dir)
        self.config_dir = Path(config_dir)
        
        # Create directories if they don't exist
        for directory in [self.watch_dir, self.logs_dir, self.archive_dir, 
                         self.pending_approval_dir, self.config_dir]:
            directory.mkdir(exist_ok=True)
        
        # Set up logging
        log_file = self.logs_dir / f"monitor_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("FileMonitor")
        
        # Load model configuration
        try:
            self.model_config = ModelConfiguration(self.config_dir / "models.yaml")
            self.logger.info("Model configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model configuration: {e}")
            raise
        
        # Initialize agents (they will load models on-demand)
        self.agents_initialized = False
        self._initialize_agents()
        
        # Statistics tracking
        self.stats = {
            'requests_processed': 0,
            'requests_completed': 0,
            'requests_pending_approval': 0,
            'security_rejections': 0,
            'execution_failures': 0,
            'clarification_requests': 0,
            'model_load_failures': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info("File Monitor initialized and ready")
    
    def _initialize_agents(self):
        """Initialize agents with model configurations"""
        try:
            # Initialize Triage Agent
            triage_config = self.model_config.get_model_config('triage')
            if not triage_config:
                raise ValueError("No triage model configuration found")
            
            # Validate triage model exists
            triage_model_path = Path(triage_config['model_path'])
            if not triage_model_path.exists():
                raise FileNotFoundError(f"Triage model not found: {triage_model_path}")
            
            self.triage_agent = TriageAgent(
                model_path=str(triage_model_path),
                n_gpu_layers=triage_config.get('n_gpu_layers', 40),
                n_ctx=triage_config.get('n_ctx', 2048),
                model_verbose=triage_config.get('verbose', False)
            )
            
            # Test triage agent
            if not self.triage_agent.test_model_loading():
                raise RuntimeError("Triage agent model loading test failed")
            
            self.logger.info(f"✅ Triage agent initialized with model: {triage_model_path.name}")
            
            # Initialize Security Agent
            security_config = self.model_config.get_model_config('security')
            if not security_config:
                raise ValueError("No security model configuration found")
            
            # Validate security model exists
            security_model_path = Path(security_config['model_path'])
            if not security_model_path.exists():
                raise FileNotFoundError(f"Security model not found: {security_model_path}")
            
            self.security_agent = SecurityAgent(
                model_path=str(security_model_path),
                n_gpu_layers=security_config.get('n_gpu_layers', 40),
                n_ctx=security_config.get('n_ctx', 2048),
                model_verbose=security_config.get('verbose', False)
            )
            
            # Test security agent
            if not self.security_agent.test_model_loading():
                raise RuntimeError("Security agent model loading test failed")
            
            self.logger.info(f"✅ Security agent initialized with model: {security_model_path.name}")
            
            # Initialize Executor Agent
            executor_config = self.model_config.get_model_config('executor')
            if not executor_config:
                raise ValueError("No executor model configuration found")
            
            # Validate executor model exists
            executor_model_path = Path(executor_config['model_path'])
            if not executor_model_path.exists():
                raise FileNotFoundError(f"Executor model not found: {executor_model_path}")
            
            self.executor_agent = CommandExecutorAgent(
                model_path=str(executor_model_path),
                n_gpu_layers=executor_config.get('n_gpu_layers', 40),
                n_ctx=executor_config.get('n_ctx', 2048),
                model_verbose=executor_config.get('verbose', False)
            )
            
            # Test executor agent
            if not self.executor_agent.test_model_loading():
                raise RuntimeError("Executor agent model loading test failed")
            
            self.logger.info(f"✅ Executor agent initialized with model: {executor_model_path.name}")
            
            self.agents_initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            self.agents_initialized = False
            raise
    
    def on_created(self, event):
        """Handle new file creation events"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process .txt files and ignore temporary/hidden files
        if (file_path.suffix.lower() == '.txt' and 
            not file_path.name.startswith('.') and
            not file_path.name.startswith('~')):
            
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            self.process_request_file(file_path)
    
    def process_request_file(self, file_path: Path):
        """
        Process a single request file through the complete pipeline
        
        This method orchestrates the complete flow with on-demand model loading:
        1. Read and validate request
        2. Triage analysis (model loaded/unloaded automatically)
        3. Security analysis (model loaded/unloaded automatically)
        4. Execution (model loaded/unloaded automatically)
        5. Archive results
        """
        request_id = str(uuid.uuid4())[:8]
        
        try:
            self.logger.info(f"[{request_id}] Processing: {file_path.name}")
            self.stats['requests_processed'] += 1
            
            # Check if agents are initialized
            if not self.agents_initialized:
                raise RuntimeError("Agents not properly initialized")
            
            # Step 1: Read and validate request
            request_content = self._read_request_file(file_path, request_id)
            if not request_content:
                return
            
            # Step 2: Triage Analysis with on-demand model loading
            self.logger.info(f"[{request_id}] Step 1: Triage analysis (loading model...)")
            
            try:
                triage_result = self.triage_agent.analyze_request(request_content)
            except Exception as e:
                self.logger.error(f"[{request_id}] Triage analysis failed: {e}")
                self.stats['model_load_failures'] += 1
                self._archive_file(file_path, "triage_error", {
                    'request_id': request_id,
                    'error': str(e),
                    'stage': 'triage_analysis'
                })
                return
            
            # Log triage results
            self._save_step_log(request_id, 'triage', {
                'request_file': file_path.name,
                'request_content': request_content,
                'triage_result': {
                    'category': triage_result.category.value,
                    'difficulty': triage_result.difficulty.value,
                    'confidence': triage_result.confidence,
                    'uncertainty_factors': triage_result.uncertainty_factors,
                    'recommended_action': triage_result.recommended_action,
                    'needs_clarification': triage_result.needs_clarification,
                    'processing_time': triage_result.processing_time,
                    'model_info': triage_result.model_info
                }
            })
            
            # Handle clarification needed
            if triage_result.needs_clarification or triage_result.confidence < 0.3:
                self.logger.info(f"[{request_id}] Request needs clarification")
                self.stats['clarification_requests'] += 1
                self._move_to_pending_clarification(file_path, request_id, triage_result)
                return
            
            # Step 3: Security Analysis with on-demand model loading
            self.logger.info(f"[{request_id}] Step 2: Security analysis (loading model...)")
            
            try:
                security_result = self.security_agent.analyze_request(request_content, triage_result)
            except Exception as e:
                self.logger.error(f"[{request_id}] Security analysis failed: {e}")
                self.stats['model_load_failures'] += 1
                self._archive_file(file_path, "security_error", {
                    'request_id': request_id,
                    'error': str(e),
                    'stage': 'security_analysis'
                })
                return
            
            # Log security results
            self._save_step_log(request_id, 'security', {
                'security_result': {
                    'risk_level': security_result.risk_level.value,
                    'approval_status': security_result.approval_status.value,
                    'risk_factors': security_result.risk_factors,
                    'whitelist_updates': security_result.whitelist_updates,
                    'reasoning': security_result.reasoning,
                    'confidence': security_result.confidence,
                    'processing_time': security_result.processing_time,
                    'model_info': security_result.model_info
                }
            })
            
            # Handle security decisions
            if security_result.approval_status.value == 'requires_human_approval':
                self.logger.info(f"[{request_id}] Requires human approval")
                self.stats['requests_pending_approval'] += 1
                self._move_to_pending_approval(file_path, request_id, triage_result, security_result)
                return
            
            elif security_result.approval_status.value == 'rejected':
                self.logger.warning(f"[{request_id}] Security rejected")
                self.stats['security_rejections'] += 1
                self._archive_file(file_path, "rejected", {
                    'request_id': request_id,
                    'reason': 'security_rejection',
                    'security_reasoning': security_result.reasoning,
                    'risk_level': security_result.risk_level.value,
                    'risk_factors': security_result.risk_factors
                })
                return
            
            # Step 4: Execute if approved
            if security_result.approval_status.value == 'approved':
                self.logger.info(f"[{request_id}] Step 3: Command execution (loading model...)")
                
                # Apply whitelist updates first
                if security_result.whitelist_updates:
                    self.security_agent.update_whitelist(security_result.whitelist_updates)
                    self.executor_agent.update_whitelist(security_result.whitelist_updates)
                    self.logger.info(f"[{request_id}] Applied {len(security_result.whitelist_updates)} whitelist updates")
                
                # Execute the task
                try:
                    execution_result = self.executor_agent.execute_task(request_content)
                except Exception as e:
                    self.logger.error(f"[{request_id}] Execution failed: {e}")
                    self.stats['execution_failures'] += 1
                    self._archive_file(file_path, "execution_error", {
                        'request_id': request_id,
                        'error': str(e),
                        'stage': 'execution'
                    })
                    return
                
                # Log execution results
                self._save_step_log(request_id, 'execution', {
                    'execution_result': execution_result
                })
                
                # Update statistics and archive
                if execution_result['status'] == 'completed':
                    self.stats['requests_completed'] += 1
                    self.logger.info(f"[{request_id}] Task completed successfully")
                    
                    self._archive_file(file_path, "completed", {
                        'request_id': request_id,
                        'status': execution_result['status'],
                        'execution_summary': execution_result.get('result', 'No result summary'),
                        'commands_executed': execution_result.get('metrics', {}).get('commands_executed', 0),
                        'total_time': execution_result.get('metrics', {}).get('total_time', 0),
                        'model_info': execution_result.get('model_info')
                    })
                elif execution_result['status'] == 'needs_clarification':
                    self.logger.info(f"[{request_id}] Execution needs clarification")
                    self.stats['clarification_requests'] += 1
                    self._move_to_pending_clarification_execution(file_path, request_id, triage_result, security_result, execution_result)
                else:
                    self.stats['execution_failures'] += 1
                    self.logger.warning(f"[{request_id}] Task failed: {execution_result.get('status', 'unknown')}")
                    
                    self._archive_file(file_path, "failed", {
                        'request_id': request_id,
                        'status': execution_result['status'],
                        'error_details': execution_result.get('error', 'Unknown error'),
                        'last_response': execution_result.get('last_response', ''),
                        'commands_executed': execution_result.get('metrics', {}).get('commands_executed', 0)
                    })
        
        except Exception as e:
            self.logger.error(f"[{request_id}] Error processing request: {e}")
            self._archive_file(file_path, "error", {
                'request_id': request_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    def _read_request_file(self, file_path: Path, request_id: str) -> Optional[str]:
        """Read and validate request file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                self.logger.warning(f"[{request_id}] Empty request file: {file_path.name}")
                self._archive_file(file_path, "empty", {'request_id': request_id})
                return None
            
            # Basic validation
            if len(content) > 1000:  # Reasonable limit
                self.logger.warning(f"[{request_id}] Request too long ({len(content)} chars)")
                content = content[:1000] + "...[truncated]"
            
            return content
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Error reading file {file_path}: {e}")
            self._archive_file(file_path, "read_error", {
                'request_id': request_id,
                'error': str(e)
            })
            return None
    
    def _save_step_log(self, request_id: str, step: str, data: Dict[str, Any]):
        """Save detailed log for each processing step"""
        log_data = {
            'request_id': request_id,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **data
        }
        
        log_file = self.logs_dir / f"{step}_{request_id}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def _move_to_pending_clarification(self, file_path: Path, request_id: str, triage_result: TriageAnalysis):
        """Move request to pending clarification folder"""
        pending_file = self.pending_approval_dir / f"clarification_{request_id}_{file_path.name}"
        
        # Create metadata file with instructions
        metadata = {
            'request_id': request_id,
            'type': 'clarification_needed',
            'original_file': file_path.name,
            'triage_result': {
                'category': triage_result.category.value,
                'confidence': triage_result.confidence,
                'uncertainty_factors': triage_result.uncertainty_factors,
                'reasoning': triage_result.reasoning,
                'processing_time': triage_result.processing_time,
                'model_info': triage_result.model_info
            },
            'instructions': [
                'This request needs clarification due to low confidence or ambiguity.',
                'Review the uncertainty factors and either:',
                '1. Create approved_[request_id].txt to approve as-is',
                '2. Create rejected_[request_id].txt to reject',
                '3. Create clarified_[request_id].txt with a clearer version of the request'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.pending_approval_dir / f"clarification_{request_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Move the original file
        shutil.move(str(file_path), str(pending_file))
        self.logger.info(f"[{request_id}] Moved to pending clarification: {pending_file}")
    
    def _move_to_pending_approval(self, file_path: Path, request_id: str, triage_result: TriageAnalysis, security_result: SecurityAnalysis):
        """Move request to pending human approval folder"""
        pending_file = self.pending_approval_dir / f"approval_{request_id}_{file_path.name}"
        
        # Create comprehensive metadata file
        metadata = {
            'request_id': request_id,
            'type': 'requires_approval',
            'original_file': file_path.name,
            'triage_result': {
                'category': triage_result.category.value,
                'confidence': triage_result.confidence,
                'reasoning': triage_result.reasoning,
                'processing_time': triage_result.processing_time,
                'model_info': triage_result.model_info
            },
            'security_result': {
                'risk_level': security_result.risk_level.value,
                'risk_factors': security_result.risk_factors,
                'reasoning': security_result.reasoning,
                'confidence': security_result.confidence,
                'processing_time': security_result.processing_time,
                'model_info': security_result.model_info
            },
            'instructions': [
                'This request requires human approval due to security concerns.',
                'Review the risk factors and reasoning above.',
                'To approve: create approved_[request_id].txt',
                'To reject: create rejected_[request_id].txt',
                'The system will automatically process your decision.'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.pending_approval_dir / f"approval_{request_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Move the original file
        shutil.move(str(file_path), str(pending_file))
        self.logger.info(f"[{request_id}] Moved to pending approval: {pending_file}")
    
    def _move_to_pending_execution(self, file_path: Path, request_id: str, triage_result: TriageAnalysis, security_result: SecurityAnalysis):
        """Move request to pending execution (temporary until executor agent is converted)"""
        pending_file = self.pending_approval_dir / f"pending_exec_{request_id}_{file_path.name}"
        
        # Create metadata file
        metadata = {
            'request_id': request_id,
            'type': 'awaiting_executor_implementation',
            'original_file': file_path.name,
            'triage_result': {
                'category': triage_result.category.value,
                'difficulty': triage_result.difficulty.value,
                'confidence': triage_result.confidence,
                'uncertainty_factors': triage_result.uncertainty_factors,
                'recommended_action': triage_result.recommended_action,
                'reasoning': triage_result.reasoning,
                'processing_time': triage_result.processing_time,
                'model_info': triage_result.model_info
            },
            'security_result': {
                'risk_level': security_result.risk_level.value,
                'approval_status': security_result.approval_status.value,
                'risk_factors': security_result.risk_factors,
                'whitelist_updates': security_result.whitelist_updates,
                'reasoning': security_result.reasoning,
                'confidence': security_result.confidence,
                'processing_time': security_result.processing_time,
                'model_info': security_result.model_info
            },
            'status': 'Triage and Security analysis completed successfully. Awaiting executor agent conversion to llama.cpp.',
            'next_steps': [
                'Executor agent will be converted to llama.cpp in next iteration',
                'Full pipeline will then be available for automatic execution',
                'Whitelist updates have been applied if any were generated'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.pending_approval_dir / f"pending_exec_{request_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Move the original file
        shutil.move(str(file_path), str(pending_file))
        self.logger.info(f"[{request_id}] Moved to pending execution: {pending_file}")
    
    def _move_to_pending_clarification_execution(self, file_path: Path, request_id: str, triage_result: TriageAnalysis, 
                                                security_result: SecurityAnalysis, execution_result: Dict[str, Any]):
        """Move request to pending clarification after execution needs more info"""
        pending_file = self.pending_approval_dir / f"exec_clarification_{request_id}_{file_path.name}"
        
        # Create metadata file
        metadata = {
            'request_id': request_id,
            'type': 'execution_clarification_needed',
            'original_file': file_path.name,
            'triage_result': {
                'category': triage_result.category.value,
                'confidence': triage_result.confidence,
                'reasoning': triage_result.reasoning,
                'model_info': triage_result.model_info
            },
            'security_result': {
                'risk_level': security_result.risk_level.value,
                'approval_status': security_result.approval_status.value,
                'reasoning': security_result.reasoning,
                'model_info': security_result.model_info
            },
            'execution_result': {
                'status': execution_result['status'],
                'message': execution_result.get('message', ''),
                'metrics': execution_result.get('metrics', {}),
                'model_info': execution_result.get('model_info')
            },
            'instructions': [
                'The executor needs clarification to complete the task.',
                'Review the execution message and either:',
                '1. Create clarified_[request_id].txt with additional details',
                '2. Create rejected_[request_id].txt to reject the request',
                '3. Create approved_[request_id].txt to approve with current info'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.pending_approval_dir / f"exec_clarification_{request_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Move the original file
        shutil.move(str(file_path), str(pending_file))
        self.logger.info(f"[{request_id}] Moved to pending clarification (execution): {pending_file}")
    
    def _archive_file(self, file_path: Path, status: str, metadata: Dict[str, Any] = None):
        """Archive processed file with comprehensive metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{timestamp}_{status}_{file_path.name}"
        archive_path = self.archive_dir / archive_name
        
        # Move the file
        try:
            shutil.move(str(file_path), str(archive_path))
            
            # Save metadata if provided
            if metadata:
                metadata_enhanced = {
                    **metadata,
                    'original_filename': file_path.name,
                    'archived_filename': archive_name,
                    'archive_timestamp': datetime.now().isoformat(),
                    'status': status
                }
                
                metadata_file = self.archive_dir / f"{timestamp}_{status}_{file_path.stem}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata_enhanced, f, indent=2)
            
            self.logger.info(f"Archived as {status}: {archive_path}")
            
        except Exception as e:
            self.logger.error(f"Error archiving file {file_path}: {e}")
    
    def check_pending_approvals(self):
        """
        Check for human approval decisions in pending_approval directory
        
        This method processes human decisions by looking for specially named files
        that indicate approval or rejection decisions.
        """
        try:
            # Look for approval decision files
            approval_files = list(self.pending_approval_dir.glob("approved_*.txt"))
            rejection_files = list(self.pending_approval_dir.glob("rejected_*.txt"))
            clarified_files = list(self.pending_approval_dir.glob("clarified_*.txt"))
            
            # Process approvals
            for approval_file in approval_files:
                self._process_human_decision(approval_file, decision_type='approved')
            
            # Process rejections  
            for rejection_file in rejection_files:
                self._process_human_decision(rejection_file, decision_type='rejected')
            
            # Process clarified requests
            for clarified_file in clarified_files:
                self._process_clarified_request(clarified_file)
                
        except Exception as e:
            self.logger.error(f"Error checking pending approvals: {e}")
    
    def _process_human_decision(self, decision_file: Path, decision_type: str):
        """Process human approval/rejection decision"""
        try:
            # Extract request_id from filename (approved_[request_id].txt)
            request_id = decision_file.name.split('_')[1].split('.')[0]
            
            # Find the original request file
            original_file = None
            for pending_file in self.pending_approval_dir.glob(f"*_{request_id}_*"):
                if not pending_file.name.endswith('_metadata.json') and pending_file != decision_file:
                    original_file = pending_file
                    break
            
            if not original_file:
                self.logger.error(f"Could not find original file for decision {request_id}")
                decision_file.unlink()  # Clean up orphaned decision file
                return
            
            # Read the original request
            with open(original_file, 'r') as f:
                request_content = f.read().strip()
            
            if decision_type == 'approved':
                self.logger.info(f"[{request_id}] Human approved - executing")
                
                # Execute the approved request
                try:
                    execution_result = self.executor_agent.execute_task(request_content)
                    
                    # Log the execution
                    self._save_step_log(request_id, 'execution_after_approval', {
                        'human_approved': True,
                        'execution_result': execution_result
                    })
                    
                    # Update stats and archive
                    if execution_result['status'] == 'completed':
                        self.stats['requests_completed'] += 1
                        status = "human_approved_completed"
                        summary = execution_result.get('result', 'Task completed successfully')
                    else:
                        self.stats['execution_failures'] += 1
                        status = "human_approved_failed"
                        summary = execution_result.get('error', execution_result.get('message', 'Execution failed'))
                    
                    # Archive the original request
                    self._archive_file(original_file, status, {
                        'request_id': request_id,
                        'human_decision': 'approved',
                        'execution_status': execution_result['status'],
                        'execution_summary': summary,
                        'commands_executed': execution_result.get('metrics', {}).get('commands_executed', 0),
                        'decision_timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    self.logger.error(f"[{request_id}] Execution failed after human approval: {e}")
                    self.stats['execution_failures'] += 1
                    
                    self._archive_file(original_file, "human_approved_execution_error", {
                        'request_id': request_id,
                        'human_decision': 'approved',
                        'error': str(e),
                        'decision_timestamp': datetime.now().isoformat()
                    })
            
            elif decision_type == 'rejected':
                self.logger.info(f"[{request_id}] Human rejected")
                self.stats['security_rejections'] += 1
                
                # Archive as rejected
                self._archive_file(original_file, "human_rejected", {
                    'request_id': request_id,
                    'human_decision': 'rejected',
                    'decision_timestamp': datetime.now().isoformat(),
                    'reason': 'human_rejection'
                })
            
            # Clean up decision and metadata files
            decision_file.unlink()
            
            # Clean up metadata files
            metadata_files = list(self.pending_approval_dir.glob(f"*_{request_id}_metadata.json"))
            for metadata_file in metadata_files:
                metadata_file.unlink()
                
        except Exception as e:
            self.logger.error(f"Error processing human decision: {e}")
    
    def _process_clarified_request(self, clarified_file: Path):
        """Process a clarified request (resubmit to pipeline)"""
        try:
            # Extract request_id from filename
            request_id = clarified_file.name.split('_')[1].split('.')[0]
            
            # Read the clarified request
            with open(clarified_file, 'r') as f:
                clarified_content = f.read().strip()
            
            self.logger.info(f"[{request_id}] Processing clarified request")
            
            # Create a new request file in the watch directory
            new_request_file = self.watch_dir / f"clarified_{request_id}_{datetime.now().strftime('%H%M%S')}.txt"
            with open(new_request_file, 'w') as f:
                f.write(clarified_content)
            
            # Clean up original files
            clarified_file.unlink()
            
            # Clean up any associated files
            for cleanup_file in self.pending_approval_dir.glob(f"*_{request_id}_*"):
                cleanup_file.unlink()
            
            self.logger.info(f"[{request_id}] Clarified request resubmitted as {new_request_file.name}")
            
        except Exception as e:
            self.logger.error(f"Error processing clarified request: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        runtime = datetime.now() - self.stats['start_time']
        
        # Get agent statistics
        triage_stats = self.triage_agent.get_statistics() if self.triage_agent else {}
        security_stats = self.security_agent.get_statistics() if self.security_agent else {}
        executor_stats = self.executor_agent.get_statistics() if self.executor_agent else {}
        
        return {
            'system_stats': {
                **self.stats,
                'runtime_hours': runtime.total_seconds() / 3600,
                'pending_approval_count': len(list(self.pending_approval_dir.glob("approval_*.txt"))),
                'pending_clarification_count': len(list(self.pending_approval_dir.glob("clarification_*.txt"))),
                'pending_execution_count': len(list(self.pending_approval_dir.glob("pending_exec_*.txt"))),
                'pending_exec_clarification_count': len(list(self.pending_approval_dir.glob("exec_clarification_*.txt"))),
                'agents_initialized': self.agents_initialized
            },
            'agent_stats': {
                'triage': triage_stats,
                'security': security_stats,
                'executor': executor_stats
            },
            'processing_rates': {
                'completion_rate': self.stats['requests_completed'] / max(1, self.stats['requests_processed']),
                'clarification_rate': self.stats['clarification_requests'] / max(1, self.stats['requests_processed']),
                'security_rejection_rate': self.stats['security_rejections'] / max(1, self.stats['requests_processed']),
                'execution_failure_rate': self.stats['execution_failures'] / max(1, self.stats['requests_processed']),
                'model_failure_rate': self.stats['model_load_failures'] / max(1, self.stats['requests_processed'])
            },
            'model_config': {
                'config_path': str(self.model_config.config_path),
                'models_configured': list(self.model_config.config.get('models', {}).keys())
            }
        }
    
    def print_status_summary(self):
        """Print a concise status summary"""
        stats = self.get_statistics()
        system = stats['system_stats']
        
        print(f"📊 System Status:")
        print(f"   Processed: {system['requests_processed']} | "
              f"Completed: {system['requests_completed']} | "
              f"Pending Approval: {system['pending_approval_count']} | "
              f"Clarification: {system['pending_clarification_count']} | "
              f"Execution: {system['pending_execution_count']} | "
              f"Runtime: {system['runtime_hours']:.1f}h")
        
        if not system['agents_initialized']:
            print("   ⚠️  Warning: Not all agents initialized properly")
    
    def cleanup(self):
        """Cleanup all agent resources"""
        try:
            if hasattr(self, 'triage_agent') and self.triage_agent:
                self.triage_agent.cleanup()
                self.logger.info("Triage agent cleanup completed")
            
            if hasattr(self, 'security_agent') and self.security_agent:
                self.security_agent.cleanup()
                self.logger.info("Security agent cleanup completed")
            
            if hasattr(self, 'executor_agent') and self.executor_agent:
                self.executor_agent.cleanup()
                self.logger.info("Executor agent cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def main():
    """Main monitoring loop with comprehensive error handling"""
    print("Multi-Agent Request Processor with llama.cpp Integration")
    print("=" * 60)
    
    try:
        # Initialize the request processor
        processor = RequestProcessor()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("\nTroubleshooting:")
        print("1. Check that all model files exist in the paths specified in config/models.yaml")
        print("2. Ensure llama-cpp-python is installed: pip install llama-cpp-python")
        print("3. Verify your GGUF model files are compatible")
        print("4. Check GPU settings and available VRAM")
        print("5. You can use the same model file for all agents initially")
        return
    
    # Set up file system watcher
    observer = Observer()
    observer.schedule(processor, processor.watch_dir, recursive=False)
    observer.start()
    
    print(f"📁 Monitoring: {processor.watch_dir}")
    print(f"📝 Logs: {processor.logs_dir}")
    print(f"📦 Archive: {processor.archive_dir}")
    print(f"⏳ Pending: {processor.pending_approval_dir}")
    print(f"⚙️  Config: {processor.config_dir}")
    print("\n🚀 System ready! Drop .txt files in requests/ to process them")
    print("🔄 Complete pipeline: Triage → Security → Execution")
    print("⚡ All models load/unload automatically for optimal resource usage")
    print("🛡️  Full security analysis with dynamic whitelist management")
    print("⚙️  Intelligent command execution with conversation context")
    print("⌨️  Press Ctrl+C to stop\n")
    
    try:
        status_counter = 0
        while True:
            # Check for human approval decisions every 10 seconds
            processor.check_pending_approvals()
            
            # Print status summary every 60 seconds (6 cycles)
            status_counter += 1
            if status_counter >= 6:
                processor.print_status_summary()
                status_counter = 0
            
            time.sleep(10)
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        observer.stop()
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        observer.stop()
    
    observer.join()
    
    # Cleanup all resources
    print("\n🧹 Cleaning up resources...")
    processor.cleanup()
    
    # Final statistics
    print("\n📈 Final Statistics:")
    final_stats = processor.get_statistics()
    for category, stats in final_stats.items():
        if category == 'model_config':
            continue
        print(f"\n{category.replace('_', ' ').upper()}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            elif isinstance(value, dict):
                continue  # Skip nested dicts for summary
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()