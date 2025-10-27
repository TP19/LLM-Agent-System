#!/usr/bin/env python3
"""
Base Agent with enhanced logging and lazy model loading
"""

import logging
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class BaseAgent:
    """Base agent with enhanced logging and lazy model loading"""
    
    def __init__(self, agent_name: str, model_manager):
        self.agent_name = agent_name
        self.model_manager = model_manager
        self.logger = logging.getLogger(f"{agent_name}Agent")
        
        # Enhanced logging setup
        self._setup_detailed_logging()
        
        # Performance tracking
        self.stats = {
            'requests_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'model_load_count': 0
        }
    
    def _setup_detailed_logging(self):
        """Setup detailed logging for agent reasoning"""
        
        # Create agent-specific log file
        log_file = Path("logs") / f"{self.agent_name}_detailed.log"
        log_file.parent.mkdir(exist_ok=True)
        
        # Add file handler for detailed logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def generate_with_logging(self, prompt: str, request_id: str, **kwargs) -> str:
        """Generate response with detailed logging"""
        
        # Log the prompt (first 200 chars)
        self.logger.debug(f"[{request_id}] 📝 Prompt preview: {prompt[:200]}...")
        self.logger.debug(f"[{request_id}] ⚙️ Generation params: {kwargs}")
        
        start_time = time.time()
        
        # Get model (triggers lazy loading)
        model = self.model_manager.get_model_for_agent(self.agent_name)
        self.stats['model_load_count'] += 1
        
        # Generate response
        response = model(prompt, **kwargs)
        generation_time = time.time() - start_time
        
        # Log response details
        response_text = response['choices'][0]['text']
        self.logger.debug(f"[{request_id}] ⚡ Generated in {generation_time:.2f}s")
        self.logger.debug(f"[{request_id}] 📄 Response length: {len(response_text)} chars")
        self.logger.debug(f"[{request_id}] 🔤 Response preview: {response_text[:300]}...")
        
        # Log full response to separate file for debugging
        self._log_full_response(request_id, prompt, response_text, generation_time)
        
        return response_text
    
    def _log_full_response(self, request_id: str, prompt: str, response: str, time_taken: float):
        """Log full prompt and response for debugging"""
        
        debug_file = Path("logs") / f"{self.agent_name}_full_responses.log"
        
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"REQUEST ID: {request_id}\n")
            f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
            f.write(f"AGENT: {self.agent_name}\n")
            f.write(f"GENERATION TIME: {time_taken:.2f}s\n")
            f.write(f"\nPROMPT:\n{prompt}\n")
            f.write(f"\nRESPONSE:\n{response}\n")
            f.write(f"{'='*80}\n")
    
    def update_stats(self, processing_time: float):
        """Update agent performance statistics"""
        self.stats['requests_processed'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['average_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['requests_processed']
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            **self.stats,
            'success_rate': self.stats.get('successful_operations', 0) / max(1, self.stats['requests_processed']),
            'avg_processing_time': self.stats['average_processing_time']
        }
    
    def cleanup(self):
        """Cleanup agent resources"""
        # Model cleanup is handled by model manager
        # Agents can override for custom cleanup
        self.logger.info(f"🧹 {self.agent_name} agent cleanup complete")