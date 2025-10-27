#!/usr/bin/env python3
"""
Lazy Model Manager - Load models only when needed
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from llama_cpp import Llama
except ImportError:
    print("Install llama-cpp-python: pip install llama-cpp-python")
    exit(1)

class LazyModelManager:
    """Load models only when needed, unload when done"""
    
    def __init__(self, models_config: Dict[str, Any]):
        self.models_config = models_config
        self.current_model = None
        self.current_agent = None
        self.load_times = {}
        self.logger = logging.getLogger("LazyModelManager")
    
    def get_model_for_agent(self, agent_name: str):
        """Load model for specific agent, unload others"""
        
        if self.current_agent == agent_name and self.current_model:
            # Already loaded for this agent
            return self.current_model
        
        # Unload current model if different agent
        if self.current_model and self.current_agent != agent_name:
            self._unload_current_model()
        
        # Load new model
        model = self._load_model_for_agent(agent_name)
        self.current_model = model
        self.current_agent = agent_name
        
        return model
    
    def _load_model_for_agent(self, agent_name: str):
        """Load model with timing and logging"""
        
        start_time = time.time()
        self.logger.info(f"🔄 Loading model for {agent_name}...")
        
        config = self.models_config.get(agent_name, {})
        model_path = config.get('model_path')
        
        if not model_path or not Path(model_path).exists():
            raise RuntimeError(f"Model not found for {agent_name}: {model_path}")
        
        model = Llama(
            model_path=model_path,
            n_gpu_layers=config.get('n_gpu_layers', 30),
            n_ctx=config.get('n_ctx', 4096),
            verbose=False
        )
        
        load_time = time.time() - start_time
        self.load_times[agent_name] = load_time
        
        self.logger.info(f"✅ Model loaded for {agent_name} in {load_time:.2f}s")
        return model
    
    def _unload_current_model(self):
        """Unload current model to free memory"""
        if self.current_model:
            self.logger.info(f"🗑️ Unloading model for {self.current_agent}")
            del self.current_model
            self.current_model = None
            self.current_agent = None
            
            # Force garbage collection
            import gc
            gc.collect()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "current_agent": self.current_agent,
            "model_loaded": self.current_model is not None,
            "load_times": self.load_times
        }
    
    def cleanup(self):
        """Cleanup all resources"""
        self._unload_current_model()