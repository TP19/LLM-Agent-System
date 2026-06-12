#!/usr/bin/env python3
"""
Lazy Model Manager - Load models only when needed
Supports both llama-cpp-python and llama-server backends
"""

import os
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Try to import llama_cpp, but don't fail if it's not available or has API issues
LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # Either not installed, or API mismatch with libllama.so
    # This is expected - we'll use llama-server backend instead
    print(f"⚠️  llama-cpp-python not available ({type(e).__name__}), will use llama-server backend")
    Llama = None

from core.llama_server_backend import LlamaServerBackend

class LazyModelManager:
    """Load models only when needed, unload when done"""

    def __init__(self, models_config: Dict[str, Any], base_port: int = 8090):
        self.models_config = models_config
        self.current_model = None
        self.current_agent = None
        self.load_times = {}
        self.logger = logging.getLogger("LazyModelManager")
        self.base_port = base_port  # Base port for llama-server instances
        self.port_counter = 0  # Counter for unique ports
    
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
        """Load model with timing and logging - tries llama-cpp-python first, falls back to llama-server"""

        start_time = time.time()
        self.logger.info(f"🔄 Loading model for {agent_name}...")

        config = self.models_config.get(agent_name, {})
        model_path = config.get('model_path')
        if model_path:
            model_path = os.path.expanduser(model_path)

        if not model_path or not Path(model_path).exists():
            raise RuntimeError(f"Model not found for {agent_name}: {model_path}")

        # Check backend preference (auto = try both with fallback)
        backend = config.get('backend', 'auto')

        model = None
        errors = []

        # Strategy based on backend preference
        if backend == 'llama-server':
            # User explicitly wants llama-server
            model = self._load_with_server(agent_name, config, model_path)

        elif backend == 'llama-cpp-python':
            # User explicitly wants llama-cpp-python
            model = self._load_with_python_bindings(agent_name, config, model_path)

        else:  # 'auto' - try both with smart fallback
            # Try llama-cpp-python first (faster, less overhead)
            try:
                self.logger.info("Attempting llama-cpp-python backend...")
                model = self._load_with_python_bindings(agent_name, config, model_path)
            except Exception as e:
                errors.append(f"llama-cpp-python failed: {e}")
                self.logger.warning(f"⚠️ llama-cpp-python failed, trying llama-server fallback...")

                # Fallback to llama-server
                try:
                    model = self._load_with_server(agent_name, config, model_path)
                except Exception as e2:
                    errors.append(f"llama-server failed: {e2}")
                    raise RuntimeError(
                        f"Both backends failed for {agent_name}:\n" +
                        "\n".join(errors)
                    )

        load_time = time.time() - start_time
        self.load_times[agent_name] = load_time

        self.logger.info(f"✅ Model loaded for {agent_name} in {load_time:.2f}s")
        return model

    def _load_with_python_bindings(self, agent_name: str, config: Dict, model_path: str):
        """Load model using llama-cpp-python"""
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python not available (import failed)")

        model = Llama(
            model_path=model_path,
            n_gpu_layers=config.get('n_gpu_layers', 30),
            n_ctx=config.get('n_ctx', 4096),
            verbose=config.get('verbose', False)
        )
        self.logger.info(f"✅ Using llama-cpp-python backend")
        return model

    def _find_llama_server(self) -> str:
        """Find llama-server binary, checking PATH and common locations."""
        # Check PATH first
        found = shutil.which('llama-server')
        if found:
            return found
        # Common locations (project bin/ first, then system locations)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for path in [
            os.path.join(project_root, 'bin', 'llama-server'),
            os.path.expanduser('~/llama.cpp/build/bin/llama-server'),
            '/usr/local/bin/llama-server',
            os.path.expanduser('~/.local/bin/llama-server'),
        ]:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            "llama-server not found. Run 'python install.py' or set llama_server_path in config/models.yaml"
        )

    def _load_with_server(self, agent_name: str, config: Dict, model_path: str):
        """Load model using llama-server subprocess"""
        port = self.base_port + self.port_counter
        self.port_counter += 1

        server_path = config.get('llama_server_path')
        if server_path:
            server_path = os.path.expanduser(server_path)
        else:
            server_path = self._find_llama_server()

        model = LlamaServerBackend(
            model_path=model_path,
            port=port,
            n_gpu_layers=config.get('n_gpu_layers', 30),
            n_ctx=config.get('n_ctx', 4096),
            llama_server_path=server_path,
            verbose=config.get('verbose', False),
            use_chat_api=config.get('use_chat_api', False),
            enable_thinking=config.get('enable_thinking', False)
        )
        self.logger.info(f"✅ Using llama-server backend on port {port}")
        return model
    
    def _unload_current_model(self):
        """Unload current model to free memory"""
        if self.current_model:
            self.logger.info(f"🗑️ Unloading model for {self.current_agent}")

            try:
                # If it's a server backend, cleanup properly
                if isinstance(self.current_model, LlamaServerBackend):
                    self.current_model.cleanup()

                # Try to explicitly close the model if possible
                if hasattr(self.current_model, 'close'):
                    self.current_model.close()
            except Exception as e:
                self.logger.warning(f"Model cleanup warning (non-fatal): {e}")

            try:
                del self.current_model
            except Exception as e:
                self.logger.warning(f"Model deletion warning: {e}")

            self.current_model = None
            self.current_agent = None

            # Force garbage collection
            import gc
            gc.collect()
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get the full config dict for an agent"""
        return self.models_config.get(agent_name, {})

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