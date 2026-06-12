#!/usr/bin/env python3
"""
Llama Server Backend - Use llama.cpp server as subprocess
Bypasses Python binding version issues by using C++ binary directly
"""

import subprocess
import time
import os
import re
import shutil
import requests
import logging
import atexit
import socket
import random
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List


def _default_llama_server_path() -> str:
    """Resolve llama-server binary path.

    Resolution order:
    1. $LLM_ENGINE_LLAMA_SERVER env var
    2. 'llama-server' on $PATH (via shutil.which)
    3. ~/.local/bin/llama-server
    """
    return (
        os.environ.get("LLM_ENGINE_LLAMA_SERVER")
        or shutil.which("llama-server")
        or str(Path.home() / ".local" / "bin" / "llama-server")
    )

from core.retry import retry, exponential_backoff

class LlamaServerBackend:
    """Manages llama-server subprocess with OpenAI-compatible API"""

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    @staticmethod
    def find_process_on_port(port: int) -> Optional[int]:
        """Find process ID using a specific port"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        return proc.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    @staticmethod
    def kill_server_on_port(port: int, logger=None) -> bool:
        """Kill any llama-server process using the specified port"""
        pid = LlamaServerBackend.find_process_on_port(port)
        if pid:
            try:
                proc = psutil.Process(pid)
                # Verify it's a llama-server process
                if 'llama-server' in ' '.join(proc.cmdline()):
                    if logger:
                        logger.info(f"Killing existing llama-server (PID {pid}) on port {port}")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)

                    # Verify it's dead
                    time.sleep(0.5)
                    if LlamaServerBackend.is_port_in_use(port):
                        if logger:
                            logger.warning(f"Port {port} still in use after killing process")
                        return False
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                if logger:
                    logger.warning(f"Failed to kill process on port {port}: {e}")
                return False
        return False

    @staticmethod
    def find_available_port(start_port: int = 8090, max_attempts: int = 100) -> int:
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            if not LlamaServerBackend.is_port_in_use(port):
                return port
        # Fallback to random port
        return random.randint(10000, 65000)

    def __init__(
        self,
        model_path: str,
        port: int = 8090,  # Changed from 8080 to avoid conflicts
        n_gpu_layers: int = 10,
        n_ctx: int = 4096,
        llama_server_path: Optional[str] = None,
        verbose: bool = False,
        auto_cleanup: bool = True,  # Auto-kill existing servers on port
        use_chat_api: bool = False,  # Use /v1/chat/completions instead of /v1/completions
        enable_thinking: bool = False  # Enable reasoning with --reasoning-format deepseek
    ):
        self.model_path = os.path.expanduser(model_path)
        self.llama_server_path = os.path.expanduser(
            llama_server_path if llama_server_path is not None else _default_llama_server_path()
        )
        self.process = None
        self.logger = logging.getLogger("LlamaServerBackend")
        self.verbose = verbose
        self.use_chat_api = use_chat_api
        self.enable_thinking = enable_thinking

        # Verify paths exist
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not Path(self.llama_server_path).exists():
            raise FileNotFoundError(f"llama-server not found: {self.llama_server_path}")

        # Handle port conflicts
        original_port = port
        if self.is_port_in_use(port):
            if auto_cleanup:
                self.logger.info(f"Port {port} in use, attempting cleanup...")
                if self.kill_server_on_port(port, self.logger):
                    self.logger.info(f"Successfully cleaned up port {port}")
                    self.port = port
                else:
                    # Cleanup failed, find alternative port
                    self.logger.warning(f"Failed to clean up port {port}, finding alternative...")
                    self.port = self.find_available_port(start_port=port + 1)
                    self.logger.info(f"Using alternative port {self.port} (original: {original_port})")
            else:
                # No auto-cleanup, find alternative port
                self.logger.warning(f"Port {port} in use, finding alternative...")
                self.port = self.find_available_port(start_port=port + 1)
                self.logger.info(f"Using port {self.port} (original: {original_port})")
        else:
            self.port = port

        # Start server
        self._start_server(n_gpu_layers, n_ctx)

        # Register cleanup
        atexit.register(self.cleanup)

    def _start_server(self, n_gpu_layers: int, n_ctx: int):
        """Start llama-server subprocess"""
        cmd = [
            self.llama_server_path,
            '-m', self.model_path,
            '--port', str(self.port),
            '--n-gpu-layers', str(n_gpu_layers),
            '--ctx-size', str(n_ctx),
            '--log-disable',  # Disable llama.cpp logging
        ]

        # Enable reasoning/thinking support - separates think tokens from content
        # With deepseek format, thinking goes to message.reasoning_content
        # and actual answer goes to message.content (prevents empty responses)
        if self.enable_thinking:
            cmd.extend(['--reasoning-format', 'deepseek'])
            cmd.extend(['--jinja'])  # Required for thinking support

        if self.verbose:
            self.logger.info(f"Starting llama-server: {' '.join(cmd)}")

        # Start process with LLM-Agent-System identifier in environment
        # This allows selective cleanup without killing other llama-server instances
        env = dict(os.environ)
        env['LLM_ENGINE_PROCESS'] = '1'
        env['LLM_ENGINE_PORT'] = str(self.port)

        # Add llama-server's directory to LD_LIBRARY_PATH for bundled .so files
        server_dir = os.path.dirname(self.llama_server_path)
        env['LD_LIBRARY_PATH'] = server_dir + ':' + env.get('LD_LIBRARY_PATH', '')

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL if not self.verbose else None,
            stderr=subprocess.DEVNULL if not self.verbose else None,
            env=env
        )

        # Wait for server to be ready
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 300):
        """Wait for server to respond to health check"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f'http://localhost:{self.port}/health', timeout=1)
                if response.status_code == 200:
                    if self.verbose:
                        self.logger.info(f"Server ready on port {self.port}")
                    return
            except requests.exceptions.RequestException:
                pass

            # Check if process died
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server process died with code {self.process.returncode}")

            time.sleep(0.5)

        raise TimeoutError(f"Server failed to start within {timeout}s")

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        timeout: int = 120,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion - compatible with llama-cpp-python interface

        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            timeout: Request timeout in seconds (default: 120s)
            **kwargs: Additional generation parameters
        """
        payload = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
        }

        if stop:
            payload['stop'] = stop

        # Add any additional kwargs (excluding timeout which is handled separately)
        for k, v in kwargs.items():
            if k != 'timeout':
                payload[k] = v

        return self._make_request(payload, timeout=timeout)

    def _prompt_to_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Convert a completion-style prompt to chat messages.

        Splits the prompt into system + user messages for proper chat template
        handling. Strips trailing role markers (e.g. 'Oracle:', 'Assistant:')
        that are used for completion continuation but conflict with chat templates.
        """
        # Strip trailing role marker (e.g., "Oracle:", "Assistant:", "Coder:")
        # These are continuation prompts for /v1/completions but are redundant
        # and sometimes cause empty responses with chat templates
        cleaned = re.sub(r'\n+[A-Z][a-zA-Z]*:\s*$', '', prompt).rstrip()

        # Try to split into system prompt and user message
        # Look for the last "User:" line as the actual user message
        last_user_match = None
        for m in re.finditer(r'\nUser:\s*', cleaned):
            last_user_match = m

        if last_user_match:
            system_content = cleaned[:last_user_match.start()].strip()
            user_content = cleaned[last_user_match.end():].strip()

            if system_content and user_content:
                return [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]

        # Fallback: send as single user message
        return [{"role": "user", "content": cleaned}]

    @retry(
        max_attempts=3,
        backoff_strategy="exponential",
        base_delay=2.0,
        max_delay=30.0,
        exceptions=(requests.exceptions.RequestException, requests.exceptions.Timeout, ConnectionError)
    )
    def _make_request(self, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        """Make HTTP request to llama-server with retry logic

        Uses /v1/chat/completions when use_chat_api is enabled (applies model's
        chat template properly for chat-finetuned models).
        Otherwise uses /v1/completions (raw completion mode).

        Args:
            payload: Request payload
            timeout: Request timeout in seconds (default: 120s)
        """
        try:
            if self.use_chat_api:
                # Convert prompt to chat messages format
                prompt = payload.pop('prompt')
                messages = self._prompt_to_messages(prompt)
                payload['messages'] = messages
                endpoint = f'http://localhost:{self.port}/v1/chat/completions'

                # When thinking is enabled on the server, ensure reasoning format
                # is set for proper separation of thinking from content
                if self.enable_thinking:
                    payload.setdefault('reasoning_format', 'deepseek')
            else:
                endpoint = f'http://localhost:{self.port}/v1/completions'

            if self.use_chat_api and self.verbose:
                self.logger.debug(f"Chat API messages: {len(payload.get('messages', []))} messages, "
                                  f"roles: {[m['role'] for m in payload.get('messages', [])]}")

            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            # Normalize chat response to match completion format
            # so callers can always use result['choices'][0]['text']
            if self.use_chat_api and result.get('choices'):
                for choice in result['choices']:
                    if 'message' in choice and 'text' not in choice:
                        content = choice['message'].get('content', '') or ''
                        choice['text'] = content
                    # Preserve reasoning_content if present (--reasoning-format deepseek)
                    if 'message' in choice and 'reasoning_content' not in choice:
                        reasoning = choice['message'].get('reasoning_content')
                        if reasoning:
                            choice['reasoning_content'] = reasoning

            return result

        except requests.exceptions.Timeout as e:
            self.logger.error(f"Server request timed out after {timeout}s")
            raise TimeoutError(f"LLM generation timed out after {timeout}s") from e

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Server request failed: {e}")
            raise

    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Alternative method name for compatibility"""
        return self(prompt, max_tokens, temperature, top_p, stop, **kwargs)

    def cleanup(self):
        """Stop server and cleanup resources"""
        if self.process:
            if self.verbose:
                self.logger.info(f"Stopping llama-server on port {self.port}")

            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

            self.process = None

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()

    @property
    def metadata(self):
        """Get model metadata - compatible with llama-cpp-python"""
        return {
            'model_path': self.model_path,
            'backend': 'llama-server',
            'port': self.port
        }
