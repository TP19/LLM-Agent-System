#!/usr/bin/env python3
"""
Base Agent with enhanced logging and lazy model loading
"""

import logging
import time
import uuid
import re
import json
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable

class BaseAgent:
    """Base agent with enhanced logging and lazy model loading"""

    def __init__(self, agent_name: str, model_manager, interactive: bool = False,
                 checkpoint_callback: Optional[Callable] = None):
        """
        Initialize base agent

        Args:
            agent_name: Name of the agent
            model_manager: Model manager instance
            interactive: Enable interactive mode with checkpoints
            checkpoint_callback: Callback function for checkpoints (data, risk_level) -> bool
        """
        self.agent_name = agent_name
        self.model_manager = model_manager
        self.logger = logging.getLogger(f"{agent_name}Agent")

        # Interactive mode support
        self.interactive = interactive
        self.checkpoint_callback = checkpoint_callback

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
    
    def generate_with_logging(self, prompt: str, request_id: str, timeout: int = 120, **kwargs) -> str:
        """Generate response with detailed logging and timeout

        Args:
            prompt: The prompt to send to the model
            request_id: Request ID for tracking
            timeout: Maximum seconds to wait for response (default: 120)
            **kwargs: Additional parameters for model generation
        """

        # Log the prompt (first 200 chars)
        self.logger.debug(f"[{request_id}] 📝 Prompt preview: {prompt[:200]}...")
        self.logger.debug(f"[{request_id}] ⚙️ Generation params: {kwargs}")

        start_time = time.time()

        # Get model (triggers lazy loading)
        model = self.model_manager.get_model_for_agent(self.agent_name)
        self.stats['model_load_count'] += 1

        # Log that we're starting generation
        self.logger.info(f"[{request_id}] ⏳ Starting LLM generation (timeout: {timeout}s)...")

        # Generate response with timeout handling
        try:
            # Temporarily disable timeout to debug
            # def timeout_handler(signum, frame):
            #     raise TimeoutError(f"LLM generation exceeded {timeout}s timeout")

            # Try with timeout (llama-server), fallback without (llama-cpp-python)
            try:
                response = model(prompt, timeout=timeout, **kwargs)
            except TypeError as e:
                if 'timeout' in str(e):
                    # llama-cpp-python doesn't support timeout, retry without it
                    self.logger.debug(f"[{request_id}] Backend doesn't support timeout, retrying without")
                    response = model(prompt, **kwargs)
                else:
                    raise

            generation_time = time.time() - start_time

            # Log response details
            choice0 = response['choices'][0]
            response_text = choice0.get('text', '') or ''
            reasoning_text = choice0.get('reasoning_content', '')

            self.logger.debug(f"[{request_id}] 🔍 Choice keys: {list(choice0.keys())}, "
                            f"finish_reason: {choice0.get('finish_reason', 'N/A')}")
            if reasoning_text:
                self.logger.debug(f"[{request_id}] 🧠 Reasoning: {len(reasoning_text)} chars")

            # Handle thinking/reasoning content
            agent_config = self.model_manager.get_agent_config(self.agent_name)
            if agent_config.get('enable_thinking'):
                # With --reasoning-format deepseek, thinking is in reasoning_content
                # and content has the actual answer. Strip any residual think tags.
                response_text = self._strip_think_tags(response_text)

                # If content is empty but we have reasoning, use reasoning as fallback
                if not response_text.strip() and reasoning_text:
                    self.logger.warning(f"[{request_id}] Content empty, using reasoning_content as fallback")
                    response_text = self._strip_think_tags(reasoning_text)

            self.logger.debug(f"[{request_id}] ⚡ Generated in {generation_time:.2f}s")
            self.logger.debug(f"[{request_id}] 📄 Response length: {len(response_text)} chars")
            self.logger.debug(f"[{request_id}] 🔤 Response preview: {response_text[:300]}...")

            # Log full response to separate file for debugging
            self._log_full_response(request_id, prompt, response_text, generation_time)

            return response_text

        except TimeoutError as e:
            self.logger.error(f"[{request_id}] ⏰ {e}")
            raise
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Generation failed: {e}")
            raise
    
    def _strip_think_tags(self, text: str) -> str:
        """Strip <think>...</think> tags from model output, keeping content outside tags.

        For models with enable_thinking, the model may wrap reasoning in <think> tags.
        We extract only the content after the closing </think> tag.
        """
        # Remove complete <think>...</think> blocks (greedy to handle nested)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Handle unclosed <think> at start (model started thinking, never closed)
        if cleaned.strip().startswith('<think>'):
            # No closing tag found - take content after the tag
            cleaned = re.sub(r'^<think>', '', cleaned.strip(), count=1)

        return cleaned.strip()

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

    # ========================================================================
    # Chat Interface
    # ========================================================================

    def chat(self, message: str, history: list = None) -> str:
        """
        Chat interface for interactive conversation with the agent.

        This default implementation uses the agent's LLM to generate responses.
        Agents can override this method for specialized behavior.

        Args:
            message: User's message
            history: Optional conversation history (list of dicts with 'role' and 'content')

        Returns:
            Agent's response string
        """
        request_id = str(uuid.uuid4())[:8]

        # Build context from history if provided
        context = ""
        if history:
            for msg in history[-5:]:  # Last 5 messages for context
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                context += f"{role.capitalize()}: {content}\n"

        # Build prompt
        system_prompt = self._get_chat_system_prompt()
        user_prompt = f"{context}User: {message}\n{self.agent_name.title()}:"

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response = self.generate_with_logging(
                prompt=full_prompt,
                request_id=request_id,
                max_tokens=1024,
                temperature=0.7
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"[{request_id}] Chat error: {e}")
            return f"I encountered an error: {str(e)}"

    def _get_chat_system_prompt(self) -> str:
        """
        Get the system prompt for chat mode.

        Override this in subclasses for agent-specific behavior.
        """
        return f"""You are {self.agent_name.title()}, an AI assistant in the LLM-Agent-System system.

Your role varies by agent type:
- Oracle: Strategic planning and task coordination
- Security: Security analysis and threat assessment
- Operator: Command execution and system operations
- Navigator: Resource discovery and knowledge retrieval
- Coder: Code generation and analysis
- Validator: Result validation and quality assurance
- Summarizer: Document summarization and key insights extraction

Be helpful, concise, and focus on your specialty area.
Respond directly to the user's question or request."""

    # ========================================================================
    # Interactive Mode Support
    # ========================================================================

    def checkpoint(self, data: Dict[str, Any], risk_level: str = "medium",
                   auto_approve: bool = False) -> bool:
        """
        Create checkpoint for interactive review

        Args:
            data: Checkpoint data to present to user
            risk_level: Risk level ("low", "medium", "high")
            auto_approve: Auto-approve this checkpoint

        Returns:
            True if approved, False if rejected
        """
        if not self.interactive or auto_approve:
            return True  # Auto-approve in non-interactive mode or when specified

        if self.checkpoint_callback:
            try:
                return self.checkpoint_callback(data, risk_level)
            except Exception as e:
                self.logger.error(f"Checkpoint callback failed: {e}")
                return False

        # No callback provided, default to approve
        return True

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def build_prompt(self, template: str, **kwargs) -> str:
        """
        Build prompt from template with variables

        Args:
            template: Prompt template string
            **kwargs: Variables to substitute

        Returns:
            Formatted prompt
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            return template

    def parse_json_response(self, response: str, default: Optional[Dict] = None) -> Dict:
        """
        Parse JSON from LLM response with fallback

        Args:
            response: LLM response text
            default: Default value if parsing fails

        Returns:
            Parsed dictionary or default
        """
        try:
            # Try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Try finding any JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            # Fallback to default
            if default is not None:
                return default
            else:
                return {"raw": response}

    def retry_generate(self, prompt: str, max_attempts: int = 3,
                       request_id: Optional[str] = None, **kwargs) -> str:
        """
        Generate with automatic retry on failure

        Args:
            prompt: Prompt text
            max_attempts: Maximum retry attempts
            request_id: Request ID for logging
            **kwargs: Additional generation parameters

        Returns:
            Generated response

        Raises:
            Exception: If all attempts fail
        """
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]

        for attempt in range(max_attempts):
            try:
                return self.generate_with_logging(prompt, request_id, **kwargs)
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def truncate_to_context(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit context window

        Args:
            text: Text to truncate
            max_tokens: Maximum token count

        Returns:
            Truncated text
        """
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        return truncated + "\n\n[...truncated to fit context window...]"