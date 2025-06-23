#!/usr/bin/env python3
"""
Triage Agent - Intelligent Request Classification with llama.cpp Integration

This agent analyzes incoming requests using llama.cpp models loaded on-demand.
Models are loaded only when needed and unloaded after processing to conserve resources.

Key Features:
- On-demand model loading/unloading
- Support for larger models through dynamic resource management
- Maintains original classification logic with llama.cpp backend
- Comprehensive error handling and resource cleanup
"""

import logging
import re
import time
import os
import gc
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError as e:
    print("Error: llama-cpp-python not installed.")
    print("Install with: pip install llama-cpp-python")
    raise e

class TaskDifficulty(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    COLLABORATIVE = "collaborative"

class TaskCategory(Enum):
    SYSADMIN = "sysadmin"
    FILEOPS = "fileops"
    NETWORK = "network"
    DEVELOPMENT = "development"
    CONTENT = "content"
    SECURITY = "security"
    UNKNOWN = "unknown"

@dataclass
class TriageAnalysis:
    """Results of triage analysis"""
    category: TaskCategory
    difficulty: TaskDifficulty
    confidence: float
    uncertainty_factors: List[str]
    recommended_action: str
    reasoning: str
    needs_clarification: bool
    processing_time: float
    timestamp: datetime
    model_info: Dict[str, Any] = None

class ModelManager:
    """
    Manages llama.cpp model lifecycle for on-demand loading/unloading
    
    This component handles:
    - Model loading with specified parameters
    - Resource cleanup and memory management
    - Error handling for model operations
    - Performance monitoring
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
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger = logging.getLogger("ModelManager")
    
    def load_model(self) -> bool:
        """Load the llama.cpp model into memory"""
        if self.llm is not None:
            self.logger.debug("Model already loaded")
            return True
        
        try:
            start_time = time.time()
            self.logger.info(f"Loading model: {self.model_path.name}")
            
            self.llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=self.verbose
            )
            
            self.load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {self.load_time:.1f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.llm = None
            return False
    
    def unload_model(self):
        """Unload model and free memory"""
        if self.llm is not None:
            self.logger.info("Unloading model")
            del self.llm
            self.llm = None
            
            # Force garbage collection to free memory
            gc.collect()
            
            self.logger.debug("Model unloaded and memory freed")
    
    def is_loaded(self) -> bool:
        """Check if model is currently loaded"""
        return self.llm is not None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the loaded model"""
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Set default generation parameters
            generation_params = {
                'temperature': kwargs.get('temperature', 0.2),
                'max_tokens': kwargs.get('max_tokens', 300),
                'top_p': kwargs.get('top_p', 0.9),
                'stop': kwargs.get('stop_sequence', ["\n\nRequest:", "Human:", "User:", "\n\n---"]),
                'stream': False  # We want complete response for parsing
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
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_path': str(self.model_path),
            'model_name': self.model_path.name,
            'n_gpu_layers': self.n_gpu_layers,
            'n_ctx': self.n_ctx,
            'load_time': self.load_time,
            'is_loaded': self.is_loaded()
        }

class TriageAgent:
    """
    Intelligent triaging agent with on-demand model loading
    
    This agent uses llama.cpp models to classify requests and assess confidence.
    Models are loaded only when needed and unloaded after processing to optimize
    resource usage and enable larger model support.
    """
    
    def __init__(self, 
                 model_path: str,
                 n_gpu_layers: int = 40,
                 n_ctx: int = 2048,
                 model_verbose: bool = False):
        
        # Set up logging
        self.logger = logging.getLogger("TriageAgent")
        
        # Initialize model manager
        self.model_manager = ModelManager(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=model_verbose
        )
        
        # LLM parameters optimized for classification
        self.llm_params = {
            'temperature': 0.2,  # Lower for consistency
            'max_tokens': 300,
            'top_p': 0.9,
            'stop_sequence': ["\n\nRequest:", "Human:", "User:", "\n\n---"]
        }
        
        # Create the focused prompt for triage analysis
        self.triage_prompt = self._create_triage_prompt()
        
        # Statistics tracking
        self.stats = {
            'total_analyses': 0,
            'high_confidence_count': 0,
            'medium_confidence_count': 0,
            'low_confidence_count': 0,
            'clarification_requests': 0,
            'avg_processing_time': 0.0,
            'model_load_count': 0,
            'total_model_load_time': 0.0
        }
        
        self.logger.info(f"Triage Agent initialized with model: {self.model_manager.model_path.name}")
    
    def _create_triage_prompt(self) -> str:
        """Create optimized prompt for task classification and confidence assessment"""
        return """You are a Task Classifier. Analyze each request carefully and be honest about uncertainty.

TASK CATEGORIES:
- sysadmin: System commands (IP check, disk space, memory, install packages, system info)
- fileops: File operations (list files, backup, find files, file management)
- network: Network tests (ping, download, connectivity checks, curl requests)
- development: Code tasks (git operations, python scripts, building, debugging)
- content: Writing, documentation, text generation
- security: Security-related commands, system access, dangerous operations
- unknown: Cannot classify clearly

DIFFICULTY LEVELS:
- simple: One clear command, straightforward task
- moderate: Multiple steps, some complexity
- complex: Needs problem-solving, multiple tools
- collaborative: Needs expert guidance or human oversight

CONFIDENCE ASSESSMENT (BE STRICT AND HONEST):
- 0.9-1.0: Crystal clear, specific, unambiguous request
- 0.7-0.9: Clear with minor ambiguity or assumptions needed
- 0.5-0.7: Reasonable interpretation, some uncertainty
- 0.3-0.5: Multiple possible interpretations, significant uncertainty
- 0.0-0.3: Very unclear, vague, or potentially dangerous

UNCERTAINTY ANALYSIS:
Think about what makes you uncertain:
- Vague language or missing details
- Multiple possible interpretations
- Unclear scope or requirements
- Potentially dangerous implications
- Technical ambiguity

ACTIONS:
- route_to_agent: High confidence, clear classification
- request_clarification: Low confidence or ambiguous
- escalate_security: Potentially dangerous or security-sensitive

FORMAT (STRICT):
CATEGORY: [category]
DIFFICULTY: [difficulty]
CONFIDENCE: [0.0-1.0]
UNCERTAINTY: [list specific uncertainty factors]
ACTION: [route_to_agent|request_clarification|escalate_security]
REASONING: [explain your analysis briefly]

Request: """
    
    def analyze_request(self, request: str) -> TriageAnalysis:
        """
        Analyze a request using on-demand model loading
        
        This method:
        1. Loads the model into memory
        2. Performs triage analysis
        3. Unloads the model to free resources
        4. Returns comprehensive analysis results
        
        Args:
            request: The user request to analyze
            
        Returns:
            TriageAnalysis object with complete classification results
        """
        start_time = time.time()
        model_info = None
        
        try:
            # Handle empty requests
            if not request.strip():
                return self._create_empty_analysis(start_time)
            
            # Load model
            self.logger.info("Loading model for triage analysis...")
            model_load_start = time.time()
            
            if not self.model_manager.load_model():
                raise RuntimeError("Failed to load model")
            
            model_load_time = time.time() - model_load_start
            self.stats['model_load_count'] += 1
            self.stats['total_model_load_time'] += model_load_time
            
            # Get model info for analysis record
            model_info = self.model_manager.get_model_info()
            
            # Perform analysis
            self.logger.info(f"Analyzing request: {request[:50]}...")
            llm_response = self._query_llm(request)
            
            # Parse the LLM response
            analysis = self._parse_llm_response(llm_response, request, start_time)
            analysis.model_info = model_info
            
            # Update statistics
            self._update_statistics(analysis)
            
            # Log the analysis
            self.logger.info(
                f"Triage completed: {analysis.category.value} "
                f"(conf: {analysis.confidence:.2f}, "
                f"time: {analysis.processing_time:.1f}s)"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error during triage analysis: {e}")
            return self._create_error_analysis(request, str(e), start_time, model_info)
        
        finally:
            # Always unload model to free resources
            self.model_manager.unload_model()
            self.logger.debug("Model unloaded after analysis")
    
    def _query_llm(self, request: str) -> str:
        """Send request to LLM and get response"""
        full_prompt = self.triage_prompt + request
        
        try:
            response = self.model_manager.generate(full_prompt, **self.llm_params)
            
            if len(response) < 10:
                raise ValueError("LLM returned very short response")
            
            return response
            
        except Exception as e:
            raise Exception(f"LLM processing error: {e}")
    
    def _parse_llm_response(self, llm_response: str, original_request: str, start_time: float) -> TriageAnalysis:
        """Parse LLM response into structured TriageAnalysis"""
        try:
            # Extract fields using regex patterns
            patterns = {
                'category': r'CATEGORY:\s*(\w+)',
                'difficulty': r'DIFFICULTY:\s*(\w+)',
                'confidence': r'CONFIDENCE:\s*([\d.]+)',
                'uncertainty': r'UNCERTAINTY:\s*(.+?)(?=\nACTION:|$)',
                'action': r'ACTION:\s*(\w+)',
                'reasoning': r'REASONING:\s*(.+)'
            }
            
            extracted = {}
            for field, pattern in patterns.items():
                match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)
                extracted[field] = match.group(1).strip() if match else None
            
            # Parse and validate category
            category_str = extracted['category']
            if category_str:
                category_map = {
                    'sysadmin': TaskCategory.SYSADMIN,
                    'fileops': TaskCategory.FILEOPS,
                    'network': TaskCategory.NETWORK,
                    'development': TaskCategory.DEVELOPMENT,
                    'content': TaskCategory.CONTENT,
                    'security': TaskCategory.SECURITY,
                    'unknown': TaskCategory.UNKNOWN
                }
                category = category_map.get(category_str.lower(), TaskCategory.UNKNOWN)
            else:
                category = TaskCategory.UNKNOWN
            
            # Parse and validate difficulty
            difficulty_str = extracted['difficulty']
            if difficulty_str:
                difficulty_map = {
                    'simple': TaskDifficulty.SIMPLE,
                    'moderate': TaskDifficulty.MODERATE,
                    'complex': TaskDifficulty.COMPLEX,
                    'collaborative': TaskDifficulty.COLLABORATIVE
                }
                difficulty = difficulty_map.get(difficulty_str.lower(), TaskDifficulty.MODERATE)
            else:
                difficulty = TaskDifficulty.MODERATE
            
            # Parse and validate confidence
            confidence_str = extracted['confidence']
            if confidence_str:
                try:
                    confidence = float(confidence_str)
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except ValueError:
                    confidence = 0.3  # Default for parse errors
            else:
                confidence = 0.3
            
            # Parse uncertainty factors
            uncertainty_text = extracted['uncertainty'] or "General uncertainty"
            uncertainty_factors = [
                factor.strip() 
                for factor in uncertainty_text.replace('\n', ',').split(',') 
                if factor.strip()
            ]
            if not uncertainty_factors:
                uncertainty_factors = [uncertainty_text]
            
            # Parse action
            action = extracted['action']
            if action:
                action = action.lower()
                if action not in ['route_to_agent', 'request_clarification', 'escalate_security']:
                    action = 'request_clarification'  # Default safe action
            else:
                action = 'request_clarification'
            
            # Parse reasoning
            reasoning = extracted['reasoning'] or "No reasoning provided"
            
            # Determine if clarification is needed
            needs_clarification = (
                action == 'request_clarification' or 
                confidence < 0.5 or
                category == TaskCategory.UNKNOWN
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return TriageAnalysis(
                category=category,
                difficulty=difficulty,
                confidence=confidence,
                uncertainty_factors=uncertainty_factors,
                recommended_action=action,
                reasoning=reasoning,
                needs_clarification=needs_clarification,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return self._create_error_analysis(original_request, f"Parse error: {e}", start_time)
    
    def _create_empty_analysis(self, start_time: float) -> TriageAnalysis:
        """Create analysis for empty requests"""
        return TriageAnalysis(
            category=TaskCategory.UNKNOWN,
            difficulty=TaskDifficulty.SIMPLE,
            confidence=0.0,
            uncertainty_factors=["Empty request"],
            recommended_action="request_clarification",
            reasoning="No content to analyze",
            needs_clarification=True,
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    def _create_error_analysis(self, request: str, error: str, start_time: float, model_info: Dict = None) -> TriageAnalysis:
        """Create analysis for error cases"""
        analysis = TriageAnalysis(
            category=TaskCategory.UNKNOWN,
            difficulty=TaskDifficulty.COMPLEX,
            confidence=0.0,
            uncertainty_factors=[f"Processing error: {error}"],
            recommended_action="escalate_security",  # Escalate errors for safety
            reasoning=f"Could not analyze request due to error: {error}",
            needs_clarification=True,
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )
        if model_info:
            analysis.model_info = model_info
        return analysis
    
    def _update_statistics(self, analysis: TriageAnalysis):
        """Update internal statistics"""
        self.stats['total_analyses'] += 1
        
        # Update confidence distribution
        if analysis.confidence >= 0.7:
            self.stats['high_confidence_count'] += 1
        elif analysis.confidence >= 0.4:
            self.stats['medium_confidence_count'] += 1
        else:
            self.stats['low_confidence_count'] += 1
        
        if analysis.needs_clarification:
            self.stats['clarification_requests'] += 1
        
        # Update average processing time
        total_time = self.stats['avg_processing_time'] * (self.stats['total_analyses'] - 1)
        self.stats['avg_processing_time'] = (total_time + analysis.processing_time) / self.stats['total_analyses']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current triage statistics including model performance"""
        total = self.stats['total_analyses']
        if total == 0:
            return self.stats
        
        enhanced_stats = {
            **self.stats,
            'high_confidence_rate': self.stats['high_confidence_count'] / total,
            'medium_confidence_rate': self.stats['medium_confidence_count'] / total,
            'low_confidence_rate': self.stats['low_confidence_count'] / total,
            'clarification_rate': self.stats['clarification_requests'] / total,
            'avg_model_load_time': (
                self.stats['total_model_load_time'] / self.stats['model_load_count'] 
                if self.stats['model_load_count'] > 0 else 0.0
            ),
            'model_info': self.model_manager.get_model_info()
        }
        
        return enhanced_stats
    
    def test_model_loading(self) -> bool:
        """Test model loading capability"""
        try:
            self.logger.info("Testing model loading...")
            
            if not self.model_manager.load_model():
                return False
            
            # Try a simple generation
            test_response = self.model_manager.generate("Test: ", max_tokens=5)
            
            # Unload model
            self.model_manager.unload_model()
            
            self.logger.info("✅ Model loading test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model loading test failed: {e}")
            # Ensure cleanup
            self.model_manager.unload_model()
            return False
    
    def cleanup(self):
        """Cleanup resources (call when agent is no longer needed)"""
        self.model_manager.unload_model()
        self.logger.info("Triage agent cleanup completed")


def main():
    """Standalone testing interface for the llama.cpp triage agent"""
    print("Triage Agent - llama.cpp Integration Test")
    print("=" * 45)
    
    # Configuration
    model_path = input("Enter path to your GGUF model file: ").strip()
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
        agent = TriageAgent(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            model_verbose=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Test model loading
    if not agent.test_model_loading():
        print("❌ Model loading test failed. Check your model file and parameters.")
        return
    
    # Interactive testing
    test_requests = [
        "What is my public IP address?",
        "Check disk space with df -h",
        "Help me with my computer",
        "rm -rf / --no-preserve-root",
        "Install htop on Ubuntu",
        "Ping google.com 4 times"
    ]
    
    print("\nExample test requests:")
    for i, req in enumerate(test_requests, 1):
        print(f"{i}. {req}")
    
    print("\nEnter a request to analyze (or 'quit' to exit):")
    
    while True:
        user_input = input("> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input.isdigit() and 1 <= int(user_input) <= len(test_requests):
            user_input = test_requests[int(user_input) - 1]
        
        if not user_input:
            continue
        
        print(f"\nAnalyzing: {user_input}")
        print("-" * 40)
        
        # Perform analysis
        analysis = agent.analyze_request(user_input)
        
        # Display results
        print(f"Category: {analysis.category.value}")
        print(f"Difficulty: {analysis.difficulty.value}")
        print(f"Confidence: {analysis.confidence:.3f}")
        print(f"Action: {analysis.recommended_action}")
        print(f"Needs clarification: {analysis.needs_clarification}")
        print(f"Processing time: {analysis.processing_time:.2f}s")
        print(f"Uncertainty factors: {', '.join(analysis.uncertainty_factors)}")
        print(f"Reasoning: {analysis.reasoning}")
        
        if analysis.model_info:
            print(f"Model load time: {analysis.model_info.get('load_time', 'N/A'):.2f}s")
        print()
    
    # Show final statistics
    stats = agent.get_statistics()
    print("\nSession Statistics:")
    for key, value in stats.items():
        if key == 'model_info':
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Cleanup
    agent.cleanup()
    print("\n✅ Agent cleanup completed")


if __name__ == "__main__":
    main()