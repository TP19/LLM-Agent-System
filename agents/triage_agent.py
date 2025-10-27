#!/usr/bin/env python3
"""
Triage Agent - Request Classification

Classifies user requests and routes them to appropriate agents.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from core.base_agent import BaseAgent


class TaskCategory(Enum):
    """Task categories for routing"""
    SYSADMIN = "sysadmin"
    FILEOPS = "fileops" 
    NETWORK = "network"
    DEVELOPMENT = "development"
    CONTENT = "content"
    SECURITY = "security"
    CODING = "coding"
    SUMMARIZATION = "summarization"
    KNOWLEDGE_QUERY = "knowledge_query"
    UNKNOWN = "unknown"


class TaskDifficulty(Enum):
    """Task difficulty levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class TriageResult:
    """Triage classification result"""
    category: TaskCategory
    difficulty: TaskDifficulty
    confidence: float
    reasoning: str
    needs_clarification: bool
    processing_time: float
    timestamp: datetime


class ModularTriageAgent(BaseAgent):
    """
    Triage agent using modular architecture
    
    Classifies user requests into categories for routing.
    """
    
    def __init__(self, model_manager):
        super().__init__("triage", model_manager)
        
        # Create the classification prompt
        self.prompt = self._create_practical_prompt()
        
        self.logger.info("✅ Modular triage agent initialized")
    
    def analyze(self, request: str, request_id: str = None) -> TriageResult:
        """
        Analyze and classify a user request
        
        Args:
            request: User's request text
            request_id: Optional request tracking ID
            
        Returns:
            TriageResult with classification details
        """
        
        if not request_id:
            request_id = str(uuid.uuid4())[:8]
        
        start_time = time.time()
        self.logger.info(f"[{request_id}] 🔍 Analyzing: {request[:50]}...")
        
        try:
            # Generate analysis using base class
            full_prompt = self.prompt + request
            response = self.generate_with_logging(
                full_prompt,
                request_id,
                max_tokens=300,
                temperature=0.3,
                top_p=0.9
            )
            
            # Parse the response
            result = self._parse_response(response, request, start_time)
            
            # Update stats
            self.update_stats(result.processing_time)
            
            # Log result
            self.logger.info(
                f"[{request_id}] ✅ Classification: {result.category.value} "
                f"(confidence: {result.confidence:.2f})"
            )
            
            if result.needs_clarification:
                self.logger.warning(
                    f"[{request_id}] ❓ Needs clarification: {result.reasoning}"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{request_id}] ❌ Triage analysis failed: {e}")
            return self._create_error_result(request, str(e), start_time)
    
    def _create_practical_prompt(self) -> str:
        """Create classification prompt"""
        return """You are a Triage Agent that quickly classifies user requests to route them to the right agent.

BE PRACTICAL, NOT OVERLY CAUTIOUS. If a request is reasonably clear, classify it confidently.

CATEGORIES (in priority order):
- summarization: ANALYZING EXISTING FILES - summarize, analyze, extract insights, give overview of existing text/markdown files
- knowledge_query: SEARCHING KNOWLEDGE BASE - find information in documents, search for topics, query stored knowledge
- sysadmin: System tasks (check disk, memory, processes, SSH operations, server maintenance)
- fileops: File operations (find files, backup, directory operations) 
- network: Network tasks (ping, connectivity, downloads, API calls)
- development: Code tasks (git, scripts, building, debugging, CREATE MODULAR SYSTEMS, BUILD FROM SPECIFICATIONS)
- content: CREATING NEW CONTENT - Writing NEW documents, creating NEW documentation, generating NEW text
- security: High-risk operations (user management, permissions, dangerous commands)
- coding: Building projects from specifications
- unknown: Truly unclear requests only

CRITICAL DISTINCTIONS:
- "Summarize FILE.txt" = summarization (analyzing EXISTING file)
- "Find information about X" = knowledge_query (searching knowledge base)
- "What did Feynman say about Y?" = knowledge_query (query stored knowledge)
- "Search for Z in documents" = knowledge_query (search)
- "Write a summary of X" = content (creating NEW document)
- "Analyze FILE.md" = summarization (reading EXISTING file)
- "Create analysis document" = content (making NEW file)

KNOWLEDGE_QUERY requests include:
- "Find information about..."
- "Search for..."
- "What does X say about Y?"
- "Show me information on..."
- "Query about..."
- Any request to search or find in knowledge base

SUMMARIZATION requests ALWAYS include:
- "Summarize [file path]"
- "Analyze [file path]"
- "Give me overview of [file path]"
- "Extract key points from [file path]"
- "Read and summarize [file path]"
- ANY request with "summarize/analyze" AND a file path

CONTENT requests are about CREATING NEW TEXT:
- "Write a blog post about X"
- "Create documentation for Y"
- "Generate a report on Z"
- "Draft an email about..."

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
    
    def _parse_response(self, response: str, request: str, start_time: float) -> TriageResult:
        """Parse LLM response into structured result"""
        import re
        
        try:
            # Extract fields using regex
            category_match = re.search(r'CATEGORY:\s*(\w+)', response, re.IGNORECASE)
            difficulty_match = re.search(r'DIFFICULTY:\s*(\w+)', response, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.IGNORECASE)
            reasoning_match = re.search(
                r'REASONING:\s*(.+?)(?=\nCLARIFICATION:|$)', 
                response, 
                re.IGNORECASE | re.DOTALL
            )
            clarification_match = re.search(r'CLARIFICATION:\s*(\w+)', response, re.IGNORECASE)
            
            # Parse category
            category_str = category_match.group(1).lower() if category_match else 'unknown'
            try:
                category = TaskCategory(category_str)
            except ValueError:
                self.logger.warning(f"Unknown category: {category_str}")
                category = TaskCategory.UNKNOWN
            
            # Parse difficulty
            difficulty_str = difficulty_match.group(1).lower() if difficulty_match else 'moderate'
            try:
                difficulty = TaskDifficulty(difficulty_str)
            except ValueError:
                difficulty = TaskDifficulty.MODERATE
            
            # Parse confidence
            confidence = 0.5
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
            
            # Parse reasoning
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "Analysis completed"
            
            # Parse clarification need
            needs_clarification = False
            if clarification_match:
                clarification_str = clarification_match.group(1).lower()
                needs_clarification = clarification_str in ['yes', 'true', '1']
            
            # Override clarification logic - be less strict
            if confidence >= 0.5 and category != TaskCategory.UNKNOWN:
                needs_clarification = False
            
            processing_time = time.time() - start_time
            
            return TriageResult(
                category=category,
                difficulty=difficulty,
                confidence=confidence,
                reasoning=reasoning,
                needs_clarification=needs_clarification,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}")
            return self._create_error_result(request, f"Parse error: {e}", start_time)
    
    def _create_error_result(self, request: str, error: str, start_time: float) -> TriageResult:
        """Create error result when analysis fails"""
        return TriageResult(
            category=TaskCategory.UNKNOWN,
            difficulty=TaskDifficulty.MODERATE,
            confidence=0.0,
            reasoning=f"Error in analysis: {error}",
            needs_clarification=True,
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    def cleanup(self):
        """Cleanup resources"""
        # Model cleanup is handled by the model manager
        pass


# Export all classes for easy importing
__all__ = [
    'ModularTriageAgent',
    'TaskCategory',
    'TaskDifficulty',
    'TriageResult'
]