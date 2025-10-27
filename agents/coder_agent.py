#!/usr/bin/env python3
"""
Modular Coder Agent - Lazy Loading Version

This agent can:
- Parse modular design specifications 
- Create directory structures
- Generate and write code files
- Understand project organization patterns
- Work within the existing agent collaboration framework

Focus: Transform specifications into working code automatically.
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Import base agent for modular architecture
from core.base_agent import BaseAgent

class CodeFileType(Enum):
    PYTHON_MODULE = "python_module"
    PYTHON_CLASS = "python_class"
    PYTHON_SCRIPT = "python_script"
    CONFIG_FILE = "config_file"
    INIT_FILE = "init_file"
    TEST_FILE = "test_file"

@dataclass
class CodeFile:
    """Represents a code file to be generated"""
    path: str
    file_type: CodeFileType
    content: str
    dependencies: List[str]
    description: str

@dataclass
class ProjectStructure:
    """Represents the project structure to be created"""
    base_path: str
    directories: List[str]
    files: List[CodeFile]
    creation_order: List[str]

@dataclass
class CoderResult:
    """Result of coder agent operation"""
    success: bool
    files_created: List[str]
    directories_created: List[str]
    errors: List[str]
    warnings: List[str]
    processing_time: float
    timestamp: datetime

class ModularCoderAgent(BaseAgent):
    """Modular AI-powered coder agent that can generate and organize code"""

    def __init__(self, model_manager):
        super().__init__("coder", model_manager)

        # Specialized prompts for different code generation tasks
        self.prompts = {
            'parse_specification': self._create_parsing_prompt(),
            'generate_structure': self._create_structure_prompt(),
            'generate_code': self._create_code_prompt(),
            'generate_init': self._create_init_prompt()
        }

        self.stats = {
            'projects_processed': 0,
            'files_generated': 0,
            'directories_created': 0,
            'total_lines_written': 0,
            'successful_generations': 0
        }

    def process_specification(self, spec_content: str, base_path: str = ".") -> CoderResult:
        """Process a modular design specification and create the project"""

        self.logger.info("🔧 Processing modular specification...")
        start_time = time.time()

        try:
            # 1. Parse the specification to understand structure
            self.logger.info("📋 Step 1: Parsing specification")
            project_structure = self._parse_specification(spec_content, base_path)

            # 2. Create directory structure
            self.logger.info("📁 Step 2: Creating directory structure")
            dirs_created = self._create_directories(project_structure)

            # 3. Generate and write code files
            self.logger.info("⚡ Step 3: Generating code files")
            files_created, errors, warnings = self._generate_code_files(project_structure)

            # 4. Create result summary
            processing_time = time.time() - start_time
            result = CoderResult(
                success=len(errors) == 0,
                files_created=files_created,
                directories_created=dirs_created,
                errors=errors,
                warnings=warnings,
                processing_time=processing_time,
                timestamp=datetime.now()
            )

            # Update stats
            self._update_stats(result)

            self.logger.info(f"✅ Specification processed in {processing_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Specification processing failed: {e}")
            return CoderResult(
                success=False,
                files_created=[],
                directories_created=[],
                errors=[str(e)],
                warnings=[],
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )

    def _parse_specification(self, spec_content: str, base_path: str) -> ProjectStructure:
        """Parse specification to extract project structure"""

        self.logger.info("🔍 Analyzing specification content...")

        # Use LLM to parse the specification intelligently
        prompt = self.prompts['parse_specification'] + spec_content

        response = self.generate_with_logging(
            prompt,
            max_tokens=1500,
            temperature=0.3,
            top_p=0.9
        )

        # Parse LLM response to extract structure
        structure_data = self._extract_structure_from_response(response)

        # Convert to ProjectStructure object
        project_structure = ProjectStructure(
            base_path=base_path,
            directories=structure_data['directories'],
            files=[],  # Will be populated during code generation
            creation_order=structure_data['creation_order']
        )

        self.logger.info(f"📊 Found {len(project_structure.directories)} directories to create")
        return project_structure

    def _create_directories(self, project_structure: ProjectStructure) -> List[str]:
        """Create the directory structure"""

        created_dirs = []
        base_path = Path(project_structure.base_path)

        for directory in project_structure.directories:
            dir_path = base_path / directory

            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
                self.logger.info(f"📁 Created: {dir_path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to create {dir_path}: {e}")

        return created_dirs

    def _generate_code_files(self, project_structure: ProjectStructure) -> Tuple[List[str], List[str], List[str]]:
        """Generate and write all code files"""

        files_created = []
        errors = []
        warnings = []

        # Generate files in the specified order
        for file_spec in project_structure.creation_order:
            try:
                # Determine file type and generate appropriate content
                code_file = self._generate_single_file(file_spec, project_structure)

                if code_file:
                    # Write the file
                    file_path = Path(project_structure.base_path) / code_file.path
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(code_file.content)

                    files_created.append(str(file_path))
                    self.logger.info(f"✅ Generated: {file_path}")

                    # Count lines for stats
                    lines = len(code_file.content.split('\n'))
                    self.stats['total_lines_written'] += lines

                else:
                    warnings.append(f"Skipped file generation for: {file_spec}")

            except Exception as e:
                error_msg = f"Failed to generate {file_spec}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(f"❌ {error_msg}")

        return files_created, errors, warnings

    def _generate_single_file(self, file_spec: str, project_structure: ProjectStructure) -> Optional[CodeFile]:
        """Generate a single code file based on specification"""

        # Determine file type based on path and context
        file_type = self._determine_file_type(file_spec)

        if file_type == CodeFileType.INIT_FILE:
            return self._generate_init_file(file_spec)
        else:
            # Use unified generation method for all Python files
            return self._generate_any_python_file(file_spec, file_type)

    def _generate_any_python_file(self, file_spec: str, file_type: CodeFileType) -> CodeFile:
        """Generate any Python file with unified approach"""

        # Extract module name and purpose from file_spec
        module_name = Path(file_spec).stem
        module_purpose = self._extract_module_purpose(file_spec)

        # Create context-aware prompt
        prompt = f"""{self._create_code_prompt()}

Generate a Python file for: {file_spec}
Module name: {module_name}
Purpose: {module_purpose}
File type: {file_type.value}

Requirements:
- Include proper docstring
- Add necessary imports
- Create appropriate classes/functions based on the module name
- Follow clean code principles
- Add error handling where appropriate
- Use type hints
- NO EXPLANATIONS - ONLY PYTHON CODE

CRITICAL: Return ONLY the Python code, no markdown, no explanations, no extra text.

File: {file_spec}
"""

        try:
            response = self.generate_with_logging(
                prompt,
                max_tokens=2500,  # Increased for complex files
                temperature=0.3,  # Lower temperature for more consistent output
                top_p=0.8
            )

            raw_content = response
            content = self._aggressively_clean_code(raw_content)

            # Fallback if cleaning failed
            if not content or len(content) < 50:
                self.logger.warning(f"⚠️ Generated content too short for {file_spec}, using template")
                content = self._create_template_file(file_spec, module_purpose)

            return CodeFile(
                path=file_spec,
                file_type=file_type,
                content=content,
                dependencies=[],
                description=f"Python file: {module_name}"
            )

        except Exception as e:
            self.logger.error(f"❌ Error generating {file_spec}: {e}")
            # Return template as fallback
            content = self._create_template_file(file_spec, module_purpose)
            return CodeFile(
                path=file_spec,
                file_type=file_type,
                content=content,
                dependencies=[],
                description=f"Template file: {module_name}"
            )

    def _aggressively_clean_code(self, raw_code: str) -> str:
        """Aggressively clean LLM output to extract only Python code"""

        # Step 1: Remove common LLM response patterns
        cleaned = raw_code

        # Remove explanatory text before code
        patterns_to_remove = [
            r"Here's.*?:\s*\n",
            r"This.*?:\s*\n", 
            r"I'll.*?:\s*\n",
            r"The following.*?:\s*\n",
            r"Below is.*?:\s*\n",
            r"This version includes.*?\n(?:\d+\..*?\n)*",  # Remove numbered lists
            r"This file.*?:\s*\n"
        ]

        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Step 2: Extract code blocks if present
        # Look for code between 
        # Step 2: Extract code blocks if present
        # Look for code between ```python and ``` or ``` and ```
        code_block_patterns = [
            r'```python\n?(.*?)```',
            r'```\n?(.*?)```',
            r'`([^`]+)`'  # Single backticks
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 50:  # Only use if substantial
                    cleaned = extracted
                    break
        
        # Step 3: Remove markdown formatting
        cleaned = re.sub(r'```python\n?', '', cleaned)
        cleaned = re.sub(r'```\n?', '', cleaned)
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
        
        # Step 4: Find the start of actual Python code
        lines = cleaned.split('\n')
        code_start = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Look for Python code indicators
            if (stripped.startswith('"""') or 
                stripped.startswith("'''") or
                stripped.startswith('import ') or 
                stripped.startswith('from ') or
                stripped.startswith('class ') or
                stripped.startswith('def ') or
                stripped.startswith('#')):
                code_start = i
                break
        
        # Step 5: Find the end of actual Python code
        code_lines = lines[code_start:]
        code_end = len(code_lines)
        
        for i in reversed(range(len(code_lines))):
            line = code_lines[i].strip()
            if line and not line.startswith('#') and 'This version' not in line:
                code_end = i + 1
                break
        
        # Step 6: Reconstruct clean code
        final_lines = code_lines[:code_end]
        
        # Remove any remaining explanatory lines
        clean_lines = []
        for line in final_lines:
            stripped = line.strip()
            # Skip obviously non-code lines
            if (not stripped or 
                stripped.startswith('#') or
                any(phrase in stripped.lower() for phrase in [
                    'this version includes',
                    'this file contains',
                    'note:',
                    'example:',
                    'the above',
                    'explanation:'
                ])):
                if stripped.startswith('#') or not stripped:
                    clean_lines.append(line)  # Keep comments and empty lines
                # Skip other explanatory text
            else:
                clean_lines.append(line)
        
        # Step 7: Ensure proper file ending
        code = '\n'.join(clean_lines).strip()
        if code and not code.endswith('\n'):
            code += '\n'
        
        return code
    
    def _create_template_file(self, file_spec: str, purpose: str) -> str:
        """Create a basic template file when generation fails - NEW FALLBACK"""
        
        module_name = Path(file_spec).stem
        
        if file_spec.endswith('__init__.py'):
            return f'''"""
{Path(file_spec).parent.name} package
"""

__version__ = "0.1.0"

# Package imports will be added here
'''
        
        # Basic Python module template
        template = f'''#!/usr/bin/env python3
"""
{module_name.replace('_', ' ').title()}

{purpose}
"""

import logging
from typing import Dict, Any, Optional

class {self._to_class_name(module_name)}:
    """TODO: Implement {module_name.replace('_', ' ')}"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: Implement main processing logic"""
        self.logger.info("Processing data...")
        return {{"status": "success", "data": data}}

# TODO: Add additional functions and classes as needed
'''
        
        return template
    
    def _to_class_name(self, module_name: str) -> str:
        """Convert module_name to ClassName"""
        parts = module_name.split('_')
        return ''.join(word.capitalize() for word in parts)
    
    def _generate_init_file(self, file_spec: str) -> CodeFile:
        """Generate an __init__.py file - IMPROVED"""
        
        directory = str(Path(file_spec).parent)
        
        prompt = f"""{self._create_init_prompt()}

Generate __init__.py for directory: {directory}

Include:
- Package docstring
- Version info if appropriate  
- Main exports (__all__)
- Simple imports of key components

RESPOND WITH ONLY THE PYTHON CODE - NO EXPLANATIONS.

Directory: {directory}
"""
        
        try:
            response = self.model(
                prompt,
                max_tokens=800,
                temperature=0.2,  # Very low temperature for consistent init files
                top_p=0.8
            )
            
            raw_content = response['choices'][0]['text']
            content = self._aggressively_clean_code(raw_content)
            
            # Fallback for init files
            if not content or len(content) < 20:
                self.logger.warning(f"⚠️ Init file too short for {directory}, using template")
                content = f'''"""
{directory} package
"""

__version__ = "0.1.0"

# Package imports will be added here
'''
            
            return CodeFile(
                path=file_spec,
                file_type=CodeFileType.INIT_FILE,
                content=content,
                dependencies=[],
                description=f"Package init file for {directory}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error generating init file {file_spec}: {e}")
            # Simple fallback
            content = f'''"""
{directory} package
"""

__version__ = "0.1.0"
'''
            return CodeFile(
                path=file_spec,
                file_type=CodeFileType.INIT_FILE,
                content=content,
                dependencies=[],
                description=f"Template init file for {directory}"
            )
    
    def _determine_file_type(self, file_spec: str) -> CodeFileType:
        """Determine the type of file to generate"""
        
        file_path = Path(file_spec)
        
        if file_path.name == "__init__.py":
            return CodeFileType.INIT_FILE
        elif file_path.suffix == ".py":
            # Check if it's a class, module, or script based on name patterns
            if any(word in file_path.stem.lower() for word in ['agent', 'manager', 'engine', 'validator']):
                return CodeFileType.PYTHON_CLASS
            elif file_path.stem.startswith('test_'):
                return CodeFileType.TEST_FILE
            else:
                return CodeFileType.PYTHON_MODULE
        elif file_path.suffix in ['.yaml', '.yml', '.json', '.toml']:
            return CodeFileType.CONFIG_FILE
        else:
            return CodeFileType.PYTHON_SCRIPT
    
    def _extract_module_purpose(self, file_spec: str) -> str:
        """Extract the purpose of a module from its path and name"""
        
        path_parts = Path(file_spec).parts
        file_name = Path(file_spec).stem
        
        # Common patterns
        purpose_mapping = {
            'base_agent': 'Abstract base class for all agents',
            'model_manager': 'Unified model management for all agents',
            'agent_registry': 'Agent discovery and registration system',
            'message_bus': 'Inter-agent communication system',
            'collaborator': 'Agent collaboration logic',
            'session_manager': 'Collaboration session tracking',
            'command_validator': 'Command safety validation',
            'execution_engine': 'Safe command execution',
            'safety_monitor': 'Runtime safety monitoring',
            'result_analyzer': 'Command result analysis',
            'pattern_learner': 'Learning from execution patterns',
            'workflow_engine': 'Multi-step workflow orchestration',
            'system_manager': 'Overall system management'
        }
        
        if file_name in purpose_mapping:
            return purpose_mapping[file_name]
        
        # Fallback: generate purpose from directory context
        if 'core' in path_parts:
            return f"Core framework component for {file_name}"
        elif 'collaboration' in path_parts:
            return f"Agent collaboration component for {file_name}"
        elif 'safety' in path_parts:
            return f"Safety and execution component for {file_name}"
        elif 'analysis' in path_parts:
            return f"Result analysis component for {file_name}"
        else:
            return f"System component for {file_name}"
    
    def _clean_generated_code(self, raw_code: str) -> str:
        """Clean and format generated code"""
        
        # Remove markdown code blocks if present
        code = re.sub(r'```python\n?', '', raw_code)
        code = re.sub(r'```\n?', '', code)
        
        # Clean up extra whitespace
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines at the beginning
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)
        
        # Ensure proper file ending
        code = '\n'.join(cleaned_lines)
        if not code.endswith('\n'):
            code += '\n'
        
        return code
    
    def _extract_structure_from_response(self, response: str) -> Dict[str, Any]:
        """Extract directory structure from LLM response"""
        
        # Look for directory structure in the response
        directories = []
        creation_order = []
        
        # Find directory patterns
        dir_pattern = r'[\s]*([a-zA-Z_][a-zA-Z0-9_/]*/)[\s]*#'
        dir_matches = re.findall(dir_pattern, response)
        
        for match in dir_matches:
            directory = match.rstrip('/')
            if directory and directory not in directories:
                directories.append(directory)
        
        # Common file patterns for this type of project
        common_files = [
            'core/__init__.py',
            'core/base_agent.py',
            'core/model_manager.py',
            'core/agent_registry.py',
            'collaboration/__init__.py',
            'collaboration/message_bus.py',
            'collaboration/collaborator.py',
            'collaboration/session_manager.py',
            'safety/__init__.py',
            'safety/command_validator.py',
            'safety/execution_engine.py',
            'safety/safety_monitor.py',
            'agents/__init__.py',
            'agents/triage_agent.py',
            'agents/security_agent.py',
            'agents/executor_agent.py',
            'agents/coordinator_agent.py',
            'analysis/__init__.py',
            'analysis/result_analyzer.py',
            'analysis/pattern_learner.py',
            'analysis/insight_extractor.py',
            'config/__init__.py',
            'config/settings.py',
            'config/agent_configs.py',
            'orchestration/__init__.py',
            'orchestration/file_monitor.py',
            'orchestration/workflow_engine.py',
            'orchestration/system_manager.py'
        ]
        
        # If no directories found in response, use default structure
        if not directories:
            directories = [
                'core',
                'collaboration', 
                'safety',
                'agents',
                'analysis',
                'config',
                'orchestration',
                'tests'
            ]
        
        creation_order = common_files
        
        return {
            'directories': directories,
            'creation_order': creation_order
        }
    
    def _create_parsing_prompt(self) -> str:
        """Create prompt for parsing specifications"""
        return """You are a Coder Agent that parses modular software specifications.

Your task is to analyze the specification and identify:
1. Directory structure to create
2. Python files to generate
3. Configuration files needed
4. Dependencies between modules

Look for directory patterns like:
- core/ (framework components)
- collaboration/ (communication)
- safety/ (execution control)
- agents/ (specific agents)
- analysis/ (result processing)
- config/ (configuration)
- orchestration/ (system management)

Extract the structure and reply with:
DIRECTORIES:
[list of directories to create]

FILES:
[list of files to generate]

ANALYSIS:
[brief analysis of the specification]

Specification to analyze:
"""
    
    def _create_code_prompt(self) -> str:
        """Create prompt for code generation - IMPROVED"""
        return """You are a Coder Agent that generates clean, professional Python code.

CRITICAL INSTRUCTIONS:
- Write ONLY Python code, no explanations
- No markdown formatting (no ```)
- No introductory text like "Here's the code:"
- No explanatory comments after the code
- Start directly with the Python code

Generate code that is:
- Well-documented with docstrings
- Follows Python best practices  
- Includes proper error handling
- Has clear class and function structure
- Uses type hints where appropriate
- Includes necessary imports

RESPOND WITH ONLY THE PYTHON CODE - NOTHING ELSE.
"""
    
    def _create_init_prompt(self) -> str:
        """Create prompt for __init__.py generation - IMPROVED"""
        return """Generate a clean __init__.py file.

CRITICAL: Return ONLY Python code, no explanations or markdown.

Include:
- Package docstring
- Version if appropriate
- Main imports and exports (__all__)
- Keep it minimal and clean

RESPOND WITH ONLY THE PYTHON CODE.
"""
    
    def _create_structure_prompt(self) -> str:
        """Create prompt for structure analysis"""
        return """Analyze this modular specification and extract the directory structure.

Focus on identifying:
- Main module directories
- Subdirectory organization
- File dependencies
- Creation order

Reply with structured information about directories and files to create.
"""
    
    def _update_stats(self, result: CoderResult):
        """Update internal statistics"""
        self.stats['projects_processed'] += 1
        self.stats['files_generated'] += len(result.files_created)
        self.stats['directories_created'] += len(result.directories_created)
        
        if result.success:
            self.stats['successful_generations'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            **self.stats,
            'success_rate': self.stats['successful_generations'] / max(1, self.stats['projects_processed']),
            'avg_files_per_project': self.stats['files_generated'] / max(1, self.stats['projects_processed']),
            'avg_lines_per_file': self.stats['total_lines_written'] / max(1, self.stats['files_generated'])
        }