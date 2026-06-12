#!/usr/bin/env python3
"""
Enhanced Metadata Extractor

Extracts rich metadata from code and text chunks:
- Programming languages (10+ languages)
- Domains (security, science, systems, web, AI, graphics)
- Topics (buffer_overflow, physics_simulation, neural_networks)
- Code patterns (unsafe operations, optimizations)
- Quality indicators (complexity, examples, comments)
"""

import re
import ast
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import logging


@dataclass
class EnhancedMetadata:
    """Complete metadata for a chunk"""

    # Language & Structure
    language: str = "unknown"
    language_confidence: float = 0.0

    # Code Structure
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    # Semantic Content
    keywords: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    # Quality Indicators
    complexity: str = "medium"  # simple, medium, complex
    has_examples: bool = False
    has_comments: bool = False
    has_documentation: bool = False

    # Code Patterns
    code_patterns: List[str] = field(default_factory=list)

    # Relationships
    related_concepts: List[str] = field(default_factory=list)


class LanguageDetector:
    """
    Advanced language detection for 10+ programming languages

    Supports: Python, C, C++, Rust, Ruby, GDScript, JavaScript,
              Bash/Shell, Go, Java, and more
    """

    LANGUAGE_PATTERNS = {
        'python': {
            'strong': [
                r'^\s*def\s+\w+\s*\(',
                r'^\s*class\s+\w+',
                r'^\s*import\s+\w+',
                r'^\s*from\s+\w+\s+import',
                r'if\s+__name__\s*==\s*["\']__main__["\']',
                r'@\w+\s*\n\s*def',  # Decorators
            ],
            'weak': [
                r'self\.\w+',
                r'\.append\(',
                r'\.format\(',
                r'print\(',
            ],
            'extensions': ['.py', '.pyw'],
        },
        'c': {
            'strong': [
                r'#include\s*<\w+\.h>',
                r'^\s*int\s+main\s*\(',
                r'printf\s*\(',
                r'malloc\s*\(',
                r'typedef\s+struct',
            ],
            'weak': [
                r'->\w+',
                r'&\w+',
                r'NULL',
            ],
            'extensions': ['.c', '.h'],
        },
        'cpp': {
            'strong': [
                r'#include\s*<\w+>',
                r'namespace\s+\w+',
                r'std::',
                r'class\s+\w+\s*{',
                r'template\s*<',
                r'public:|private:|protected:',
            ],
            'weak': [
                r'::\w+',
                r'cout\s*<<',
                r'new\s+\w+',
            ],
            'extensions': ['.cpp', '.cc', '.cxx', '.hpp', '.hxx'],
        },
        'rust': {
            'strong': [
                r'fn\s+\w+\s*\(',
                r'impl\s+\w+',
                r'pub\s+fn',
                r'use\s+\w+::',
                r'let\s+mut\s+',
                r'match\s+\w+\s*{',
            ],
            'weak': [
                r'Ok\(',
                r'Err\(',
                r'Some\(',
                r'None',
                r'&str',
            ],
            'extensions': ['.rs'],
        },
        'ruby': {
            'strong': [
                r'^\s*def\s+\w+',
                r'^\s*class\s+\w+',
                r'^\s*module\s+\w+',
                r'require\s+["\']',
                r'end\s*$',
                r'do\s+\|',
            ],
            'weak': [
                r'@\w+',
                r'puts\s+',
                r'\.each\s+',
                r'=>',
            ],
            'extensions': ['.rb'],
        },
        'gdscript': {
            'strong': [
                r'extends\s+\w+',
                r'func\s+_ready\(',
                r'func\s+_process\(',
                r'@export',
                r'signal\s+\w+',
                r'var\s+\w+:\s*\w+',
            ],
            'weak': [
                r'\$\w+',
                r'get_node\(',
                r'queue_free\(',
                r'Vector[23]\(',
            ],
            'extensions': ['.gd'],
        },
        'javascript': {
            'strong': [
                r'function\s+\w+\s*\(',
                r'const\s+\w+\s*=',
                r'let\s+\w+\s*=',
                r'var\s+\w+\s*=',
                r'=>',
                r'import\s+.+\s+from',
                r'export\s+(default|const)',
            ],
            'weak': [
                r'console\.log',
                r'\.then\(',
                r'async\s+',
                r'await\s+',
            ],
            'extensions': ['.js', '.jsx', '.mjs'],
        },
        'bash': {
            'strong': [
                r'^#!/bin/(ba)?sh',
                r'^\s*function\s+\w+',
                r'\[\[\s+',
                r'if\s+\[\s+',
                r'for\s+\w+\s+in\s+',
                r'while\s+\[\s+',
            ],
            'weak': [
                r'\$\{?\w+\}?',
                r'echo\s+',
                r'export\s+',
                r'\|\s*\w+',
            ],
            'extensions': ['.sh', '.bash'],
        },
        'go': {
            'strong': [
                r'package\s+\w+',
                r'func\s+\w+\s*\(',
                r'import\s+\(',
                r'type\s+\w+\s+struct',
                r'go\s+\w+\(',
                r'defer\s+',
            ],
            'weak': [
                r':=',
                r'fmt\.',
                r'make\(',
                r'append\(',
            ],
            'extensions': ['.go'],
        },
        'java': {
            'strong': [
                r'public\s+class\s+\w+',
                r'public\s+static\s+void\s+main',
                r'import\s+java\.',
                r'@Override',
                r'extends\s+\w+',
                r'implements\s+\w+',
            ],
            'weak': [
                r'System\.out',
                r'new\s+\w+\s*\(',
                r'\.length\(\)',
            ],
            'extensions': ['.java'],
        },
    }

    def __init__(self):
        self.logger = logging.getLogger("LanguageDetector")

    def detect(self, content: str, file_path: str = "") -> tuple[str, float]:
        """
        Detect programming language with confidence score

        Returns:
            (language, confidence) where confidence is 0.0-1.0
        """
        # Check file extension first
        if file_path:
            ext = file_path.lower().split('.')[-1] if '.' in file_path else ""
            ext_with_dot = f".{ext}" if ext else ""

            for lang, patterns in self.LANGUAGE_PATTERNS.items():
                if ext_with_dot in patterns.get('extensions', []):
                    return (lang, 0.9)  # High confidence from extension

        # Content-based detection
        scores = {}
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0.0

            # Check strong patterns (high weight)
            for pattern in patterns.get('strong', []):
                matches = len(re.findall(pattern, content, re.MULTILINE))
                score += matches * 3.0

            # Check weak patterns (low weight)
            for pattern in patterns.get('weak', []):
                matches = len(re.findall(pattern, content, re.MULTILINE))
                score += matches * 1.0

            scores[lang] = score

        if not scores or max(scores.values()) == 0:
            return ("generic", 0.0)

        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]

        # Normalize confidence (0-1)
        confidence = min(best_score / 10.0, 1.0)

        return (best_lang, confidence)


class DomainExtractor:
    """
    Extract domains and topics from code/text

    Domains: High-level categories (security, science, systems, web, AI, graphics)
    Topics: Specific subjects (buffer_overflow, neural_networks, physics_simulation)
    """

    DOMAIN_PATTERNS = {
        'security': {
            'keywords': [
                'exploit', 'vulnerability', 'buffer overflow', 'overflow',
                'injection', 'xss', 'csrf', 'authentication', 'authorization',
                'encrypt', 'decrypt', 'crypto', 'hash', 'password', 'salt',
                'privilege', 'escalation', 'sandbox', 'shellcode', 'payload',
                'attack', 'defend', 'secure', 'insecure', 'unsafe',
            ],
            'patterns': [
                r'buffer\s*overflow',
                r'stack\s*overflow',
                r'heap\s*overflow',
                r'use.after.free',
                r'format\s*string',
                r'sql\s*injection',
                r'remote\s*code\s*execution',
            ],
            'functions': [
                'strcpy', 'strcat', 'gets', 'sprintf', 'scanf',
                'malloc', 'free', 'mprotect', 'execve',
            ],
        },
        'systems': {
            'keywords': [
                'memory', 'allocation', 'pointer', 'address', 'kernel',
                'process', 'thread', 'mutex', 'semaphore', 'lock',
                'file system', 'inode', 'syscall', 'interrupt',
                'scheduling', 'context switch', 'virtual memory', 'page',
            ],
            'patterns': [
                r'memory\s*(management|allocation)',
                r'file\s*system',
                r'process\s*management',
                r'virtual\s*memory',
            ],
            'functions': [
                'fork', 'exec', 'wait', 'pthread', 'mmap',
                'open', 'read', 'write', 'close', 'ioctl',
            ],
        },
        'networking': {
            'keywords': [
                'socket', 'tcp', 'udp', 'http', 'https', 'packet',
                'protocol', 'ip', 'dns', 'routing', 'firewall',
                'port', 'network', 'bandwidth', 'latency',
            ],
            'patterns': [
                r'http[s]?://',
                r'tcp/ip',
                r'socket\s*programming',
            ],
            'functions': [
                'socket', 'bind', 'listen', 'accept', 'connect',
                'send', 'recv', 'sendto', 'recvfrom',
            ],
        },
        'physics': {
            'keywords': [
                'velocity', 'acceleration', 'force', 'mass', 'energy',
                'momentum', 'collision', 'gravity', 'friction',
                'simulation', 'particle', 'rigid body', 'dynamics',
                'kinematics', 'newton', 'physics engine',
            ],
            'patterns': [
                r'f\s*=\s*m\s*\*\s*a',
                r'physics\s*simulation',
                r'collision\s*detection',
            ],
        },
        'neuroscience': {
            'keywords': [
                'neuron', 'synapse', 'neural', 'brain', 'spike',
                'membrane potential', 'action potential', 'dendrite',
                'axon', 'neurotransmitter', 'plasticity', 'learning',
                'hebbian', 'stdp', 'receptive field',
            ],
            'patterns': [
                r'neural\s*network',
                r'spike\s*train',
                r'membrane\s*potential',
            ],
        },
        'ai_ml': {
            'keywords': [
                'neural network', 'deep learning', 'machine learning',
                'training', 'inference', 'model', 'layer', 'weights',
                'gradient', 'backpropagation', 'optimizer', 'loss',
                'accuracy', 'precision', 'recall', 'dataset',
                'tensor', 'activation', 'embedding',
            ],
            'patterns': [
                r'neural\s*network',
                r'deep\s*learning',
                r'machine\s*learning',
                r'gradient\s*descent',
            ],
            'functions': [
                'train', 'predict', 'fit', 'transform',
            ],
        },
        'graphics': {
            'keywords': [
                'render', 'shader', 'vertex', 'fragment', 'pixel',
                'texture', 'mesh', 'polygon', 'rasterization',
                'opengl', 'vulkan', 'directx', 'gpu', 'framebuffer',
                'lighting', 'shadow', 'ray tracing',
            ],
            'patterns': [
                r'vertex\s*shader',
                r'fragment\s*shader',
                r'ray\s*tracing',
            ],
        },
        'web': {
            'keywords': [
                'http', 'api', 'rest', 'graphql', 'json', 'xml',
                'request', 'response', 'endpoint', 'route',
                'middleware', 'cors', 'cookie', 'session',
            ],
            'patterns': [
                r'@app\.route',
                r'app\.(get|post|put|delete)',
                r'fetch\(',
            ],
        },
    }

    TOPIC_PATTERNS = {
        'buffer_overflow': [
            r'buffer\s*overflow', r'strcpy', r'strcat', r'gets',
            r'stack\s*overflow', r'heap\s*overflow',
        ],
        'cryptography': [
            r'encrypt', r'decrypt', r'cipher', r'aes', r'rsa',
            r'hash', r'sha256', r'md5',
        ],
        'memory_management': [
            r'malloc', r'free', r'realloc', r'memory\s*leak',
            r'garbage\s*collection',
        ],
        'concurrency': [
            r'thread', r'mutex', r'semaphore', r'deadlock',
            r'race\s*condition', r'atomic',
        ],
        'algorithms': [
            r'sort', r'search', r'binary\s*tree', r'graph',
            r'dynamic\s*programming', r'recursion',
        ],
    }

    def __init__(self):
        self.logger = logging.getLogger("DomainExtractor")

    def extract_domains(self, content: str, language: str = "unknown") -> List[str]:
        """Extract domains from content"""
        content_lower = content.lower()
        detected_domains = set()

        for domain, config in self.DOMAIN_PATTERNS.items():
            score = 0.0

            # Check keywords
            for keyword in config.get('keywords', []):
                if keyword.lower() in content_lower:
                    score += 1.0

            # Check patterns
            for pattern in config.get('patterns', []):
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                score += matches * 2.0

            # Check functions
            for func in config.get('functions', []):
                if func in content:
                    score += 1.5

            # Threshold for inclusion
            if score >= 2.0:
                detected_domains.add(domain)

        return sorted(list(detected_domains))

    def extract_topics(self, content: str) -> List[str]:
        """Extract specific topics from content"""
        detected_topics = set()

        for topic, patterns in self.TOPIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    detected_topics.add(topic)
                    break

        return sorted(list(detected_topics))


class CodeAnalyzer:
    """Analyze code structure and extract metadata"""

    def __init__(self):
        self.logger = logging.getLogger("CodeAnalyzer")

    def extract_python_metadata(self, content: str) -> Dict:
        """Extract metadata from Python code using AST"""
        metadata = {
            'functions': [],
            'classes': [],
            'imports': [],
            'has_comments': False,
            'has_documentation': False,
        }

        try:
            tree = ast.parse(content)

            # Extract functions, classes, imports
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    metadata['functions'].append(node.name)
                    if ast.get_docstring(node):
                        metadata['has_documentation'] = True

                elif isinstance(node, ast.ClassDef):
                    metadata['classes'].append(node.name)
                    if ast.get_docstring(node):
                        metadata['has_documentation'] = True

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        metadata['imports'].append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        metadata['imports'].append(node.module)

            # Check for comments
            if '#' in content:
                metadata['has_comments'] = True

        except SyntaxError:
            pass

        return metadata

    def extract_generic_metadata(self, content: str, language: str) -> Dict:
        """Extract metadata from non-Python code using patterns"""
        metadata = {
            'functions': [],
            'classes': [],
            'imports': [],
            'has_comments': False,
            'has_documentation': False,
        }

        # Function detection (C, C++, Rust, etc.)
        func_patterns = {
            'c': r'^\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{',
            'cpp': r'^\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{',
            'rust': r'fn\s+(\w+)\s*\(',
            'ruby': r'def\s+(\w+)',
            'gdscript': r'func\s+(\w+)\s*\(',
            'go': r'func\s+(\w+)\s*\(',
            'java': r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(',
        }

        if language in func_patterns:
            matches = re.findall(func_patterns[language], content, re.MULTILINE)
            metadata['functions'] = matches[:20]  # Limit to 20

        # Class detection
        class_patterns = {
            'cpp': r'class\s+(\w+)',
            'rust': r'struct\s+(\w+)',
            'ruby': r'class\s+(\w+)',
            'gdscript': r'class_name\s+(\w+)',
            'java': r'class\s+(\w+)',
        }

        if language in class_patterns:
            matches = re.findall(class_patterns[language], content)
            metadata['classes'] = matches[:20]

        # Import detection
        import_patterns = {
            'c': r'#include\s*[<"](\w+(?:\.\w+)?)[>"]',
            'cpp': r'#include\s*[<"](\w+(?:\.\w+)?)[>"]',
            'rust': r'use\s+([\w:]+)',
            'ruby': r'require\s+["\'](\w+)["\']',
            'go': r'import\s+"([^"]+)"',
            'java': r'import\s+([\w.]+)',
        }

        if language in import_patterns:
            matches = re.findall(import_patterns[language], content)
            metadata['imports'] = matches[:20]

        # Comments detection
        comment_patterns = [
            r'//',  # C++, Rust, Go, GDScript
            r'#',   # Python, Ruby, Bash
            r'/\*', # C, C++
        ]

        for pattern in comment_patterns:
            if re.search(pattern, content):
                metadata['has_comments'] = True
                break

        # Documentation (docstrings, javadoc, etc.)
        doc_patterns = [
            r'/\*\*',  # Javadoc, JSDoc
            r'##',     # Ruby, GDScript
            r'"""',    # Python
        ]

        for pattern in doc_patterns:
            if pattern in content:
                metadata['has_documentation'] = True
                break

        return metadata


class ComplexityAnalyzer:
    """Analyze code complexity"""

    def estimate_complexity(self, content: str, language: str) -> str:
        """
        Estimate code complexity: simple, medium, complex

        Based on:
        - Number of control structures (if, for, while)
        - Nesting depth
        - Function length
        - Number of functions/classes
        """
        # Count control structures
        control_count = 0
        control_patterns = [
            r'\bif\s+', r'\bfor\s+', r'\bwhile\s+',
            r'\bswitch\s+', r'\bmatch\s+', r'\btry\s+',
        ]

        for pattern in control_patterns:
            control_count += len(re.findall(pattern, content, re.IGNORECASE))

        # Estimate nesting depth
        max_indent = 0
        for line in content.split('\n'):
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent)

        nesting_depth = max_indent // 4  # Assume 4-space indentation

        # Line count
        line_count = len(content.split('\n'))

        # Calculate complexity score
        score = 0
        score += control_count * 2
        score += nesting_depth * 3
        score += line_count / 10

        if score < 10:
            return "simple"
        elif score < 30:
            return "medium"
        else:
            return "complex"


class MetadataExtractor:
    """Main metadata extractor - coordinates all extractors"""

    def __init__(self):
        self.language_detector = LanguageDetector()
        self.domain_extractor = DomainExtractor()
        self.code_analyzer = CodeAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.logger = logging.getLogger("MetadataExtractor")

    def extract(self, content: str, file_path: str = "",
                existing_language: str = None) -> EnhancedMetadata:
        """
        Extract complete metadata from content

        Args:
            content: The text/code content
            file_path: Optional file path for extension detection
            existing_language: Optional pre-detected language

        Returns:
            EnhancedMetadata object with all extracted metadata
        """
        metadata = EnhancedMetadata()

        # 1. Language detection
        if existing_language:
            metadata.language = existing_language
            metadata.language_confidence = 1.0
        else:
            lang, confidence = self.language_detector.detect(content, file_path)
            metadata.language = lang
            metadata.language_confidence = confidence

        # 2. Code structure extraction
        if metadata.language == 'python':
            code_meta = self.code_analyzer.extract_python_metadata(content)
        else:
            code_meta = self.code_analyzer.extract_generic_metadata(content, metadata.language)

        metadata.functions = code_meta['functions']
        metadata.classes = code_meta['classes']
        metadata.imports = code_meta['imports']
        metadata.has_comments = code_meta['has_comments']
        metadata.has_documentation = code_meta['has_documentation']

        # 3. Domain extraction
        metadata.domains = self.domain_extractor.extract_domains(content, metadata.language)
        metadata.topics = self.domain_extractor.extract_topics(content)

        # 4. Keywords extraction (top words)
        metadata.keywords = self._extract_keywords(content)

        # 5. Complexity analysis
        metadata.complexity = self.complexity_analyzer.estimate_complexity(content, metadata.language)

        # 6. Check for examples
        metadata.has_examples = self._has_examples(content)

        # 7. Code patterns
        metadata.code_patterns = self._extract_code_patterns(content, metadata.language)

        return metadata

    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract top keywords from content"""
        # Simple keyword extraction - can be enhanced with TF-IDF
        words = re.findall(r'\b[a-z_][a-z0-9_]{2,}\b', content.lower())

        # Filter common words
        stopwords = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from',
            'return', 'int', 'str', 'float', 'bool', 'void', 'var',
            'def', 'class', 'function', 'import', 'include',
        }

        words = [w for w in words if w not in stopwords]

        # Count frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]

    def _has_examples(self, content: str) -> bool:
        """Check if content contains examples"""
        example_indicators = [
            r'example:', r'e\.g\.', r'for example',
            r'usage:', r'demo:', r'test_',
            r'>>> ',  # Python REPL
        ]

        for pattern in example_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _extract_code_patterns(self, content: str, language: str) -> List[str]:
        """Detect specific code patterns"""
        patterns = []

        # Unsafe operations (security)
        if language in ['c', 'cpp']:
            if re.search(r'strcpy|strcat|gets|sprintf', content):
                patterns.append('unsafe_string_operation')
            if re.search(r'malloc|free|realloc', content):
                patterns.append('manual_memory_management')

        # Concurrency patterns
        if re.search(r'mutex|lock|atomic|thread', content, re.IGNORECASE):
            patterns.append('concurrent_code')

        # Error handling
        if re.search(r'try|catch|except|error', content, re.IGNORECASE):
            patterns.append('error_handling')

        # Optimization hints
        if re.search(r'inline|constexpr|#pragma\s+optimize', content):
            patterns.append('optimization')

        return patterns

    def to_dict(self, metadata: EnhancedMetadata) -> Dict:
        """Convert metadata to dictionary for storage"""
        return {
            'language': metadata.language,
            'language_confidence': metadata.language_confidence,
            'functions': metadata.functions,
            'classes': metadata.classes,
            'imports': metadata.imports,
            'keywords': metadata.keywords,
            'domains': metadata.domains,
            'topics': metadata.topics,
            'complexity': metadata.complexity,
            'has_examples': metadata.has_examples,
            'has_comments': metadata.has_comments,
            'has_documentation': metadata.has_documentation,
            'code_patterns': metadata.code_patterns,
        }
