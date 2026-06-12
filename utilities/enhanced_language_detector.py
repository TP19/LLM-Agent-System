"""
Enhanced Language Detector for Code Snippets
===========================================

Advanced language detection with confidence scoring for PDF-extracted code.

Features:
- Weighted pattern matching (strong/medium/weak)
- 10+ languages supported
- Context-aware detection (uses surrounding text hints)
- Confidence scoring (0.0-1.0)
- Voting-based approach for robustness

Languages Supported:
- Python, C, C++, Rust, Ruby, GDScript
- JavaScript, TypeScript, Bash/Shell, Go, Java

Author: Claude Code
Date: 2025-11-07
"""

import re
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class PatternStrength(Enum):
    """Pattern match strength levels"""
    STRONG = 3.0    # Unique to language, high confidence
    MEDIUM = 1.5    # Common but not unique
    WEAK = 0.5      # Supportive but ambiguous


@dataclass
class LanguagePattern:
    """Language detection patterns with weights"""
    strong: List[str]          # Strong indicators (unique to language)
    medium: List[str]          # Medium indicators (common)
    weak: List[str]            # Weak indicators (supportive)
    keywords: List[str]        # Language keywords
    context_hints: List[str]   # Context clues in surrounding text


@dataclass
class DetectionResult:
    """Language detection result with confidence"""
    language: str
    confidence: float
    scores: Dict[str, float]  # Scores for all languages
    pattern_matches: Dict[str, int]  # Number of patterns matched per strength
    context_boost: bool  # Whether context hints boosted confidence


class EnhancedLanguageDetector:
    """
    Advanced language detection for code snippets.

    Uses weighted pattern matching with confidence scoring.
    Particularly effective for PDF-extracted code where traditional
    detection fails due to mixed content and small snippets.
    """

    # Language patterns with weighted indicators
    LANGUAGE_PATTERNS = {
        'python': LanguagePattern(
            strong=[
                r'def\s+\w+\s*\(',           # Function definition
                r'import\s+\w+',             # Import statement
                r'from\s+\w+\s+import',      # From import
                r'class\s+\w+\s*:',          # Class definition
                r'if\s+__name__\s*==',       # Main guard
                r'@\w+\s*\n\s*def',          # Decorator
                r'elif\s+',                  # Elif keyword
            ],
            medium=[
                r'print\s*\(',               # Print function
                r':\s*$',                    # Colon at line end
                r'self\.\w+',                # Self references
                r'\.\w+\(',                  # Method calls
                r'for\s+\w+\s+in\s+',        # For-in loop
                r'with\s+\w+',               # With statement
            ],
            weak=[
                r'#\s*[A-Z]',                # Comments starting with capital
                r'\[.*\]',                   # List literals
                r'{\s*\w+\s*:',              # Dict literals
            ],
            keywords=['def', 'class', 'import', 'from', 'return', 'yield', 'lambda', 'pass', 'raise', 'try', 'except', 'finally', 'with', 'as'],
            context_hints=['python', 'py', '.py', 'python code', 'python program', 'python script']
        ),

        'c': LanguagePattern(
            strong=[
                r'#include\s*[<"]',          # Include directive
                r'int\s+main\s*\(',          # Main function
                r'printf\s*\(',              # Printf
                r'scanf\s*\(',               # Scanf
                r'struct\s+\w+\s*\{',        # Struct definition
                r'typedef\s+struct',         # Typedef struct
                r'malloc\s*\(',              # Memory allocation
                r'free\s*\(',                # Free memory
            ],
            medium=[
                r'\w+\s*\*\s*\w+',           # Pointer declaration
                r'sizeof\s*\(',              # Sizeof operator
                r'strcpy\s*\(',              # String functions
                r'strcmp\s*\(',
                r'strlen\s*\(',
                r'NULL',                     # NULL constant
                r'void\s+\w+\s*\(',          # Void function
            ],
            weak=[
                r'//\s*',                    # C++ style comments (valid in C99+)
                r'/\*.*?\*/',                # Multi-line comments
                r'return\s+\d+;',            # Return statement
            ],
            keywords=['int', 'char', 'void', 'struct', 'if', 'while', 'for', 'return', 'NULL', 'typedef', 'sizeof', 'unsigned', 'signed', 'const', 'static', 'extern'],
            context_hints=['c code', 'c program', '.c', 'gcc', 'clang', 'in c', 'c language', 'the c ', 'using c']
        ),

        'cpp': LanguagePattern(
            strong=[
                r'#include\s*<\w+>',         # C++ includes
                r'std::',                    # Standard namespace
                r'cout\s*<<',                # Cout
                r'cin\s*>>',                 # Cin
                r'class\s+\w+\s*\{',         # Class definition
                r'template\s*<',             # Template
                r'namespace\s+\w+',          # Namespace
                r'using\s+namespace',        # Using namespace
                r'public:|private:|protected:', # Access specifiers
            ],
            medium=[
                r'new\s+\w+',                # New operator
                r'delete\s+',                # Delete operator
                r'::\w+',                    # Scope resolution
                r'virtual\s+',               # Virtual keyword
                r'override\s',               # Override keyword
                r'->\w+',                    # Arrow operator
            ],
            weak=[
                r'//\s*',                    # Comments
                r'/\*.*?\*/',                # Multi-line comments
                r'bool\s+',                  # Bool type
            ],
            keywords=['class', 'public', 'private', 'protected', 'virtual', 'namespace', 'template', 'typename', 'new', 'delete', 'try', 'catch', 'throw', 'std', 'bool', 'true', 'false'],
            context_hints=['c++', 'cpp', '.cpp', '.hpp', 'c++ code', 'c++ program', 'using c++']
        ),

        'bash': LanguagePattern(
            strong=[
                r'^\s*\$\s+',                # Shell prompt
                r'^#!\s*/bin/(ba)?sh',       # Shebang
                r'apt-get\s+',               # Debian package manager
                r'yum\s+',                   # RedHat package manager
                r'sudo\s+',                  # Sudo command
                r'chmod\s+\d+',              # Chmod with permissions
                r'export\s+\w+=',            # Export variable
            ],
            medium=[
                r'\|\s*grep',                # Piping to grep
                r'>\s*\w+\.\w+',             # Output redirection
                r'2>&1',                     # Error redirection
                r'echo\s+["\']',             # Echo with quotes
                r'if\s*\[\s*',               # If statement
                r'fi\s*$',                   # Fi keyword
                r'\$\{?\w+\}?',              # Variable expansion
            ],
            weak=[
                r'#\s*',                     # Comments
                r'cd\s+',                    # Change directory
                r'ls\s+',                    # List
                r'pwd',                      # Print working directory
            ],
            keywords=['if', 'then', 'else', 'elif', 'fi', 'for', 'do', 'done', 'while', 'case', 'esac', 'function', 'return', 'exit', 'source', 'export'],
            context_hints=['bash', 'shell', 'command', 'terminal', '$', 'shell script', 'bash script', '.sh']
        ),

        'javascript': LanguagePattern(
            strong=[
                r'function\s+\w+\s*\(',      # Function declaration
                r'const\s+\w+\s*=',          # Const declaration
                r'let\s+\w+\s*=',            # Let declaration
                r'=>\s*\{',                  # Arrow function
                r'console\.log\s*\(',        # Console.log
                r'async\s+function',         # Async function
                r'await\s+',                 # Await keyword
                r'require\s*\(',             # Require (Node.js)
            ],
            medium=[
                r'var\s+\w+\s*=',            # Var declaration
                r'function\s*\(',            # Anonymous function
                r'\.then\s*\(',              # Promise then
                r'\.catch\s*\(',             # Promise catch
                r'typeof\s+',                # Typeof operator
                r'===',                      # Strict equality
                r'!==',                      # Strict inequality
            ],
            weak=[
                r'//\s*',                    # Comments
                r'/\*.*?\*/',                # Multi-line comments
                r'{\s*\w+\s*:',              # Object literals
                r'\[\s*\]',                  # Array literals
            ],
            keywords=['function', 'const', 'let', 'var', 'async', 'await', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'try', 'catch', 'throw', 'typeof', 'new'],
            context_hints=['javascript', 'js', '.js', 'node', 'nodejs', 'javascript code', 'js code']
        ),

        'typescript': LanguagePattern(
            strong=[
                r'interface\s+\w+',          # Interface declaration
                r':\s*\w+\s*=',              # Type annotation
                r'type\s+\w+\s*=',           # Type alias
                r'enum\s+\w+',               # Enum
                r'as\s+\w+',                 # Type assertion
                r'<\w+>',                    # Generic type
                r'implements\s+\w+',         # Implements
            ],
            medium=[
                r'public\s+\w+',             # Public modifier
                r'private\s+\w+',            # Private modifier
                r'readonly\s+',              # Readonly modifier
                r'constructor\s*\(',         # Constructor
            ],
            weak=[
                r'const\s+\w+\s*:',          # Const with type
                r'let\s+\w+\s*:',            # Let with type
            ],
            keywords=['interface', 'type', 'enum', 'public', 'private', 'protected', 'readonly', 'implements', 'extends', 'namespace', 'declare', 'abstract'],
            context_hints=['typescript', 'ts', '.ts', 'typescript code']
        ),

        'rust': LanguagePattern(
            strong=[
                r'fn\s+\w+\s*\(',            # Function definition
                r'let\s+mut\s+',             # Mutable variable
                r'impl\s+\w+',               # Implementation
                r'trait\s+\w+',              # Trait definition
                r'match\s+\w+\s*\{',         # Match expression
                r'pub\s+fn',                 # Public function
                r'#\[derive\(',              # Derive macro
            ],
            medium=[
                r'::',                       # Path separator
                r'&\w+',                     # Reference
                r'&mut\s+',                  # Mutable reference
                r'Some\(',                   # Option Some
                r'None',                     # Option None
                r'Ok\(',                     # Result Ok
                r'Err\(',                    # Result Err
                r'println!\(',               # Println macro
            ],
            weak=[
                r'//\s*',                    # Comments
                r'/\*.*?\*/',                # Multi-line comments
                r'let\s+\w+\s*=',            # Variable declaration
            ],
            keywords=['fn', 'let', 'mut', 'impl', 'trait', 'struct', 'enum', 'match', 'if', 'else', 'loop', 'while', 'for', 'in', 'return', 'pub', 'use', 'mod', 'crate', 'self', 'super'],
            context_hints=['rust', 'rs', '.rs', 'rust code', 'cargo']
        ),

        'ruby': LanguagePattern(
            strong=[
                r'def\s+\w+',                # Method definition
                r'class\s+\w+\s*<',          # Class inheritance
                r'module\s+\w+',             # Module definition
                r'require\s+["\']',          # Require statement
                r'attr_accessor\s+',         # Attribute accessor
                r'@@\w+',                    # Class variable
                r'@\w+',                     # Instance variable
            ],
            medium=[
                r'\.each\s+do',              # Each iterator
                r'\.map\s+do',               # Map iterator
                r'puts\s+',                  # Puts statement
                r'gets',                     # Gets statement
                r'unless\s+',                # Unless keyword
                r'elsif\s+',                 # Elsif keyword
            ],
            weak=[
                r'#\s*',                     # Comments
                r'end\s*$',                  # End keyword
                r'do\s*\|',                  # Block parameter
            ],
            keywords=['def', 'class', 'module', 'if', 'unless', 'elsif', 'else', 'case', 'when', 'while', 'until', 'for', 'do', 'end', 'return', 'yield', 'require', 'attr_accessor', 'attr_reader', 'attr_writer'],
            context_hints=['ruby', 'rb', '.rb', 'ruby code', 'rails']
        ),

        'gdscript': LanguagePattern(
            strong=[
                r'extends\s+\w+',            # Inheritance
                r'func\s+_ready\s*\(',       # Ready function
                r'func\s+_process\s*\(',     # Process function
                r'@export\s+var',            # Export annotation
                r'get_node\s*\(',            # Node access
                r'preload\s*\(',             # Resource preloading
                r'\$\w+',                    # Node reference
            ],
            medium=[
                r'func\s+\w+\s*\(',          # Function definition
                r'var\s+\w+\s*:',            # Typed variable
                r'\.connect\s*\(',           # Signal connection
                r'emit_signal\s*\(',         # Emit signal
                r'onready\s+var',            # Onready variable
                r'signal\s+\w+',             # Signal declaration
            ],
            weak=[
                r'#\s*',                     # Comments
                r'pass\s*$',                 # Pass statement
                r'var\s+\w+\s*=',            # Variable declaration
            ],
            keywords=['func', 'var', 'extends', 'signal', 'export', 'onready', 'pass', 'return', 'if', 'elif', 'else', 'for', 'while', 'match', 'break', 'continue', 'class_name'],
            context_hints=['gdscript', 'godot', '.gd', 'gdscript code', 'godot engine']
        ),

        'go': LanguagePattern(
            strong=[
                r'package\s+\w+',            # Package declaration
                r'func\s+\w+\s*\(',          # Function definition
                r'import\s+\(',              # Import block
                r'go\s+\w+\(',               # Goroutine
                r'chan\s+\w+',               # Channel type
                r'defer\s+',                 # Defer statement
                r':=',                       # Short variable declaration
            ],
            medium=[
                r'func\s*\(',                # Anonymous function
                r'interface\s*\{',           # Interface definition
                r'struct\s*\{',              # Struct definition
                r'make\s*\(',                # Make function
                r'range\s+',                 # Range keyword
                r'select\s*\{',              # Select statement
            ],
            weak=[
                r'//\s*',                    # Comments
                r'/\*.*?\*/',                # Multi-line comments
                r'fmt\.',                    # Fmt package
            ],
            keywords=['package', 'import', 'func', 'var', 'const', 'type', 'struct', 'interface', 'if', 'else', 'for', 'range', 'return', 'defer', 'go', 'chan', 'select', 'case', 'default', 'break', 'continue'],
            context_hints=['go', 'golang', '.go', 'go code', 'go program']
        ),

        'java': LanguagePattern(
            strong=[
                r'public\s+class\s+\w+',     # Public class
                r'public\s+static\s+void\s+main', # Main method
                r'import\s+java\.',          # Java import
                r'@Override',                # Override annotation
                r'extends\s+\w+',            # Class inheritance
                r'implements\s+\w+',         # Interface implementation
                r'new\s+\w+\s*\(',           # Object instantiation
            ],
            medium=[
                r'System\.out\.print',       # Print statements
                r'private\s+\w+\s+\w+',      # Private member
                r'protected\s+\w+',          # Protected member
                r'final\s+\w+',              # Final modifier
                r'static\s+\w+',             # Static modifier
                r'try\s*\{',                 # Try block
                r'catch\s*\(',               # Catch block
            ],
            weak=[
                r'//\s*',                    # Comments
                r'/\*.*?\*/',                # Multi-line comments
                r'null',                     # Null constant
            ],
            keywords=['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'import', 'package', 'new', 'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'try', 'catch', 'finally', 'throw', 'throws', 'void', 'int', 'boolean', 'String', 'static', 'final', 'abstract'],
            context_hints=['java', '.java', 'java code', 'java program']
        ),
    }

    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize language detector.

        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0)
                           Below this, return 'generic'
        """
        self.min_confidence = min_confidence

    def detect(
        self,
        code: str,
        context: Optional[str] = None,
        file_extension: Optional[str] = None
    ) -> DetectionResult:
        """
        Detect programming language with confidence score.

        Args:
            code: The code snippet to analyze
            context: Optional surrounding text for hints
            file_extension: Optional file extension (.py, .c, etc.)

        Returns:
            DetectionResult with language, confidence, and details
        """
        if not code or not code.strip():
            return DetectionResult(
                language='generic',
                confidence=0.0,
                scores={},
                pattern_matches={},
                context_boost=False
            )

        # Calculate scores for each language
        scores = {}
        all_pattern_matches = {}

        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            score, pattern_matches = self._score_language(code, patterns, context)
            scores[lang] = score
            all_pattern_matches[lang] = pattern_matches

        # Find best match
        if not scores or max(scores.values()) == 0:
            return DetectionResult(
                language='generic',
                confidence=0.0,
                scores=scores,
                pattern_matches={},
                context_boost=False
            )

        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]

        # Calculate confidence (normalize based on score distribution)
        total_score = sum(scores.values())
        if total_score == 0:
            confidence = 0.0
        else:
            # Confidence is the ratio of best score to total
            # High confidence = one language dominates
            # Low confidence = multiple languages have similar scores
            confidence = best_score / total_score

            # Boost confidence if patterns are strong
            strong_matches = all_pattern_matches[best_lang].get('strong', 0)
            if strong_matches >= 2:
                confidence = min(1.0, confidence * 1.2)
            elif strong_matches >= 1:
                confidence = min(1.0, confidence * 1.1)

        # Check if context hints boosted this result
        context_boost = False
        if context:
            context_lower = context.lower()
            lang_hints = self.LANGUAGE_PATTERNS[best_lang].context_hints
            if any(hint in context_lower for hint in lang_hints):
                context_boost = True

        # Return generic if confidence too low
        if confidence < self.min_confidence:
            return DetectionResult(
                language='generic',
                confidence=confidence,
                scores=scores,
                pattern_matches=all_pattern_matches[best_lang],
                context_boost=False
            )

        return DetectionResult(
            language=best_lang,
            confidence=confidence,
            scores=scores,
            pattern_matches=all_pattern_matches[best_lang],
            context_boost=context_boost
        )

    def _score_language(
        self,
        code: str,
        patterns: LanguagePattern,
        context: Optional[str] = None
    ) -> Tuple[float, Dict[str, int]]:
        """
        Calculate score for a specific language.

        Args:
            code: Code to analyze
            patterns: Language patterns
            context: Optional surrounding text

        Returns:
            (score, pattern_match_counts)
        """
        score = 0.0
        pattern_matches = {'strong': 0, 'medium': 0, 'weak': 0}

        # Count strong pattern matches
        for pattern in patterns.strong:
            matches = len(re.findall(pattern, code, re.MULTILINE | re.IGNORECASE))
            if matches > 0:
                pattern_matches['strong'] += matches
                score += matches * PatternStrength.STRONG.value

        # Count medium pattern matches
        for pattern in patterns.medium:
            matches = len(re.findall(pattern, code, re.MULTILINE | re.IGNORECASE))
            if matches > 0:
                pattern_matches['medium'] += matches
                score += matches * PatternStrength.MEDIUM.value

        # Count weak pattern matches
        for pattern in patterns.weak:
            matches = len(re.findall(pattern, code, re.MULTILINE | re.IGNORECASE))
            if matches > 0:
                pattern_matches['weak'] += matches
                score += matches * PatternStrength.WEAK.value

        # Keyword density (small bonus)
        if patterns.keywords:
            code_lower = code.lower()
            keyword_count = sum(1 for kw in patterns.keywords if kw.lower() in code_lower)
            score += keyword_count * 0.3

        # Context hint bonus (significant boost if explicit mention)
        if context and patterns.context_hints:
            context_lower = context.lower()
            for hint in patterns.context_hints:
                if hint in context_lower:
                    score += 2.0  # Big boost for explicit context hints
                    break

        return score, pattern_matches

    def detect_batch(
        self,
        code_snippets: List[Tuple[str, Optional[str]]],
    ) -> List[DetectionResult]:
        """
        Detect languages for multiple code snippets.

        Args:
            code_snippets: List of (code, context) tuples

        Returns:
            List of DetectionResult objects
        """
        return [
            self.detect(code, context)
            for code, context in code_snippets
        ]


# Example usage and testing
if __name__ == "__main__":
    detector = EnhancedLanguageDetector(min_confidence=0.3)

    # Test Python code
    python_code = '''
def hello_world():
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    hello_world()
'''

    result = detector.detect(python_code)
    print(f"Python test: {result.language} (confidence: {result.confidence:.2f})")
    print(f"  Pattern matches: {result.pattern_matches}")
    print()

    # Test C code
    c_code = '''
#include <stdio.h>
#include <string.h>

void vulnerable(char *input) {
    char buffer[10];
    strcpy(buffer, input);
}

int main() {
    printf("Hello\\n");
    return 0;
}
'''

    result = detector.detect(c_code)
    print(f"C test: {result.language} (confidence: {result.confidence:.2f})")
    print(f"  Pattern matches: {result.pattern_matches}")
    print()

    # Test with context
    c_code_small = '''
char buffer[10];
strcpy(buffer, input);
'''

    context = "The following C program demonstrates a buffer overflow vulnerability:"
    result = detector.detect(c_code_small, context=context)
    print(f"C with context test: {result.language} (confidence: {result.confidence:.2f})")
    print(f"  Context boost: {result.context_boost}")
    print(f"  Pattern matches: {result.pattern_matches}")
    print()

    # Test Bash
    bash_code = '''
#!/bin/bash
echo "Starting script"
sudo apt-get update
if [ $? -eq 0 ]; then
    echo "Success"
fi
'''

    result = detector.detect(bash_code)
    print(f"Bash test: {result.language} (confidence: {result.confidence:.2f})")
    print(f"  Pattern matches: {result.pattern_matches}")
