"""
Code Block Extractor for Mixed PDF Content

Extracts code blocks from text containing mixed prose and code (common in PDFs).
Works in conjunction with enhanced_language_detector.py for accurate language detection.

Key Features:
- Indentation-based code block detection
- Syntax marker recognition (braces, semicolons, function patterns)
- Context hint parsing from surrounding text
- Confidence scoring for extracted blocks
- Handling of PDF extraction artifacts

Author: LLM-Agent-System Team
Created: 2024-11-08
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import re
from enum import Enum


class BlockType(Enum):
    """Type of detected block"""
    CODE = "code"
    PROSE = "prose"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class CodeBlock:
    """Represents an extracted code block from text"""
    content: str
    start_pos: int
    end_pos: int
    context_before: str  # Text before the code block (for language hints)
    context_after: str   # Text after the code block
    confidence: float    # Confidence this is actually code (0.0-1.0)
    detected_hints: List[str]  # Context hints found (e.g., "Python code", "C example")
    block_type: BlockType = BlockType.CODE
    indentation_score: float = 0.0
    syntax_score: float = 0.0

    @property
    def context_combined(self) -> str:
        """Combined context for language detection"""
        return f"{self.context_before}\n{self.context_after}"


@dataclass
class ExtractionResult:
    """Result of code block extraction"""
    code_blocks: List[CodeBlock]
    total_blocks_found: int
    high_confidence_blocks: int  # Blocks with confidence >= 0.7
    medium_confidence_blocks: int  # Blocks with confidence >= 0.5
    low_confidence_blocks: int  # Blocks with confidence < 0.5


class CodeBlockExtractor:
    """
    Extracts code blocks from mixed text (especially PDFs).

    Uses multiple heuristics:
    1. Indentation patterns (consistent spacing)
    2. Syntax markers (braces, semicolons, operators)
    3. Code-like patterns (function calls, variable assignments)
    4. Context hints (surrounding text mentioning languages/code)

    Example:
        extractor = CodeBlockExtractor()
        result = extractor.extract(pdf_text)
        for block in result.code_blocks:
            print(f"Code (confidence {block.confidence:.2f}):")
            print(block.content)
            print(f"Hints: {block.detected_hints}")
    """

    # Context hints that indicate code follows
    LANGUAGE_HINTS = {
        'python': ['python', 'py', 'python code', 'python example', 'python script'],
        'c': ['c code', 'c example', 'c program', 'in c'],
        'cpp': ['c++', 'cpp', 'c++ code', 'c++ example'],
        'rust': ['rust', 'rust code', 'rust example'],
        'ruby': ['ruby', 'ruby code', 'ruby script'],
        'gdscript': ['gdscript', 'godot', 'gd script'],
        'javascript': ['javascript', 'js', 'javascript code', 'node.js'],
        'typescript': ['typescript', 'ts', 'typescript code'],
        'bash': ['bash', 'shell', 'shell script', 'bash script', 'sh'],
        'go': ['go', 'golang', 'go code', 'go example'],
        'java': ['java', 'java code', 'java example'],
    }

    # Generic code indicators
    CODE_INDICATORS = [
        'code', 'example', 'listing', 'snippet', 'program', 'script',
        'function', 'class', 'method', 'implementation', 'algorithm',
        'following code', 'code below', 'above code', 'this code',
        'sample code', 'source code', 'code fragment'
    ]

    # Syntax patterns that strongly suggest code
    SYNTAX_PATTERNS = {
        'function_call': re.compile(r'\w+\s*\([^)]*\)'),  # func(args)
        'assignment': re.compile(r'\w+\s*[=:]\s*[^=]'),  # var = value
        'braces': re.compile(r'[{}]'),  # { or }
        'semicolon': re.compile(r';(?!\s*$)'),  # ; (not at end of line alone)
        'brackets': re.compile(r'[\[\]]'),  # [ or ]
        'operators': re.compile(r'[-+*/%&|^<>=!]=?'),  # Various operators
        'pointer_arrow': re.compile(r'->'),  # -> (C/C++/Rust)
        'double_colon': re.compile(r'::'),  # :: (C++/Rust/etc)
        'type_annotation': re.compile(r':\s*\w+\s*[=,;\)]'),  # : type (TypeScript, Python)
        'control_flow': re.compile(r'\b(if|else|for|while|switch|case|return|break|continue)\b'),
        'keywords': re.compile(r'\b(def|class|function|const|let|var|void|int|char|float|struct|enum)\b'),
    }

    def __init__(self,
                 min_lines: int = 3,
                 min_confidence: float = 0.5,
                 context_window: int = 200,
                 min_indentation_ratio: float = 0.6):
        """
        Initialize the code block extractor.

        Args:
            min_lines: Minimum number of lines to consider as a code block
            min_confidence: Minimum confidence threshold to include a block
            context_window: Characters to capture before/after for context
            min_indentation_ratio: Minimum ratio of indented lines to consider code
        """
        self.min_lines = min_lines
        self.min_confidence = min_confidence
        self.context_window = context_window
        self.min_indentation_ratio = min_indentation_ratio

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract code blocks from text.

        Args:
            text: Input text (potentially mixed prose and code)

        Returns:
            ExtractionResult with extracted code blocks and statistics
        """
        if not text or not text.strip():
            return ExtractionResult([], 0, 0, 0, 0)

        # Step 1: Split into lines and analyze
        lines = text.split('\n')

        # Step 2: Find potential code regions using indentation
        indentation_regions = self._find_indented_regions(lines)

        # Step 3: Score each region and extract blocks
        code_blocks = []
        for start_line, end_line in indentation_regions:
            block = self._extract_block(text, lines, start_line, end_line)
            if block and block.confidence >= self.min_confidence:
                code_blocks.append(block)

        # Step 4: Merge adjacent blocks if they're close
        merged_blocks = self._merge_adjacent_blocks(code_blocks)

        # Step 5: Calculate statistics
        high_conf = sum(1 for b in merged_blocks if b.confidence >= 0.7)
        medium_conf = sum(1 for b in merged_blocks if 0.5 <= b.confidence < 0.7)
        low_conf = sum(1 for b in merged_blocks if b.confidence < 0.5)

        return ExtractionResult(
            code_blocks=merged_blocks,
            total_blocks_found=len(merged_blocks),
            high_confidence_blocks=high_conf,
            medium_confidence_blocks=medium_conf,
            low_confidence_blocks=low_conf
        )

    def _find_indented_regions(self, lines: List[str]) -> List[Tuple[int, int]]:
        """
        Find regions of consistently indented text (likely code).

        Args:
            lines: List of text lines

        Returns:
            List of (start_line, end_line) tuples for potential code regions
        """
        regions = []
        current_region_start = None
        indented_count = 0
        total_count = 0

        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                if current_region_start is not None:
                    total_count += 1
                continue

            # Check if line is indented (starts with spaces/tabs)
            is_indented = line and (line[0] in ' \t')

            if is_indented:
                if current_region_start is None:
                    current_region_start = i
                    indented_count = 1
                    total_count = 1
                else:
                    indented_count += 1
                    total_count += 1
            else:
                # Non-indented line
                if current_region_start is not None:
                    # End of potential code region
                    ratio = indented_count / total_count if total_count > 0 else 0
                    if ratio >= self.min_indentation_ratio and (i - current_region_start) >= self.min_lines:
                        regions.append((current_region_start, i - 1))

                    current_region_start = None
                    indented_count = 0
                    total_count = 0

        # Handle final region
        if current_region_start is not None:
            ratio = indented_count / total_count if total_count > 0 else 0
            if ratio >= self.min_indentation_ratio and (len(lines) - current_region_start) >= self.min_lines:
                regions.append((current_region_start, len(lines) - 1))

        return regions

    def _extract_block(self, full_text: str, lines: List[str],
                       start_line: int, end_line: int) -> Optional[CodeBlock]:
        """
        Extract a code block from a region and score it.

        Args:
            full_text: Complete text (for position calculation)
            lines: List of all lines
            start_line: Starting line index
            end_line: Ending line index

        Returns:
            CodeBlock if valid, None otherwise
        """
        # Extract content
        block_lines = lines[start_line:end_line + 1]
        content = '\n'.join(block_lines)

        # Calculate positions in full text
        chars_before = sum(len(line) + 1 for line in lines[:start_line])  # +1 for \n
        start_pos = chars_before
        end_pos = start_pos + len(content)

        # Extract context
        context_before = full_text[max(0, start_pos - self.context_window):start_pos]
        context_after = full_text[end_pos:min(len(full_text), end_pos + self.context_window)]

        # Score the block
        indentation_score = self._score_indentation(block_lines)
        syntax_score = self._score_syntax(content)
        hints, hint_score = self._find_context_hints(context_before, context_after)

        # Calculate overall confidence
        # Weights: indentation (0.3), syntax (0.5), hints (0.2)
        confidence = (indentation_score * 0.3 + syntax_score * 0.5 + hint_score * 0.2)

        # Boost confidence if multiple strong indicators
        if syntax_score >= 0.7 and indentation_score >= 0.7:
            confidence = min(1.0, confidence * 1.2)

        if hints:
            confidence = min(1.0, confidence * 1.1)

        return CodeBlock(
            content=content,
            start_pos=start_pos,
            end_pos=end_pos,
            context_before=context_before,
            context_after=context_after,
            confidence=confidence,
            detected_hints=hints,
            indentation_score=indentation_score,
            syntax_score=syntax_score
        )

    def _score_indentation(self, lines: List[str]) -> float:
        """
        Score how code-like the indentation pattern is.

        Args:
            lines: Lines to analyze

        Returns:
            Score from 0.0 to 1.0
        """
        if not lines:
            return 0.0

        # Count lines with consistent indentation
        indentation_levels = []
        for line in lines:
            if line.strip():  # Skip empty lines
                # Count leading spaces
                spaces = len(line) - len(line.lstrip(' '))
                indentation_levels.append(spaces)

        if not indentation_levels:
            return 0.0

        # Check for consistent indentation (multiples of 2, 4, or 8)
        consistent_count = 0
        for level in indentation_levels:
            if level % 2 == 0 or level % 4 == 0:
                consistent_count += 1

        consistency_ratio = consistent_count / len(indentation_levels)

        # Check for variation in indentation (nested code)
        unique_levels = len(set(indentation_levels))
        variation_score = min(1.0, unique_levels / 3)  # Good if 3+ levels

        # Combine scores
        return (consistency_ratio * 0.7 + variation_score * 0.3)

    def _score_syntax(self, content: str) -> float:
        """
        Score how much the content looks like code based on syntax patterns.

        Args:
            content: Text to analyze

        Returns:
            Score from 0.0 to 1.0
        """
        if not content:
            return 0.0

        scores = {}
        weights = {
            'function_call': 2.0,  # Strong indicator
            'control_flow': 2.0,   # Strong indicator
            'keywords': 2.0,       # Strong indicator
            'braces': 1.5,
            'assignment': 1.5,
            'operators': 1.0,
            'semicolon': 1.5,
            'brackets': 1.0,
            'pointer_arrow': 1.5,
            'double_colon': 1.5,
            'type_annotation': 1.5,
        }

        total_weight = 0.0
        total_score = 0.0

        for pattern_name, pattern in self.SYNTAX_PATTERNS.items():
            matches = pattern.findall(content)
            if matches:
                weight = weights.get(pattern_name, 1.0)
                # Normalize by content length (per 100 chars)
                match_density = len(matches) / (len(content) / 100 + 1)
                score = min(1.0, match_density * 0.5)  # Cap at 1.0

                scores[pattern_name] = score
                total_weight += weight
                total_score += score * weight

        if total_weight == 0:
            return 0.0

        # Weighted average
        avg_score = total_score / total_weight

        # Boost if multiple different patterns found
        unique_patterns = len([s for s in scores.values() if s > 0.1])
        if unique_patterns >= 3:
            avg_score = min(1.0, avg_score * 1.2)

        return avg_score

    def _find_context_hints(self, context_before: str,
                           context_after: str) -> Tuple[List[str], float]:
        """
        Find language/code hints in surrounding context.

        Args:
            context_before: Text before the code block
            context_after: Text after the code block

        Returns:
            Tuple of (list of detected hints, hint score 0.0-1.0)
        """
        context = (context_before + " " + context_after).lower()
        hints = []

        # Check for specific language mentions
        for language, keywords in self.LANGUAGE_HINTS.items():
            for keyword in keywords:
                if keyword in context:
                    hints.append(f"{language}: {keyword}")

        # Check for generic code indicators
        generic_hints = []
        for indicator in self.CODE_INDICATORS:
            if indicator in context:
                generic_hints.append(indicator)

        # Calculate score
        score = 0.0
        if hints:
            score += 0.8  # Strong language hint
        elif generic_hints:
            score += 0.5  # Generic code mention

        # Add generic hints to result
        hints.extend(generic_hints[:3])  # Limit to 3 generic hints

        return hints, score

    def _merge_adjacent_blocks(self, blocks: List[CodeBlock]) -> List[CodeBlock]:
        """
        Merge code blocks that are close to each other.

        Args:
            blocks: List of code blocks

        Returns:
            List of merged code blocks
        """
        if not blocks:
            return []

        # Sort by start position
        sorted_blocks = sorted(blocks, key=lambda b: b.start_pos)

        merged = []
        current = sorted_blocks[0]

        for next_block in sorted_blocks[1:]:
            # If blocks are close (within 100 chars), merge them
            gap = next_block.start_pos - current.end_pos
            if gap < 100:
                # Merge
                merged_content = current.content + "\n" + next_block.content
                merged_confidence = (current.confidence + next_block.confidence) / 2
                merged_hints = list(set(current.detected_hints + next_block.detected_hints))

                current = CodeBlock(
                    content=merged_content,
                    start_pos=current.start_pos,
                    end_pos=next_block.end_pos,
                    context_before=current.context_before,
                    context_after=next_block.context_after,
                    confidence=merged_confidence,
                    detected_hints=merged_hints,
                    indentation_score=(current.indentation_score + next_block.indentation_score) / 2,
                    syntax_score=(current.syntax_score + next_block.syntax_score) / 2
                )
            else:
                # Keep separate
                merged.append(current)
                current = next_block

        # Add the last block
        merged.append(current)

        return merged

    def extract_with_language_detection(self, text: str, language_detector=None) -> List[Tuple[CodeBlock, str, float]]:
        """
        Extract code blocks and detect language for each.

        Args:
            text: Input text
            language_detector: EnhancedLanguageDetector instance (optional)

        Returns:
            List of (CodeBlock, detected_language, language_confidence) tuples
        """
        result = self.extract(text)

        if language_detector is None:
            # Return blocks without language detection
            return [(block, 'unknown', 0.0) for block in result.code_blocks]

        # Detect language for each block
        results = []
        for block in result.code_blocks:
            detection = language_detector.detect(
                block.content,
                context=block.context_combined
            )
            results.append((block, detection.language, detection.confidence))

        return results


# Example usage
if __name__ == "__main__":
    # Example: Mixed PDF text with prose and code
    sample_text = """
The following C code demonstrates a simple linked list implementation.
This is commonly used in systems programming.

    struct Node {
        int data;
        struct Node* next;
    };

    void insert(struct Node** head, int value) {
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        new_node->data = value;
        new_node->next = *head;
        *head = new_node;
    }

The insert function creates a new node and adds it to the front of the list.
Time complexity is O(1) since we're inserting at the head.

Here's a Python example showing the same concept:

    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None

    def insert(head, value):
        new_node = Node(value)
        new_node.next = head
        return new_node

Python's object-oriented approach makes this more intuitive for beginners.
"""

    print("Code Block Extractor - Example Usage")
    print("=" * 60)

    extractor = CodeBlockExtractor(min_confidence=0.5)
    result = extractor.extract(sample_text)

    print(f"\nExtraction Summary:")
    print(f"  Total blocks found: {result.total_blocks_found}")
    print(f"  High confidence (>=0.7): {result.high_confidence_blocks}")
    print(f"  Medium confidence (>=0.5): {result.medium_confidence_blocks}")
    print(f"  Low confidence (<0.5): {result.low_confidence_blocks}")

    for i, block in enumerate(result.code_blocks, 1):
        print(f"\n{'='*60}")
        print(f"Block {i} - Confidence: {block.confidence:.2f}")
        print(f"Position: {block.start_pos}-{block.end_pos}")
        print(f"Indentation Score: {block.indentation_score:.2f}")
        print(f"Syntax Score: {block.syntax_score:.2f}")
        print(f"Detected Hints: {block.detected_hints}")
        print(f"\nContext Before (last 50 chars):")
        print(f"  ...{block.context_before[-50:]}")
        print(f"\nCode Content:")
        print(block.content)
        print(f"\nContext After (first 50 chars):")
        print(f"  {block.context_after[:50]}...")

    # Example with language detection
    print("\n" + "="*60)
    print("With Language Detection:")
    print("="*60)

    try:
        from enhanced_language_detector import EnhancedLanguageDetector

        detector = EnhancedLanguageDetector()
        results = extractor.extract_with_language_detection(sample_text, detector)

        for i, (block, language, lang_conf) in enumerate(results, 1):
            print(f"\nBlock {i}:")
            print(f"  Extraction confidence: {block.confidence:.2f}")
            print(f"  Detected language: {language} (confidence: {lang_conf:.2f})")
            print(f"  Hints: {block.detected_hints}")
            print(f"  Code preview: {block.content[:100]}...")
    except ImportError:
        print("\nEnhancedLanguageDetector not available. Skipping language detection demo.")
