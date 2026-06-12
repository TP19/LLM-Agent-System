#!/usr/bin/env python3
"""
Adaptive Smart Chunker - Content-Aware Chunking with Safety Margins

Automatically detects content type and chunks appropriately:
- Text: Cut at sentence boundaries
- Code: Cut at function/statement boundaries

Features:
- Config-driven chunk sizes
- Safety margins (10-20% before limits)
- Supports up to 64k token chunks
- Language-specific handling
"""

import re
import ast
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from utilities.token_counter import TokenCounter

# Enhanced PDF language detection
try:
    from utilities.enhanced_language_detector import EnhancedLanguageDetector
    from utilities.code_block_extractor import CodeBlockExtractor
    ENHANCED_DETECTION_AVAILABLE = True
except ImportError:
    ENHANCED_DETECTION_AVAILABLE = False


@dataclass
class ChunkMetadata:
    """Enhanced metadata for chunks"""
    content_type: str  # 'code' | 'text' | 'mixed'
    language: str  # 'python' | 'c' | 'text' | etc.
    is_complete: bool  # Function/paragraph not split
    functions: List[str] = field(default_factory=list)  # Function names
    has_code: bool = False
    chunk_strategy: str = ""  # How it was chunked


@dataclass
class Chunk:
    """Enhanced chunk with metadata"""
    content: str
    chunk_id: int
    token_count: int
    metadata: ChunkMetadata


class AdaptiveSmartChunker:
    """
    Adaptive chunker that handles both code and text intelligently

    Usage:
        # From config
        chunker = AdaptiveSmartChunker.from_config('config/rag_config.yaml')

        # Or manual
        chunker = AdaptiveSmartChunker(
            target_size=1024,
            max_size=4096,
            safety_margin_min=0.80,
            safety_margin_max=0.90
        )

        chunks = chunker.chunk(content, file_path="example.py")
    """

    def __init__(
        self,
        target_size: int = 1024,
        max_size: int = 4096,
        min_size: int = 256,
        overlap_tokens: int = 128,
        safety_margin_min: float = 0.80,
        safety_margin_max: float = 0.90,
        config: Dict = None,
        enable_enhanced_detection: bool = True
    ):
        self.logger = logging.getLogger("AdaptiveSmartChunker")
        self.token_counter = TokenCounter()

        # Chunk size settings
        self.target_size = target_size
        self.max_size = max_size
        self.min_size = min_size
        self.overlap_tokens = overlap_tokens

        # Safety margins (cut at 80-90% of max_size)
        self.safety_margin_min = safety_margin_min
        self.safety_margin_max = safety_margin_max

        # Calculate effective limits
        self.soft_limit_min = int(max_size * safety_margin_min)  # e.g., 3276 for 4096
        self.soft_limit_max = int(max_size * safety_margin_max)  # e.g., 3686 for 4096

        # Config (if provided)
        self.config = config or {}

        # Enhanced language detection for PDFs
        self.enable_enhanced_detection = enable_enhanced_detection and ENHANCED_DETECTION_AVAILABLE
        self.enhanced_detector = None
        self.code_block_extractor = None

        if self.enable_enhanced_detection:
            try:
                self.enhanced_detector = EnhancedLanguageDetector()
                self.code_block_extractor = CodeBlockExtractor(
                    min_confidence=0.5,
                    context_window=200
                )
                self.logger.info("✅ Enhanced PDF language detection enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize enhanced detection: {e}")
                self.enable_enhanced_detection = False

        self.logger.info(
            f"✅ AdaptiveSmartChunker initialized "
            f"(target={target_size}, max={max_size}, "
            f"safety_range={self.soft_limit_min}-{self.soft_limit_max})"
        )

    @classmethod
    def from_config(cls, config_path: str = "config/rag_config.yaml"):
        """Load chunker from config file"""
        config_file = Path(config_path)

        if not config_file.exists():
            # Try relative path
            possible_paths = [
                Path(config_path),
                Path.cwd() / config_path,
                Path(__file__).parent.parent / config_path,
            ]

            for path in possible_paths:
                if path.exists():
                    config_file = path
                    break

        if not config_file.exists():
            logging.warning(f"Config not found: {config_path}, using defaults")
            return cls()

        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)

        chunking_config = full_config.get('chunking', {})

        return cls(
            target_size=chunking_config.get('target_size', 1024),
            max_size=chunking_config.get('max_size', 4096),
            min_size=chunking_config.get('min_size', 256),
            overlap_tokens=chunking_config.get('overlap_tokens', 128),
            safety_margin_min=chunking_config.get('safety_margin_min', 0.80),
            safety_margin_max=chunking_config.get('safety_margin_max', 0.90),
            config=chunking_config
        )

    def chunk(self, content: str, file_path: str = "", is_pdf: bool = False) -> List[Chunk]:
        """
        Main chunking method - detects content type and chunks appropriately

        Args:
            content: Text or code to chunk
            file_path: Optional file path for language detection
            is_pdf: True if content is from a PDF (enables enhanced detection)

        Returns:
            List of chunks with metadata
        """
        if not content or not content.strip():
            return []

        # Auto-detect PDF from file extension
        if not is_pdf and file_path:
            is_pdf = Path(file_path).suffix.lower() == '.pdf'

        # Use enhanced PDF chunking if available and needed
        if is_pdf and self.enable_enhanced_detection:
            self.logger.info(f"Using enhanced PDF chunking for: {len(content)} chars")
            return self._chunk_pdf_content(content, file_path)

        # Standard chunking
        # Detect content type and language
        content_type, language = self._detect_content_type(content, file_path)

        self.logger.info(f"Chunking {language} ({content_type}): {len(content)} chars")

        # Route to appropriate strategy
        if content_type == 'code':
            chunks = self._chunk_code(content, language)
        else:
            chunks = self._chunk_text(content)

        self.logger.info(f"📦 Created {len(chunks)} chunks")

        return chunks

    # ========================================================================
    # Content Type Detection
    # ========================================================================

    def _detect_content_type(self, content: str, file_path: str = "") -> Tuple[str, str]:
        """
        Detect if content is code or text

        Returns:
            (content_type, language) tuple
            e.g., ('code', 'python') or ('text', 'text')
        """
        # Check file extension first
        if file_path:
            ext = Path(file_path).suffix.lower()
            language_map = {
                '.py': ('code', 'python'),
                '.c': ('code', 'c'),
                '.cpp': ('code', 'cpp'),
                '.cc': ('code', 'cpp'),
                '.h': ('code', 'c'),
                '.hpp': ('code', 'cpp'),
                '.js': ('code', 'javascript'),
                '.ts': ('code', 'typescript'),
                '.java': ('code', 'java'),
                '.go': ('code', 'go'),
                '.rs': ('code', 'rust'),
                '.rb': ('code', 'ruby'),
                '.gd': ('code', 'gdscript'),
                '.sh': ('code', 'bash'),
                '.bash': ('code', 'bash'),
                '.md': ('text', 'markdown'),
                '.txt': ('text', 'text'),
                '.rst': ('text', 'text'),
            }

            if ext in language_map:
                return language_map[ext]

        # Try enhanced detection if available
        if self.enable_enhanced_detection and self.enhanced_detector:
            try:
                detection = self.enhanced_detector.detect(content[:5000])  # Check first 5000 chars
                if detection.confidence >= 0.5 and detection.language != 'generic':
                    self.logger.debug(
                        f"Enhanced detection: {detection.language} "
                        f"(confidence: {detection.confidence:.2f})"
                    )
                    return ('code', detection.language)
                elif detection.confidence >= 0.3:
                    # Lower confidence, but still code-like
                    return ('code', detection.language)
            except Exception as e:
                self.logger.debug(f"Enhanced detection failed: {e}")

        # Fallback: Basic content-based detection
        code_indicators = {
            'python': [r'def \w+\(', r'import \w+', r'class \w+:', r'if __name__'],
            'c': [r'#include\s*<', r'int main\(', r'printf\(', r'struct \w+'],
            'cpp': [r'#include\s*<', r'namespace \w+', r'std::', r'class \w+'],
            'javascript': [r'function \w+\(', r'const \w+ =', r'=>', r'import .+ from'],
            'java': [r'public class', r'import java\.', r'public static void'],
        }

        # Count matches
        scores = {}
        for lang, patterns in code_indicators.items():
            score = sum(len(re.findall(p, content[:2000])) for p in patterns)  # Check first 2000 chars
            scores[lang] = score

        # If any language has strong signals, it's code
        if scores and max(scores.values()) >= 3:
            language = max(scores, key=scores.get)
            return ('code', language)

        # Check code-like features
        has_braces = content.count('{') > 5 or content.count('}') > 5
        has_semicolons = content.count(';') > 10
        has_indentation = len(re.findall(r'^\s{4,}', content, re.MULTILINE)) > 5

        if has_braces or (has_semicolons and has_indentation):
            return ('code', 'generic')

        # Default to text
        return ('text', 'text')

    # ========================================================================
    # PDF Content Chunking (Enhanced Detection)
    # ========================================================================

    def _chunk_pdf_content(self, content: str, file_path: str = "") -> List[Chunk]:
        """
        Chunk PDF content using enhanced code block extraction and language detection.

        Strategy:
        1. Extract code blocks using CodeBlockExtractor
        2. Detect language for each code block with EnhancedLanguageDetector
        3. Create chunks with accurate language metadata
        4. Handle mixed prose/code appropriately

        Args:
            content: PDF text content (mixed prose and code)
            file_path: Optional file path for context

        Returns:
            List of chunks with enhanced metadata
        """
        if not self.code_block_extractor or not self.enhanced_detector:
            self.logger.warning("Enhanced detection not available, falling back to standard chunking")
            content_type, language = self._detect_content_type(content, file_path)
            if content_type == 'code':
                return self._chunk_code(content, language)
            else:
                return self._chunk_text(content)

        chunks = []
        chunk_id = 0

        # Extract code blocks
        self.logger.debug("Extracting code blocks from PDF content...")
        extraction_result = self.code_block_extractor.extract(content)

        self.logger.info(
            f"Found {extraction_result.total_blocks_found} code blocks "
            f"(high: {extraction_result.high_confidence_blocks}, "
            f"medium: {extraction_result.medium_confidence_blocks}, "
            f"low: {extraction_result.low_confidence_blocks})"
        )

        if not extraction_result.code_blocks:
            # No code blocks found, treat as text
            self.logger.info("No code blocks found, treating as text")
            return self._chunk_text(content)

        # Process each code block
        last_end_pos = 0
        for code_block in extraction_result.code_blocks:
            # Handle prose before this code block
            prose_before = content[last_end_pos:code_block.start_pos].strip()
            if prose_before and len(prose_before) > 100:  # Min prose length
                # Chunk the prose section
                prose_chunks = self._chunk_text(prose_before)
                for prose_chunk in prose_chunks:
                    prose_chunk.chunk_id = chunk_id
                    chunks.append(prose_chunk)
                    chunk_id += 1

            # Detect language for code block
            detection = self.enhanced_detector.detect(
                code_block.content,
                context=code_block.context_combined
            )

            self.logger.debug(
                f"Code block {chunk_id}: {detection.language} "
                f"(confidence: {detection.confidence:.2f}, "
                f"extraction confidence: {code_block.confidence:.2f})"
            )

            # Create code chunk with detected language
            token_count = self.token_counter.count_tokens(code_block.content)

            # Check if code block is too large
            if token_count > self.soft_limit_max:
                # Split large code block
                self.logger.debug(f"Code block too large ({token_count} tokens), splitting...")
                sub_chunks = self._split_large_code_block(
                    code_block.content,
                    detection.language,
                    detection.confidence
                )
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = chunk_id
                    chunks.append(sub_chunk)
                    chunk_id += 1
            else:
                # Code block fits, create single chunk
                metadata = ChunkMetadata(
                    content_type='code',
                    language=detection.language,
                    is_complete=True,
                    has_code=True,
                    chunk_strategy=f'pdf_extraction_confidence_{code_block.confidence:.2f}'
                )

                chunks.append(Chunk(
                    content=code_block.content,
                    chunk_id=chunk_id,
                    token_count=token_count,
                    metadata=metadata
                ))
                chunk_id += 1

            last_end_pos = code_block.end_pos

        # Handle remaining prose after last code block
        prose_after = content[last_end_pos:].strip()
        if prose_after and len(prose_after) > 100:
            prose_chunks = self._chunk_text(prose_after)
            for prose_chunk in prose_chunks:
                prose_chunk.chunk_id = chunk_id
                chunks.append(prose_chunk)
                chunk_id += 1

        self.logger.info(f"📦 Created {len(chunks)} chunks from PDF (enhanced detection)")
        return chunks

    def _split_large_code_block(self, code_content: str, language: str, confidence: float) -> List[Chunk]:
        """
        Split a large code block that exceeds soft_limit_max.

        Args:
            code_content: Code content to split
            language: Detected language
            confidence: Language detection confidence

        Returns:
            List of smaller code chunks
        """
        chunks = []
        lines = code_content.split('\n')
        current_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = self.token_counter.count_tokens(line)

            if current_tokens + line_tokens > self.soft_limit_max and current_lines:
                # Create chunk
                chunk_text = '\n'.join(current_lines)
                metadata = ChunkMetadata(
                    content_type='code',
                    language=language,
                    is_complete=False,  # Split code block
                    has_code=True,
                    chunk_strategy=f'pdf_split_confidence_{confidence:.2f}'
                )

                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_id=0,  # Will be set by caller
                    token_count=current_tokens,
                    metadata=metadata
                ))

                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        # Final chunk
        if current_lines:
            chunk_text = '\n'.join(current_lines)
            metadata = ChunkMetadata(
                content_type='code',
                language=language,
                is_complete=False,
                has_code=True,
                chunk_strategy=f'pdf_split_confidence_{confidence:.2f}'
            )

            chunks.append(Chunk(
                content=chunk_text,
                chunk_id=0,
                token_count=current_tokens,
                metadata=metadata
            ))

        return chunks

    # ========================================================================
    # Text Chunking (Sentence Boundaries)
    # ========================================================================

    def _chunk_text(self, content: str) -> List[Chunk]:
        """
        Chunk text at sentence boundaries with safety margins

        Strategy:
        1. Build chunks up to soft_limit_max
        2. Find nearest sentence boundary
        3. Apply overlap
        4. Ensure minimum size
        """
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_tokens = 0
        chunk_id = 0

        for line in lines:
            line_tokens = self.token_counter.count_tokens(line)

            # Check if adding this line would exceed soft limit
            if current_tokens + line_tokens > self.soft_limit_max and current_chunk:
                # Find sentence boundary in current chunk
                chunk_text = '\n'.join(current_chunk)
                cut_point = self._find_sentence_boundary(chunk_text, self.soft_limit_min, self.soft_limit_max)

                if cut_point > 0:
                    # Split at sentence boundary
                    final_text = chunk_text[:cut_point].strip()
                    remainder = chunk_text[cut_point:].strip()

                    if final_text:
                        chunks.append(self._create_text_chunk(final_text, chunk_id))
                        chunk_id += 1

                    # Start new chunk with remainder
                    current_chunk = [remainder, line] if remainder else [line]
                    current_tokens = self.token_counter.count_tokens('\n'.join(current_chunk))
                else:
                    # No good boundary, use current chunk as-is
                    if chunk_text:
                        chunks.append(self._create_text_chunk(chunk_text, chunk_id))
                        chunk_id += 1
                    current_chunk = [line]
                    current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if chunk_text.strip():
                chunks.append(self._create_text_chunk(chunk_text, chunk_id))

        return chunks

    def _find_sentence_boundary(self, text: str, min_pos_tokens: int, max_pos_tokens: int) -> int:
        """
        Find best sentence boundary between min and max positions

        Returns character position, or -1 if no good boundary found
        """
        # Convert token positions to approximate character positions
        # Rough estimate: 1 token ≈ 4 characters
        min_pos = min_pos_tokens * 4
        max_pos = max_pos_tokens * 4

        # Clamp to actual text length
        max_pos = min(max_pos, len(text))
        min_pos = min(min_pos, max_pos)

        if max_pos <= 0:
            return -1

        # Search region
        search_text = text[min_pos:max_pos]

        # Look for sentence boundaries (prioritize later ones)
        sentence_patterns = [
            r'\.\s+[A-Z]',  # Period followed by space and capital letter
            r'\.\n',        # Period at end of line
            r'!\s+',        # Exclamation
            r'\?\s+',       # Question mark
            r'\.\s+',       # Period with space
        ]

        best_pos = -1
        for pattern in sentence_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Take the last match (closest to max_pos)
                match = matches[-1]
                best_pos = min_pos + match.end()
                break

        # Fallback: paragraph break
        if best_pos == -1:
            para_break = search_text.rfind('\n\n')
            if para_break != -1:
                best_pos = min_pos + para_break + 2

        return best_pos

    def _create_text_chunk(self, text: str, chunk_id: int) -> Chunk:
        """Create chunk with text metadata"""
        token_count = self.token_counter.count_tokens(text)

        metadata = ChunkMetadata(
            content_type='text',
            language='text',
            is_complete=True,  # Sentences are complete
            has_code=False,
            chunk_strategy='sentence_boundary'
        )

        return Chunk(
            content=text,
            chunk_id=chunk_id,
            token_count=token_count,
            metadata=metadata
        )

    # ========================================================================
    # Code Chunking (Function/Statement Boundaries)
    # ========================================================================

    def _chunk_code(self, content: str, language: str) -> List[Chunk]:
        """
        Chunk code at function/statement boundaries

        Strategy:
        1. Try to keep functions together
        2. If function too large, split at safe boundaries
        3. Never cut mid-variable or mid-expression
        """
        if language == 'python':
            return self._chunk_python_code(content)
        elif language in ['c', 'cpp']:
            return self._chunk_c_like_code(content, language)
        else:
            return self._chunk_generic_code(content, language)

    def _chunk_python_code(self, content: str) -> List[Chunk]:
        """Chunk Python code using AST for perfect boundaries"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Invalid Python, fall back to generic
            self.logger.warning("Python syntax error, using generic chunking")
            return self._chunk_generic_code(content, 'python')

        chunks = []
        lines = content.split('\n')
        chunk_id = 0

        # Extract imports (add to all chunks for context)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'lineno'):
                    imports.append(lines[node.lineno - 1])

        import_text = '\n'.join(imports) if imports else ""

        # Process each top-level definition
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Extract function/class
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1

                func_lines = lines[start_line:end_line]
                func_text = '\n'.join(func_lines)

                # Add imports if not already present
                if import_text and import_text not in func_text:
                    full_text = import_text + '\n\n' + func_text
                else:
                    full_text = func_text

                func_tokens = self.token_counter.count_tokens(full_text)

                # Check if function fits within limits
                if func_tokens <= self.soft_limit_max:
                    # Function fits, keep it together
                    metadata = ChunkMetadata(
                        content_type='code',
                        language='python',
                        is_complete=True,
                        functions=[node.name],
                        has_code=True,
                        chunk_strategy='ast_function'
                    )

                    chunks.append(Chunk(
                        content=full_text,
                        chunk_id=chunk_id,
                        token_count=func_tokens,
                        metadata=metadata
                    ))
                    chunk_id += 1
                else:
                    # Function too large, split at safe boundaries
                    sub_chunks = self._split_large_function(full_text, 'python', node.name)
                    for sub_chunk in sub_chunks:
                        sub_chunk.chunk_id = chunk_id
                        chunks.append(sub_chunk)
                        chunk_id += 1

        # Handle module-level code (not in functions/classes)
        # TODO: Implement if needed

        return chunks if chunks else self._chunk_generic_code(content, 'python')

    def _chunk_c_like_code(self, content: str, language: str) -> List[Chunk]:
        """Chunk C/C++ code using brace counting"""
        chunks = []
        lines = content.split('\n')
        chunk_id = 0

        current_func = []
        brace_depth = 0
        in_function = False
        func_name = "unknown"

        for line in lines:
            stripped = line.strip()

            # Detect function start (simplified)
            if not in_function and ('{' in line and ('(' in line or stripped.endswith('{'))):
                # Possible function start
                in_function = True
                # Try to extract function name
                match = re.search(r'(\w+)\s*\(', line)
                if match:
                    func_name = match.group(1)

            if in_function:
                current_func.append(line)
                brace_depth += line.count('{') - line.count('}')

                # Function complete
                if brace_depth == 0 and '{' in ''.join(current_func):
                    func_text = '\n'.join(current_func)
                    func_tokens = self.token_counter.count_tokens(func_text)

                    if func_tokens <= self.soft_limit_max:
                        metadata = ChunkMetadata(
                            content_type='code',
                            language=language,
                            is_complete=True,
                            functions=[func_name],
                            has_code=True,
                            chunk_strategy='brace_counting'
                        )

                        chunks.append(Chunk(
                            content=func_text,
                            chunk_id=chunk_id,
                            token_count=func_tokens,
                            metadata=metadata
                        ))
                        chunk_id += 1
                    else:
                        # Function too large, split it
                        sub_chunks = self._split_large_function(func_text, language, func_name)
                        for sub_chunk in sub_chunks:
                            sub_chunk.chunk_id = chunk_id
                            chunks.append(sub_chunk)
                            chunk_id += 1

                    # Reset
                    current_func = []
                    in_function = False
                    func_name = "unknown"

        return chunks if chunks else self._chunk_generic_code(content, language)

    def _chunk_generic_code(self, content: str, language: str) -> List[Chunk]:
        """
        Generic code chunking for unknown languages

        Strategy: Split at blank lines and safe patterns
        """
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_tokens = 0
        chunk_id = 0

        for i, line in enumerate(lines):
            line_tokens = self.token_counter.count_tokens(line)

            # Check if we should cut here
            should_cut = False
            if current_tokens + line_tokens > self.soft_limit_max:
                # Look for safe cut point in recent lines
                safe_cut = self._find_safe_code_cut(current_chunk[-20:] if len(current_chunk) > 20 else current_chunk)
                if safe_cut is not None:
                    should_cut = True

            if should_cut and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                if chunk_text.strip():
                    metadata = ChunkMetadata(
                        content_type='code',
                        language=language,
                        is_complete=False,  # Generic split, not sure if complete
                        has_code=True,
                        chunk_strategy='generic_safe_cut'
                    )

                    chunks.append(Chunk(
                        content=chunk_text,
                        chunk_id=chunk_id,
                        token_count=current_tokens,
                        metadata=metadata
                    ))
                    chunk_id += 1

                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

        # Final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if chunk_text.strip():
                metadata = ChunkMetadata(
                    content_type='code',
                    language=language,
                    is_complete=False,
                    has_code=True,
                    chunk_strategy='generic_safe_cut'
                )

                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    token_count=current_tokens,
                    metadata=metadata
                ))

        return chunks

    def _find_safe_code_cut(self, recent_lines: List[str]) -> Optional[int]:
        """
        Find a safe place to cut code in recent lines

        Returns index in recent_lines, or None if no safe cut found
        """
        # Safe patterns (from config)
        safe_patterns = [
            r'\}\s*$',      # Closing brace
            r';\s*$',       # Statement end
            r'\]\s*$',      # Closing bracket
            r'^\s*$',       # Blank line
        ]

        # Search from end backwards
        for i in range(len(recent_lines) - 1, -1, -1):
            line = recent_lines[i]
            for pattern in safe_patterns:
                if re.search(pattern, line):
                    return i + 1  # Cut after this line

        return None

    def _split_large_function(self, func_text: str, language: str, func_name: str) -> List[Chunk]:
        """
        Split a function that's too large

        Strategy: Find logical boundaries within the function
        """
        self.logger.warning(f"Splitting large function: {func_name} ({self.token_counter.count_tokens(func_text)} tokens)")

        # For now, use generic splitting
        # TODO: Implement smarter splitting (nested blocks, if/else boundaries, etc.)
        lines = func_text.split('\n')
        chunks = []
        current_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = self.token_counter.count_tokens(line)

            if current_tokens + line_tokens > self.soft_limit_max and current_lines:
                # Create chunk
                chunk_text = '\n'.join(current_lines)
                metadata = ChunkMetadata(
                    content_type='code',
                    language=language,
                    is_complete=False,  # Split function
                    functions=[func_name],
                    has_code=True,
                    chunk_strategy='large_function_split'
                )

                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_id=0,  # Will be set by caller
                    token_count=current_tokens,
                    metadata=metadata
                ))

                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        # Final chunk
        if current_lines:
            chunk_text = '\n'.join(current_lines)
            metadata = ChunkMetadata(
                content_type='code',
                language=language,
                is_complete=False,
                functions=[func_name],
                has_code=True,
                chunk_strategy='large_function_split'
            )

            chunks.append(Chunk(
                content=chunk_text,
                chunk_id=0,
                token_count=current_tokens,
                metadata=metadata
            ))

        return chunks
