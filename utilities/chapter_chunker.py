#!/usr/bin/env python3
"""
Chapter-Aware Chunker Enhanced Summarizer

Chunks documents while respecting chapter boundaries for
parallel chapter-by-chapter summarization.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from utilities.semantic_chunker import SemanticChunker
from utilities.token_counter import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class ChapterChunks:
    """A chapter with its chunks and metadata"""
    chapter_num: int
    chapter_title: str
    chunks: List[str]
    token_count: int
    start_pos: int = 0
    end_pos: int = 0
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def __repr__(self):
        return f"ChapterChunks(num={self.chapter_num}, title='{self.chapter_title[:30]}...', chunks={self.chunk_count}, tokens={self.token_count})"


@dataclass
class DocumentStructure:
    """Analyzed document structure"""
    doc_type: str  # 'book', 'article', 'documentation', 'other'
    chapters: List[ChapterChunks]
    total_tokens: int
    has_toc: bool = False
    title: Optional[str] = None
    author: Optional[str] = None


class ChapterAwareChunker:
    """
    Chunks documents while preserving chapter structure.

    Features:
    - Detects chapter boundaries using multiple patterns
    - Chunks within chapters for parallel processing
    - Preserves chapter metadata (title, position)
    - Falls back to section-based chunking for non-book content

    Usage:
        chunker = ChapterAwareChunker()
        structure = chunker.chunk_document(content, "book.pdf")

        for chapter in structure.chapters:
            print(f"Chapter {chapter.chapter_num}: {chapter.chapter_title}")
            print(f"  {chapter.chunk_count} chunks, {chapter.token_count} tokens")
    """

    # Chapter detection patterns (order matters - more specific first)
    # These require explicit "Chapter"/"Part"/"Section" keywords for reliability.
    # Bare numbered headings (e.g. "1 Introduction") are handled separately
    # with stricter validation (see _find_bare_numbered_headings).
    CHAPTER_PATTERNS = [
        # Standard chapter markers - title on same line
        # Note: (\d[\d ]*\d|\d+) handles OCR artifacts like "1 3" for "13"
        r'^Chapter\s+(\d[\d ]*\d|\d+)[:\.\s]+(.+?)(?:\n|$)',
        r'^CHAPTER\s+(\d[\d ]*\d|\d+)[:\.\s]+(.+?)(?:\n|$)',
        r'^Chapter\s+([IVXLC]+)[:\.\s]+(.+?)(?:\n|$)',  # Roman numerals

        # Standard chapter markers - title on NEXT line (common in PDFs)
        # e.g., "Chapter 1\nWeapons of Influence"
        r'^Chapter\s+(\d[\d ]*\d|\d+)\s*\n([A-Z][^\n]{3,80})(?:\n|$)',
        r'^CHAPTER\s+(\d[\d ]*\d|\d+)\s*\n([A-Z][^\n]{3,80})(?:\n|$)',

        # Part/Section markers
        r'^Part\s+(\d+)[:\s]+(.+?)(?:\n|$)',
        r'^PART\s+(\d+)[:\s]+(.+?)(?:\n|$)',
        r'^Section\s+(\d+)[:\s]+(.+?)(?:\n|$)',

        # CTF/Technical patterns
        r'^(0x[0-9a-fA-F]+)\s*[-:]?\s*(.+?)(?:\n|$)',  # Hex chapters

        # Book-specific
        r'^(Prologue|Epilogue|Introduction|Preface|Foreword|Conclusion)\s*[-:]?\s*(.*)(?:\n|$)',

        # Russian chapter markers
        r'^ГЛАВА\s+(\d[\d ]*\d|\d+)[:\.\s]+(.+?)(?:\n|$)',
        r'^Глава\s+(\d[\d ]*\d|\d+)[:\.\s]+(.+?)(?:\n|$)',
        r'^ГЛАВА\s+(\d[\d ]*\d|\d+)\s*$',
        r'^ЧАСТЬ\s+(\d+)[:\s]+(.+?)(?:\n|$)',
        r'^Часть\s+(\d+)[:\s]+(.+?)(?:\n|$)',
        r'^(Пролог|Эпилог|Предисловие|Вступление|Введение|Заключение|Послесловие|Хронология)\s*[-:]?\s*(.*)(?:\n|$)',
    ]

    # Special chapter names that don't have numbers
    SPECIAL_CHAPTERS = [
        'prologue', 'epilogue', 'introduction', 'preface',
        'foreword', 'conclusion', 'afterword', 'appendix',
        # Russian
        'пролог', 'эпилог', 'введение', 'вступление',
        'предисловие', 'послесловие', 'заключение', 'приложение',
        'хронология', 'оглавление', 'примечания',
    ]

    # Patterns for introductory chapters that may appear before body_start
    INTRO_PATTERNS = [
        r'(?:^|\n)\s*(Introduction)\s*\n',
        r'(?:^|\n)\s*(Foreword)\s*\n',
        r'(?:^|\n)\s*(Preface)\s*\n',
        r"(?:^|\n)\s*(Author'?s?\s+Note)\s*\n",
        r'(?:^|\n)\s*(Prologue)\s*\n',
        # Russian
        r'(?:^|\n)\s*(Предисловие)\s*\n',
        r'(?:^|\n)\s*(Вступление)\s*\n',
        r'(?:^|\n)\s*(Введение)\s*\n',
        r'(?:^|\n)\s*(Пролог)\s*\n',
        r'(?:^|\n)\s*(Хронология)\s*\n',
    ]

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        min_chapter_size: int = 500,  # Minimum tokens for a chapter
    ):
        """
        Initialize chapter-aware chunker.

        Args:
            chunk_size: Target tokens per chunk within chapters
            chunk_overlap: Overlap between chunks
            min_chapter_size: Minimum tokens to consider as separate chapter
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chapter_size = min_chapter_size

        # Initialize token counter and base chunker
        self.token_counter = TokenCounter()
        self.base_chunker = SemanticChunker(token_counter=self.token_counter)

        logger.info(f"ChapterAwareChunker initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def chunk_document(self, content: str, file_path: str = "") -> DocumentStructure:
        """
        Chunk a document respecting chapter boundaries.

        Args:
            content: Full document text
            file_path: Optional file path for context

        Returns:
            DocumentStructure with chapters and metadata
        """
        logger.info(f"Chunking document: {file_path or 'unnamed'}")

        # Normalize PDF artifacts: form feeds (\x0c) are page breaks that
        # interfere with ^ line-start matching in chapter detection patterns.
        content = content.replace('\x0c', '\n')

        # Normalize OCR artifacts in chapter markers:
        # 1. Spaced-out letters: "C H A P T E R" -> "CHAPTER"
        content = re.sub(r'C\s+H\s+A\s+P\s+T\s+E\s+R', 'CHAPTER', content)
        # 2. OCR digit->letter substitutions in chapter numbers (I=1, O=0, t=1)
        #    "CHAPTER IO" -> "CHAPTER 10", "CHAPTER It" -> "CHAPTER 11"
        def _fix_ocr_chapter_num(m):
            prefix = m.group(1)
            num_str = m.group(2)
            fixed = num_str.replace('I', '1').replace('O', '0').replace('t', '1').replace('l', '1')
            # Only apply if the result is a valid number
            if fixed.replace(' ', '').isdigit():
                return prefix + fixed
            return m.group(0)
        content = re.sub(r'(CHAPTER\s+)([IOtl\d][\d IOtl]*)', _fix_ocr_chapter_num, content)

        # Detect document type
        doc_type = self._detect_document_type(content, file_path)
        logger.info(f"Detected document type: {doc_type}")

        # Extract title early so we can use it for single-document case
        doc_title = self._extract_title(content)
        if not doc_title and file_path:
            from pathlib import Path
            doc_title = Path(file_path).stem.replace('_', ' ').replace('-', ' ').title()

        # Find chapter boundaries
        chapter_markers = self._find_chapters(content)
        logger.info(f"Found {len(chapter_markers)} chapter markers")

        if not chapter_markers:
            # No chapters found - use document title as the chapter title
            logger.info("No chapters found, creating single-chapter structure")
            chapter_title = doc_title or "Full Document"
            chapter_markers = [(0, chapter_title, 0)]

        # Split content by chapters and chunk each
        chapters = self._split_and_chunk_chapters(content, chapter_markers)

        # Calculate total tokens
        total_tokens = sum(ch.token_count for ch in chapters)

        structure = DocumentStructure(
            doc_type=doc_type,
            chapters=chapters,
            total_tokens=total_tokens,
            title=doc_title,
        )

        logger.info(f"Document structure: {len(chapters)} chapters, {total_tokens} tokens")
        return structure

    def _detect_document_type(self, content: str, file_path: str) -> str:
        """Detect document type based on content and file path"""
        content_lower = content.lower()[:3000]

        # Check for book indicators
        book_indicators = ['chapter ', 'prologue', 'epilogue', 'table of contents']
        if any(ind in content_lower for ind in book_indicators):
            return 'book'

        # Academic paper
        academic_indicators = ['abstract', 'methodology', 'references', 'doi:']
        if any(ind in content_lower for ind in academic_indicators):
            return 'article'

        # Documentation
        if Path(file_path).suffix.lower() in ['.md', '.rst']:
            return 'documentation'

        return 'other'

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Clean up chapter title - remove mojibake, control chars, extra whitespace."""
        import unicodedata
        # Remove control characters
        title = ''.join(c for c in title if unicodedata.category(c)[0] != 'C')
        # Replace common mojibake sequences (ï¿½ = replacement char)
        title = title.replace('\ufffd', '').replace('ï¿½', '')
        # Normalize unicode dashes to regular dash
        title = title.replace('\u2013', '-').replace('\u2014', '-')
        # Strip leading/trailing punctuation and whitespace
        title = title.strip(' .\t\n\r-–—')
        # Collapse multiple spaces
        title = ' '.join(title.split())
        return title

    @staticmethod
    def _is_valid_chapter_title(title: str) -> bool:
        """Validate that a detected title looks like an actual chapter heading,
        not a sentence fragment from PDF extraction."""
        if not title or len(title) < 2:
            return False
        # Reject single characters or very short fragments
        if len(title.strip()) <= 3:
            return False
        # Reject if it starts with a lowercase letter (sentence continuation)
        stripped = title.strip()
        if stripped and stripped[0].islower():
            return False
        # Reject if it looks like a sentence fragment (ends with common conjunctions/prepositions)
        fragment_endings = [' and', ' the', ' of', ' in', ' to', ' a', ' is', ' are',
                           ' was', ' were', ' that', ' which', ' with', ' for', ' on',
                           ' at', ' from', ' by', ' or', ' an', ' as', ' but']
        title_lower = title.lower().rstrip(' .,;:')
        if any(title_lower.endswith(ending) for ending in fragment_endings):
            return False
        # Reject if it starts with words suggesting mid-sentence fragment
        # Note: We already reject lowercase starts above, so these catch cases like
        # "Also consider..." or "Is this really..." that start capitalized but aren't titles.
        # We EXCLUDE common title starters: The, A, An, Of, In, On, For, With, From, By
        # since "The Fallacy of Supply", "A Brief History", "On War" are valid titles.
        fragment_starts = ['s ', 's.', 'is ', 'are ', 'was ', 'were ',
                          'that ', 'which ', 'have ', 'had ', 'not ',
                          'also ', 'even ', 'than ']
        if any(stripped.lower().startswith(s) for s in fragment_starts):
            return False
        # Reject overly long titles (real chapter titles are usually < 80 chars)
        if len(title) > 100:
            return False
        return True

    def _find_bare_numbered_headings(self, content: str) -> List[Tuple[int, str, int]]:
        """Find numbered headings like '1. Introduction' or '1\nAtoms in Motion'
        with strict validation to avoid matching numbered content in paragraphs.

        Requirements for a valid bare-numbered heading:
        - Must be preceded by a blank line (or start of text)
        - Title must pass _is_valid_chapter_title validation
        - Must be in first 75% of document (avoids matching endnotes/footnotes)
        """
        results = []
        # Only search first 75% of document to avoid matching endnote entries
        search_limit = int(len(content) * 0.75)
        search_content = content[:search_limit]

        # Pattern 1: "N. Title" with preceding blank line
        # Matches: "1. Introduction" but not "94 percent of those asked"
        for match in re.finditer(
            r'(?:^|\n\n)(\d{1,3})\.\s+([A-Z][A-Za-z\s\'\-\u2019,()]{5,80})(?:\n|$)',
            search_content, re.MULTILINE
        ):
            num = int(match.group(1))
            title = self._normalize_title(match.group(2))
            # Adjust position to the actual number, not the preceding newlines
            pos = match.start()
            if content[pos] == '\n':
                pos = content.index(match.group(1), pos)
            if self._is_valid_chapter_title(title):
                results.append((num, title, pos))

        # Pattern 2: Number on own line followed by title on next line (Feynman style)
        # e.g., "1\nAtoms in Motion"
        for match in re.finditer(
            r'(?:^|\n\n)(\d{1,3})\n([A-Z][A-Za-z\s\'\-\u2019,()]{5,80})(?:\n|$)',
            search_content, re.MULTILINE
        ):
            num = int(match.group(1))
            title = self._normalize_title(match.group(2))
            pos = match.start()
            if content[pos] == '\n':
                pos = content.index(match.group(1), pos)
            if self._is_valid_chapter_title(title):
                results.append((num, title, pos))

        # Pattern 3: Bare "N Title" (most dangerous - only accept if very clean)
        # Requires blank line before AND title must be 2+ proper words
        for match in re.finditer(
            r'\n\n(\d{1,2})\s+([A-Z][a-z]+(?:\s+[A-Za-z][a-z]*){1,8})(?:\n|$)',
            search_content, re.MULTILINE
        ):
            num = int(match.group(1))
            title = self._normalize_title(match.group(2))
            pos = match.start() + 2  # skip the \n\n
            if self._is_valid_chapter_title(title):
                results.append((num, title, pos))

        return results

    def _find_chapters_via_toc(self, content: str) -> List[Tuple[int, str, int]]:
        """Extract chapter titles from a Table of Contents, then find them in the body.

        Many books have numbered chapter titles in the TOC but only the title
        text (without numbers) in the actual body. This method:
        1. Finds a dense cluster of numbered entries in the first 5% (the TOC)
        2. Extracts the chapter titles
        3. Searches for those exact titles later in the document body
        """
        # Step 1: Find TOC entries - numbered lines densely packed at the start
        # Search first 5% for TOC entries, but body_start will be computed
        # based on where the last TOC entry actually is.
        toc_search_limit = len(content) // 20
        toc_region = content[:toc_search_limit]
        toc_lines = toc_region.split('\n')

        # Patterns for TOC entries: "1. Title", "1: Title", "1 Title"
        # Also matches "Chapter N. Title" and "Chapter N: Title"
        toc_entry_re = re.compile(
            r'^\s*(?:Chapter\s+)?(\d{1,3})[.:\s]\s*(["\'\u201c]?[A-Z][^\n]{3,80})$'
        )

        # Also handle TOC format where number is on one line and title on next:
        # "1\nWeapons of Influence\n\n1\n\n2\nReciprociation..."
        toc_split_re = re.compile(r'^\s*(\d{1,3})\s*$')

        toc_entries = []
        for i, line in enumerate(toc_lines):
            # Try same-line format first
            m = toc_entry_re.match(line.strip())
            if m:
                num = int(m.group(1))
                title = self._normalize_title(m.group(2))
                if self._is_valid_chapter_title(title):
                    toc_entries.append((num, title))
                continue

            # Try split-line format: number alone, title on next line
            m = toc_split_re.match(line.strip())
            if m and i + 1 < len(toc_lines):
                next_line = toc_lines[i + 1].strip()
                if next_line and next_line[0].isupper() and len(next_line) >= 4:
                    num = int(m.group(1))
                    # Handle multi-line titles (e.g. "Reciprocation: The Old Give\nand Take")
                    title = self._normalize_title(next_line)
                    if self._is_valid_chapter_title(title):
                        toc_entries.append((num, title))

        # Need at least 3 TOC entries to be confident this is a real TOC
        if len(toc_entries) < 3:
            return []

        logger.debug(f"Found {len(toc_entries)} TOC entries: {[t for _, t in toc_entries[:5]]}...")

        # Step 2: Compute body_start based on where TOC entries actually end.
        # The TOC is usually in the first few hundred lines. We look for
        # the last densely-packed TOC entry and start searching after that.
        last_toc_entry_pos = 0
        for i, line in enumerate(toc_lines):
            # Check both same-line format ("1. Title") and split-line format ("1\nTitle")
            is_toc_line = toc_entry_re.match(line.strip())
            if not is_toc_line and toc_split_re.match(line.strip()):
                # Split format: number alone on this line, check if next is a title
                if i + 1 < len(toc_lines) and toc_lines[i + 1].strip():
                    is_toc_line = True
            if is_toc_line:
                # Approximate position in content
                last_toc_entry_pos = sum(len(toc_lines[j]) + 1 for j in range(i + 1))

        # Start body search well after the TOC, with a reasonable margin
        # (TOC might be followed by acknowledgments, dedication, etc.)
        # Use a moderate margin past the TOC to avoid skipping early chapters.
        # The old 3x multiplier could overshoot when the TOC is long.
        # Use 1.5x with a minimum margin of 2000 chars.
        body_start = min(
            last_toc_entry_pos + max(2000, last_toc_entry_pos // 2),
            toc_search_limit
        )
        logger.debug(f"TOC ends ~pos {last_toc_entry_pos}, body search starts at {body_start}")

        results = []

        for num, title in toc_entries:
            # Search for the title as a standalone line in the body.
            # In PDF-extracted text, chapter headings appear on their own line
            # (possibly preceded by a page break \f or blank lines).
            #
            # Strategy: try multiple search keys in order of specificity:
            # 1. First 4 words of the full title
            # 2. Main title only (before colon, for "Title: Subtitle" format)
            # 3. Shorter prefix if still not found
            search_keys = []
            words = title.split()
            if len(words) >= 3:
                search_keys.append(' '.join(words[:4]))
            else:
                search_keys.append(title)

            # For "Title: Subtitle" format, also try just the main title
            if ':' in title:
                main_title = title.split(':')[0].strip()
                if main_title and len(main_title) >= 4 and main_title not in search_keys:
                    search_keys.append(main_title)

            found_match = False
            for search_key in search_keys:
                if found_match:
                    break
                # Build a regex that matches the search key with flexible whitespace.
                # PDF extraction can insert newlines, extra spaces, or page breaks
                # within titles (e.g., "WEAPONS OF\nINFLUENCE" for "Weapons of Influence").
                # Allow optional punctuation (comma, semicolon) between words
                # to handle differences like "Guilt and" vs "Guilt, and"
                flexible_pattern = r'[,;]?\s+'.join(re.escape(w) for w in search_key.split())

                for match in re.finditer(flexible_pattern, content[body_start:], re.IGNORECASE):
                    pos = body_start + match.start()

                    # Verify this is a STANDALONE heading, not mid-sentence:
                    # 1. Must be near the start of a line
                    line_start = content.rfind('\n', max(0, pos - 200), pos)
                    if line_start == -1:
                        line_start = 0
                    else:
                        line_start += 1  # skip the \n itself

                    # Text between line start and our match should be minimal
                    prefix = content[line_start:pos].strip('\n\r\t\f ')
                    if len(prefix) > 5:
                        continue  # Match is mid-sentence, skip

                    # 2. The rest of this line should be short (just the title, not a paragraph)
                    line_end = content.find('\n', pos)
                    if line_end == -1:
                        line_end = len(content)
                    line_text = content[line_start:line_end].strip()
                    if len(line_text) > len(title) + 30:
                        continue  # Too much extra text on the line

                    results.append((num, title, pos))
                    found_match = True
                    break  # Take first valid match in body

        if len(results) >= 3:
            # Check for missing chapters (TOC entries not found in body)
            found_nums = set(r[0] for r in results)
            toc_nums = set(n for n, _ in toc_entries)
            missing = sorted(toc_nums - found_nums)
            if missing:
                missing_details = []
                for n in missing:
                    title = next((t for num, t in toc_entries if num == n), "?")
                    missing_details.append(f"Ch.{n} ({title})")
                logger.warning(
                    f"TOC has {len(toc_entries)} entries but {len(missing)} not found in body: "
                    f"{', '.join(missing_details)}. "
                    f"These may be in the preamble region (body_start={body_start})."
                )

            logger.debug(f"TOC-based detection found {len(results)} chapters in body")
            return results

        return []

    def _detect_toc_region(self, content: str) -> int:
        """Detect where the Table of Contents ends and real content begins.

        Returns the character position where actual content starts.
        Uses density of chapter-like markers to find TOC boundaries.
        """
        # Look for explicit TOC markers
        toc_end_patterns = [
            r'(?i)(?:end of )?table of contents',
            r'(?i)(?:^|\n)(?:part|chapter)\s+(?:1|one|i)\b',  # First real chapter
        ]

        # Check first 10% of document for TOC-like density
        search_region = content[:len(content) // 10]

        # Count chapter-keyword matches in small windows
        # TOC has many chapter titles packed close together
        lines = search_region.split('\n')
        chapter_line_indices = []
        for i, line in enumerate(lines):
            if re.match(r'(?i)^\s*(chapter|part)\s+\d', line.strip()):
                chapter_line_indices.append(i)

        if len(chapter_line_indices) >= 3:
            # Multiple chapter markers in the front - likely a TOC
            # Find where the density drops (gap of 5+ lines between markers)
            last_toc_line = chapter_line_indices[0]
            for j in range(1, len(chapter_line_indices)):
                if chapter_line_indices[j] - chapter_line_indices[j-1] > 10:
                    # Big gap - previous marker was last TOC entry
                    break
                last_toc_line = chapter_line_indices[j]

            # Return position after the last dense cluster
            if last_toc_line < len(lines):
                toc_end_pos = sum(len(lines[k]) + 1 for k in range(last_toc_line + 1))
                logger.debug(f"TOC region detected, ends at ~line {last_toc_line} (pos {toc_end_pos})")
                return toc_end_pos

        return 0  # No TOC detected

    def _find_chapters(self, content: str) -> List[Tuple[int, str, int]]:
        """
        Find chapter markers in content using a multi-phase approach.

        Strategy: Try multiple detection methods, filter each, pick the best.
        - Explicit keywords (Chapter N, Part N, etc.)
        - Bare numbered headings (N. Title)
        - TOC-based detection (parse TOC, find titles in body)

        Returns:
            List of (chapter_num, chapter_title, start_position)
        """
        doc_len = len(content)

        # Detect TOC region to skip during filtering
        toc_end = self._detect_toc_region(content)

        # ── Explicit keyword patterns ──
        phase1 = self._find_explicit_chapters(content)
        phase1.sort(key=lambda x: x[2])  # Sort by position before filtering
        phase1_filtered = self._filter_chapters(content, phase1, toc_end) if len(phase1) > 1 else phase1

        # ── Bare numbered headings ──
        phase2 = self._find_bare_numbered_headings(content)
        phase2.sort(key=lambda x: x[2])
        phase2_filtered = self._filter_chapters(content, phase2, toc_end) if len(phase2) > 1 else phase2

        # ── TOC-based detection ──
        phase3 = self._find_chapters_via_toc(content)
        phase3.sort(key=lambda x: x[2])  # Critical: body positions may not match TOC order
        phase3_filtered = self._filter_chapters(content, phase3, toc_end) if len(phase3) > 1 else phase3

        # Pick the best result using priority-aware selection.
        # Key insight: TOC-based detection is the most authoritative because it uses
        # the author's own table of contents. Explicit "Chapter N" patterns can match
        # endnote headers. Bare numbers are the least reliable (can match footnotes).
        #
        # Also check chapter distribution: real chapters span the document,
        # endnote chapter markers are clustered in the last portion.
        def is_well_distributed(chapters):
            """Real chapters span most of the document; endnotes are clustered."""
            if len(chapters) < 3:
                return True
            positions = [pos / doc_len for _, _, pos in chapters]
            span = max(positions) - min(positions)
            return span >= 0.4  # Chapters should span at least 40% of document

        # Check each phase's distribution
        phase1_distributed = is_well_distributed(phase1_filtered)
        phase3_distributed = is_well_distributed(phase3_filtered)

        # Selection priority:
        # 1. TOC-based if it found >= 5 chapters (most authoritative)
        # 2. Explicit if well-distributed and more than alternatives
        # 3. Bare numbers as last resort
        if len(phase3_filtered) >= 5:
            best_name, best_chapters = "toc_based", phase3_filtered
        elif len(phase1_filtered) >= 3 and phase1_distributed:
            best_name, best_chapters = "explicit", phase1_filtered
        elif len(phase3_filtered) >= 3:
            best_name, best_chapters = "toc_based", phase3_filtered
        elif len(phase1_filtered) >= 3:
            # Even if clustered, it's better than nothing
            best_name, best_chapters = "explicit", phase1_filtered
        elif len(phase2_filtered) >= 3:
            best_name, best_chapters = "bare_numbered", phase2_filtered
        else:
            # Take whatever has the most
            candidates = [
                ("explicit", phase1_filtered),
                ("bare_numbered", phase2_filtered),
                ("toc_based", phase3_filtered),
            ]
            best_name, best_chapters = max(candidates, key=lambda x: len(x[1]))

        all_candidates = [
            ("explicit", phase1_filtered),
            ("bare_numbered", phase2_filtered),
            ("toc_based", phase3_filtered),
        ]

        if best_chapters:
            logger.info(f"Chapter detection: using '{best_name}' method ({len(best_chapters)} chapters)")
            for name, chs in all_candidates:
                if chs and name != best_name:
                    logger.debug(f"  Alternative '{name}': {len(chs)} chapters")
        else:
            logger.info("No chapters detected by any method")
            return []

        # Renumber: preserve original numbers if they're consistent,
        # otherwise renumber sequentially
        chapters = self._renumber_chapters(best_chapters)

        return chapters

    def _find_explicit_chapters(self, content: str) -> List[Tuple[int, str, int]]:
        """Find chapters using explicit keyword patterns (Chapter N, Part N, etc.)"""
        chapters = []
        seen_positions = set()

        for pattern in self.CHAPTER_PATTERNS:
            for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                pos = match.start()

                # Skip if we already have a chapter near this position
                if any(abs(pos - seen_pos) < 50 for seen_pos in seen_positions):
                    continue

                # Extract chapter number and title
                groups = match.groups()
                if len(groups) >= 2:
                    chapter_id = groups[0]
                    chapter_title = groups[1].strip() if groups[1] else ""
                else:
                    chapter_id = str(len(chapters) + 1)
                    chapter_title = match.group(0).strip()

                # Convert chapter_id to number
                # Handle OCR artifacts like "1 3" for "13" (spaces in numbers)
                chapter_id_clean = chapter_id.replace(' ', '')
                try:
                    if chapter_id.lower() in self.SPECIAL_CHAPTERS:
                        chapter_num = 0  # Special chapters get 0
                    elif chapter_id_clean.startswith('0x'):
                        chapter_num = int(chapter_id_clean, 16)
                    elif chapter_id_clean.isdigit():
                        chapter_num = int(chapter_id_clean)
                    else:
                        # Roman numerals or other
                        chapter_num = len(chapters) + 1
                except ValueError:
                    chapter_num = len(chapters) + 1

                # Clean up title
                chapter_title = self._normalize_title(chapter_title)
                if not chapter_title:
                    chapter_title = f"Chapter {chapter_num}"

                # Handle multi-line titles: if the captured title ends with a
                # preposition/conjunction (e.g., "WEAPONS OF" from "WEAPONS OF\nINFLUENCE"),
                # try to extend it with subsequent lines (up to 3 more lines).
                # This handles cases like:
                #   "The Problem of\nProcrastination and\nSelf-Control"
                incomplete_endings = [' of', ' and', ' the', ' in', ' to', ' or',
                                      ' a', ' an', ' for', ' with']
                scan_pos = match.end()
                for _ in range(3):  # Try extending up to 3 lines
                    title_lower = chapter_title.lower().rstrip(' .,;:')
                    if not any(title_lower.endswith(ending) for ending in incomplete_endings):
                        break  # Title is complete
                    next_nl = content.find('\n', scan_pos)
                    if next_nl == -1:
                        break
                    next_line = content[scan_pos:next_nl].strip()
                    if not next_line or len(next_line) > 80 or (next_line[0].islower()):
                        break
                    # Skip subtitle lines (e.g., "Why We Can't...")
                    if next_line.lower().startswith('why '):
                        break
                    extended = f"{chapter_title} {self._normalize_title(next_line)}"
                    chapter_title = extended
                    scan_pos = next_nl + 1

                # Reject chapter entries that are page references (notes section)
                # e.g., "CHAPTER 4 (PAGES 114-166)"
                if re.match(r'^\(PAGES?\s+\d', chapter_title, re.IGNORECASE):
                    continue

                chapters.append((chapter_num, chapter_title, pos))
                seen_positions.add(pos)

        chapters.sort(key=lambda x: x[2])
        return chapters

    def _filter_chapters(
        self,
        content: str,
        chapters: List[Tuple[int, str, int]],
        toc_end: int
    ) -> List[Tuple[int, str, int]]:
        """Filter out TOC entries, notes section duplicates, and false positives."""
        filtered = []
        seen_titles = set()

        for i, (num, title, pos) in enumerate(chapters):
            # Calculate content between this marker and the next
            if i + 1 < len(chapters):
                content_between = chapters[i + 1][2] - pos
            else:
                content_between = len(content) - pos

            # Normalize title for deduplication
            title_normalized = ''.join(
                c.lower() for c in title if c.isalpha() or c.isspace()
            ).strip()

            # Skip duplicate titles (likely notes/references section)
            if title_normalized in seen_titles:
                logger.debug(f"Filtering duplicate title '{title[:30]}' at pos {pos}")
                continue

            # Skip entries in the TOC region (before actual content starts)
            if pos < toc_end:
                logger.debug(f"Filtering TOC entry '{title[:30]}' at pos {pos} (before toc_end={toc_end})")
                continue

            # Validate title
            if not self._is_valid_chapter_title(title):
                logger.debug(f"Filtering invalid title '{title[:30]}' at pos {pos}")
                continue

            # Filter by content size between markers
            # Real chapters have substantial content (1000+ chars)
            # Notes section (last 15% of doc) needs more content to be considered real
            doc_position_ratio = pos / len(content) if len(content) > 0 else 0
            is_in_notes_section = doc_position_ratio > 0.85

            min_content = 5000 if is_in_notes_section else 1000

            if content_between >= min_content:
                filtered.append((num, title, pos))
                seen_titles.add(title_normalized)
                logger.debug(f"Keeping chapter '{title[:30]}' at pos {pos} (content: {content_between} chars)")
            else:
                logger.debug(
                    f"Filtering entry '{title[:30]}' at pos {pos} "
                    f"(only {content_between} chars, doc_ratio={doc_position_ratio:.2f})"
                )

        return filtered

    def _renumber_chapters(
        self,
        chapters: List[Tuple[int, str, int]]
    ) -> List[Tuple[int, str, int]]:
        """Renumber chapters, preserving original numbers when they form a
        reasonable sequence, otherwise renumbering sequentially."""
        if not chapters:
            return chapters

        original_nums = [ch[0] for ch in chapters]

        # Check if original numbers form a reasonable sequence:
        # - Mostly increasing
        # - No huge gaps (> 2x the count)
        # - Numbers are in a sensible range
        is_reasonable = True

        if max(original_nums) > len(chapters) * 3:
            is_reasonable = False
        else:
            # Check that numbers are mostly increasing
            inversions = sum(
                1 for i in range(1, len(original_nums))
                if original_nums[i] <= original_nums[i-1]
            )
            if inversions > len(chapters) * 0.3:
                is_reasonable = False

        if is_reasonable:
            # Keep original numbers
            logger.debug(f"Keeping original chapter numbers: {original_nums}")
            return chapters
        else:
            # Renumber sequentially
            logger.debug(f"Renumbering chapters (originals were: {original_nums})")
            return [(i + 1, title, pos) for i, (_, title, pos) in enumerate(chapters)]

    def _split_and_chunk_chapters(
        self,
        content: str,
        chapter_markers: List[Tuple[int, str, int]]
    ) -> List[ChapterChunks]:
        """Split content by chapter markers and chunk each chapter"""
        chapters = []

        for i, (chapter_num, chapter_title, start_pos) in enumerate(chapter_markers):
            # Determine end position
            if i + 1 < len(chapter_markers):
                end_pos = chapter_markers[i + 1][2]
            else:
                end_pos = len(content)

            # Extract chapter content
            chapter_content = content[start_pos:end_pos].strip()

            # Skip very small chapters (likely false positives)
            chapter_tokens = self.token_counter.count_tokens(chapter_content)
            if chapter_tokens < self.min_chapter_size and len(chapter_markers) > 1:
                logger.debug(f"Skipping small chapter {chapter_num}: {chapter_tokens} tokens")
                continue

            # Chunk the chapter content
            chunks = self.base_chunker.chunk_by_tokens(
                text=chapter_content,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )

            # Convert to string list
            chunk_texts = []
            for chunk in chunks:
                if hasattr(chunk, 'content'):
                    chunk_texts.append(chunk.content)
                elif hasattr(chunk, 'text'):
                    chunk_texts.append(chunk.text)
                elif isinstance(chunk, str):
                    chunk_texts.append(chunk)
                else:
                    chunk_texts.append(str(chunk))

            chapter = ChapterChunks(
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                chunks=chunk_texts,
                token_count=chapter_tokens,
                start_pos=start_pos,
                end_pos=end_pos,
            )
            chapters.append(chapter)

            logger.debug(f"Chapter {chapter_num}: '{chapter_title}' - {len(chunk_texts)} chunks, {chapter_tokens} tokens")

        # Fallback: if all chapters were filtered out, treat entire content as one chapter
        if not chapters and chapter_markers:
            logger.info("All chapters were below min_chapter_size, treating as single chapter")
            full_content = content.strip()
            full_tokens = self.token_counter.count_tokens(full_content)
            chunks = self.base_chunker.chunk_by_tokens(
                text=full_content,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            chunk_texts = []
            for chunk in chunks:
                if hasattr(chunk, 'content'):
                    chunk_texts.append(chunk.content)
                elif hasattr(chunk, 'text'):
                    chunk_texts.append(chunk.text)
                elif isinstance(chunk, str):
                    chunk_texts.append(chunk)
                else:
                    chunk_texts.append(str(chunk))

            chapters.append(ChapterChunks(
                chapter_num=1,
                chapter_title="Full Document",
                chunks=chunk_texts,
                token_count=full_tokens,
                start_pos=0,
                end_pos=len(content),
            ))

        return chapters

    def _extract_title(self, content: str) -> Optional[str]:
        """
        Try to extract document title from beginning.
        Uses multiple strategies:
        1. First non-empty line if it looks like a title
        2. Look for "Title:" or similar label
        3. Look for book title in cataloging info
        """
        lines = content[:5000].split('\n')

        # Skip metadata patterns aggressively
        skip_patterns = [
            'page ', 'copyright', 'isbn', 'published', 'contents', 'www.',
            'dedication', 'acknowledgment', 'illustration', 'credit',
            'edition', 'library of congress', 'reserved', 'random house',
            'doubleday', 'penguin', 'catalog', 'trademark', 'cover', 'title page',
            'jacket', 'note about', 'other books', 'all rights', 'printed in',
            'created on', 'created by', 'updated:', 'author:'
        ]

        import re

        # Strategy 1: Check first non-empty line - most likely the title
        for line in lines[:10]:
            line = line.strip()

            # Skip empty lines and very short lines
            if len(line) < 3 or len(line) > 100:
                continue

            # Skip lines that look like metadata
            if any(x in line.lower() for x in skip_patterns):
                continue

            # Skip lines with special characters that indicate metadata
            if any(c in line for c in ['©', 'http', '@', '\\', ':', '|', '[', ']']):
                continue

            # First substantive line is likely the title
            if line[0].isupper() and not line.startswith('Chapter'):
                words = line.split()
                if 1 <= len(words) <= 12:
                    logger.debug(f"Found title via first line: {line}")
                    return line

        # Strategy 2: Look for "Title:" label
        full_text = '\n'.join(lines[:50])
        title_label = re.search(r'[Tt]itle[:\s]+([^\n]+)', full_text)
        if title_label:
            potential = title_label.group(1).strip()
            if 3 <= len(potential) <= 80:
                logger.debug(f"Found title via label: {potential}")
                return potential

        # Strategy 3: Look for capitalized heading pattern
        for line in lines[:30]:
            line = line.strip()

            if len(line) < 5 or len(line) > 80:
                continue

            if any(x in line.lower() for x in skip_patterns):
                continue

            # Look for ALL CAPS or Title Case patterns
            words = line.split()
            if 2 <= len(words) <= 10:
                if line.isupper() or (line[0].isupper() and not any(c in line for c in ['©', 'http', '@', '/'])):
                    logger.debug(f"Found title via scan: {line}")
                    return line

        return None

    def get_chapter_summary_targets(
        self,
        structure: DocumentStructure,
        compression_ratio: float = 0.1
    ) -> Dict[int, int]:
        """
        Calculate target word counts for each chapter summary.

        Args:
            structure: Document structure from chunk_document()
            compression_ratio: Target output / input ratio (e.g., 0.1 = 10%)

        Returns:
            Dict mapping chapter_num to target word count
        """
        targets = {}

        for chapter in structure.chapters:
            # Estimate words from tokens (roughly 0.75 words per token)
            estimated_words = int(chapter.token_count * 0.75)
            target_words = max(100, int(estimated_words * compression_ratio))
            targets[chapter.chapter_num] = target_words

        return targets
