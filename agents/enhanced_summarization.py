#!/usr/bin/env python3
"""
Enhanced Summarization Agent

Parallel chapter-aware summarization with narrative output.
Transforms large books into condensed readable narratives.

Key Features:
- Chapter-aware document chunking
- Parallel chapter summarization (asyncio)
- Narrative style output (not bullet points)
- Configurable compression ratios
- Progress tracking callbacks
"""

import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from core.base_agent import BaseAgent
from utilities.chapter_chunker import ChapterAwareChunker, DocumentStructure, ChapterChunks
from utilities.ephemeral_store import EphemeralSummaryStore, SummaryJob
from utilities.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class SummaryStyle(Enum):
    """Summary output styles"""
    NARRATIVE = "narrative"      # Story-like condensed text (default for fiction)
    ACADEMIC = "academic"        # Formal structure with sections
    TECHNICAL = "technical"      # Documentation style
    EXPLANATORY = "explanatory"  # In-depth explanations for science/educational books
    QUICK = "quick"              # Bullet points (legacy)


class CompressionLevel(Enum):
    """Compression level presets"""
    DETAILED = "detailed"    # 20% - Preserve most detail (5x reduction)
    STANDARD = "standard"    # 10% - Balanced (10x reduction) - default
    CONDENSED = "condensed"  # 5% - Just key points (20x reduction)
    OUTLINE = "outline"      # 2% - Bare minimum (50x reduction)

    @property
    def ratio(self) -> float:
        """Get compression ratio for this level"""
        ratios = {
            "detailed": 0.20,
            "standard": 0.10,
            "condensed": 0.05,
            "outline": 0.02
        }
        return ratios[self.value]

    @property
    def description(self) -> str:
        """Human-readable description"""
        descriptions = {
            "detailed": "20% - Preserve most detail, nuance, and examples",
            "standard": "10% - Balanced summary, key events and themes",
            "condensed": "5% - Just the essential plot/arguments",
            "outline": "2% - Bare minimum, chapter-by-chapter outline"
        }
        return descriptions[self.value]


@dataclass
class ChapterSummaryResult:
    """Result of summarizing a single chapter"""
    chapter_num: int
    chapter_title: str
    original_tokens: int
    summary_tokens: int
    summary: str
    processing_time: float
    success: bool
    error: Optional[str] = None
    warnings: Optional[List[str]] = None


@dataclass
class BookSummaryResult:
    """Complete book summarization result"""
    title: str
    file_path: str
    total_chapters: int
    chapters_processed: int
    original_tokens: int
    summary_tokens: int
    compression_ratio: float
    chapter_summaries: List[ChapterSummaryResult]
    combined_summary: str
    processing_time: float
    style: SummaryStyle
    success: bool
    errors: List[str] = field(default_factory=list)
    # Track if user specified a chapter range for proper filename suffix
    user_specified_range: bool = False
    requested_chapters: Optional[List[int]] = None


# Narrative prompt templates - Optimized for longer, complete output
NARRATIVE_PROMPTS = {
    SummaryStyle.NARRATIVE: """You are condensing a book chapter into a shorter but complete narrative that PRESERVES the author's voice and atmosphere.

GROUNDING REQUIREMENTS:
- Your summary must be BASED ON the ORIGINAL CONTENT provided below
- DO NOT invent new characters, major events, or plot points not in the original
- DO NOT add references to unrelated works (movies, other books, etc.)
- Match the GENRE and TONE of the source material faithfully

PRESERVING THE AUTHOR'S VOICE:
- KEEP descriptive passages that establish atmosphere and setting
- MAINTAIN the author's distinctive tone (dark, whimsical, lyrical, etc.)
- PRESERVE key imagery, metaphors, and sensory details that define scenes
- RETAIN dialogue that reveals character or advances key moments
- Condense exposition but keep evocative descriptions that create mood
- If the author writes poetically, maintain that quality in your summary

CRITICAL REQUIREMENTS:
- Write approximately {target_words} words (between {min_words} and {max_words} words)
- {compression_guidance}
- Preserve the story flow, key points, and emotional beats
- Remove only true repetition - NOT descriptive richness that creates atmosphere
- Write in flowing prose - NO bullet points, NO headers, NO section titles
- The result should read as a complete condensed chapter that FEELS like the original

FORBIDDEN - DO NOT INCLUDE:
- Word counts or notes about word counts
- Meta-commentary like "This summary..." or "This condensed narrative..."
- Multiple versions or alternatives
- ANY information not present in the ORIGINAL CONTENT below
- Flattening the prose to mere plot points - KEEP the flavor

CHAPTER {chapter_num}: {chapter_title}

ORIGINAL CONTENT TO SUMMARIZE:
{content}

{context_note}

Write the condensed narrative now (minimum {target_words} words), preserving the author's voice and atmosphere:""",

    SummaryStyle.ACADEMIC: """You are creating an academic summary of this chapter/section.

ANTI-HALLUCINATION WARNING:
- ONLY summarize the CONTENT provided below
- DO NOT invent facts, studies, or information not in the original
- Every claim in your summary MUST come from the provided content
- If you don't have information about something, don't make it up

CRITICAL REQUIREMENTS:
- Write approximately {target_words} words (between {min_words} and {max_words} words)
- {compression_guidance}
- Use formal, scholarly language throughout
- Maintain objectivity and analytical distance
- Structure with clear logical flow

FORBIDDEN - DO NOT INCLUDE:
- Information not present in the CONTENT below
- Made-up statistics, names, or events
- Word counts or meta-commentary
- Citations to sources not in the original

CHAPTER {chapter_num}: {chapter_title}

CONTENT TO SUMMARIZE:
{content}

{context_note}

Write an academic summary using ONLY information from the content above:
1. Central thesis/argument with full elaboration
2. Supporting evidence and data points
3. Methodology and approach (if applicable)
4. Key findings and their significance
5. Implications and conclusions

Academic summary (minimum {target_words} words):""",

    SummaryStyle.TECHNICAL: """You are creating a technical documentation summary.

ANTI-HALLUCINATION WARNING:
- ONLY document what is ACTUALLY in the CONTENT below
- DO NOT invent code examples, commands, or configurations not present
- DO NOT make up technical terms, APIs, or implementations
- DO NOT add fictional scenarios or use cases not in the source
- If the content describes something, summarize THAT, not what you imagine it might be
- Every technical claim MUST come from the provided content

CRITICAL REQUIREMENTS:
- Write approximately {target_words} words (between {min_words} and {max_words} words)
- {compression_guidance}
- Use precise technical terminology FROM the source material
- Include code snippets, commands, or configurations ONLY if present in the original
- Structure with clear sections and bullet points

FORBIDDEN - DO NOT INCLUDE:
- Information, APIs, or code not in the CONTENT below
- Made-up implementation details or examples
- Vague or imprecise language
- Word counts or meta-commentary
- "Summary:", "Technical summary:", or similar prefixes

CHAPTER {chapter_num}: {chapter_title}

CONTENT TO SUMMARIZE:
{content}

{context_note}

Write a well-structured technical summary covering ONLY what's in the content above:
- Core concepts and terminology (from the source)
- Implementation details and workflows (from the source)
- Code examples or configurations (ONLY if in the source)
- Important parameters and gotchas (from the source)

Begin your summary directly (no prefixes):""",

    SummaryStyle.EXPLANATORY: """You are creating an explanatory summary of educational/scientific content that makes complex concepts CLEAR and MEMORABLE.

This mode is designed for books like Feynman's Lectures, science texts, philosophy, and any material with difficult concepts that benefit from explicit explanation.

CORE MISSION:
- Help readers UNDERSTAND, not just recall
- When the author gives an example, EXPLAIN what it demonstrates: "In this example, Feynman shows us that..."
- When mathematical formulas appear, EXPLAIN what each part means and why it matters
- When concepts build on each other, REMIND the reader of prior context
- Bridge gaps that might confuse a reader who hasn't read the material recently

EXPLANATION REQUIREMENTS:
- For EXAMPLES: Explicitly state what the example illustrates and why it's significant
  * "The author uses the analogy of X to show that Y works because..."
  * "This example demonstrates the principle that..."
- For MATH/FORMULAS: Break down the notation and meaning
  * "The equation E=mc² tells us that energy (E) equals mass (m) times the speed of light squared (c²), meaning..."
  * "When the author writes ∫f(x)dx, this represents the area under the curve, which in this context means..."
- For CONCEPTS: Connect to what came before and what comes next
  * "Building on the earlier discussion of X, we now see that..."
  * "This prepares us to understand Y in later chapters..."
- For DIFFICULT IDEAS: Rephrase in multiple ways if helpful
  * "In other words..." or "Another way to think about this..."

GROUNDING REQUIREMENTS:
- Base explanations ONLY on the ORIGINAL CONTENT below
- DO NOT introduce concepts or examples not in the source
- If the author doesn't fully explain something, you may clarify the logic, but mark speculation clearly

CRITICAL REQUIREMENTS:
- Write approximately {target_words} words (between {min_words} and {max_words} words)
- {compression_guidance}
- Prioritize CLARITY over brevity - it's okay to be longer if it aids understanding
- Write in engaging, accessible prose - avoid dry academic tone
- Use the author's examples but add explicit explanations of what they show
- Maintain narrative flow while being pedagogically clear

FORBIDDEN - DO NOT INCLUDE:
- Word counts or meta-commentary
- Unexplained jargon or formulas
- Skipping over difficult parts - these need THE MOST attention
- Assuming the reader remembers everything from previous chapters

CHAPTER {chapter_num}: {chapter_title}

CONTENT TO EXPLAIN AND SUMMARIZE:
{content}

{context_note}

Write an explanatory summary that makes this material CLEAR and UNDERSTANDABLE, especially focusing on examples and complex concepts:""",

    SummaryStyle.QUICK: """Create a bullet-point summary of this chapter.

CHAPTER {chapter_num}: {chapter_title}

CONTENT:
{content}

{context_note}

Provide a comprehensive bullet-point summary:
- Include ALL major events, characters, and concepts
- Each bullet should be a complete thought (1-2 sentences)
- Minimum 8-12 bullet points covering everything important
- Use clear, concise language

Bullet points:"""
}

# Russian narrative prompt templates
NARRATIVE_PROMPTS_RU = {
    SummaryStyle.NARRATIVE: """Вы сокращаете главу книги в более короткий, но полный рассказ, СОХРАНЯЯ голос и атмосферу автора.

ТРЕБОВАНИЯ:
- Резюме должно быть ОСНОВАНО на ОРИГИНАЛЬНОМ СОДЕРЖАНИИ ниже
- НЕ выдумывайте новых персонажей, событий или сюжетных моментов
- Соответствуйте ЖАНРУ и ТОНУ исходного материала

СОХРАНЕНИЕ ГОЛОСА АВТОРА:
- СОХРАНЯЙТЕ описательные фрагменты, создающие атмосферу
- ПОДДЕРЖИВАЙТЕ уникальный тон автора
- СОХРАНЯЙТЕ ключевые образы, метафоры и чувственные детали
- ОСТАВЛЯЙТЕ диалоги, раскрывающие характер или продвигающие ключевые моменты
- Сокращайте экспозицию, но сохраняйте образные описания

КРИТИЧЕСКИЕ ТРЕБОВАНИЯ:
- Напишите примерно {target_words} слов (от {min_words} до {max_words} слов)
- {compression_guidance}
- Сохраняйте течение истории, ключевые моменты и эмоциональные переломы
- Пишите в плавной прозе — БЕЗ маркеров, БЕЗ заголовков, БЕЗ названий разделов
- Результат должен читаться как полная сокращенная глава

ЗАПРЕЩЕНО:
- Подсчет слов или мета-комментарии
- ЛЮБУЮ информацию, которой НЕТ в оригинальном содержании
- Упрощение прозы до голых сюжетных пунктов

ГЛАВА {chapter_num}: {chapter_title}

ОРИГИНАЛЬНОЕ СОДЕРЖАНИЕ ДЛЯ РЕЗЮМЕ:
{content}

{context_note}

Напишите сокращенный рассказ (примерно {target_words} слов), сохраняя голос автора:""",

    SummaryStyle.ACADEMIC: """Вы создаете академическое резюме этой главы.

ТРЕБОВАНИЯ:
- ТОЛЬКО резюмируйте содержание, предоставленное ниже
- НЕ выдумывайте факты или информацию
- Используйте формальный, научный язык

- Напишите примерно {target_words} слов (от {min_words} до {max_words} слов)
- {compression_guidance}

ГЛАВА {chapter_num}: {chapter_title}

СОДЕРЖАНИЕ:
{content}

{context_note}

Напишите академическое резюме (примерно {target_words} слов):""",

    SummaryStyle.QUICK: """Создайте краткое резюме в виде списка ключевых пунктов.

- Напишите примерно {target_words} слов
- {compression_guidance}
- Минимум 8-12 пунктов, покрывающих всё важное

ГЛАВА {chapter_num}: {chapter_title}

{content}

{context_note}

Ключевые пункты:""",
}

# Context note templates for continuity
CONTEXT_TEMPLATES = {
    "first_chapter": "This is the first chapter - establish the setting and introduce key characters.",
    "with_previous": """PREVIOUSLY: {previous_summary}

Continue the narrative from where we left off, maintaining character and plot continuity.""",
    "standalone": ""  # No context needed
}

CONTEXT_TEMPLATES_RU = {
    "first_chapter": "Это первая глава — установите обстановку и представьте ключевых персонажей.",
    "with_previous": """РАНЕЕ: {previous_summary}

Продолжайте рассказ, сохраняя последовательность персонажей и сюжета.""",
    "standalone": ""
}


class EnhancedSummarizationAgent(BaseAgent):
    """
    Enhanced summarization with parallel chapter processing.

    Transforms large documents into condensed narratives through:
    - Chapter-aware chunking and boundary detection
    - Parallel chapter summarization with asyncio
    - Narrative style output (configurable)
    - Progress tracking for UI updates

    Usage (Interactive mode):
        "Summarize /path/to/book.epub"
        "Summarize chapter 2 from /path/to/book.epub"
        "Summarize /path/to/paper.pdf with academic style"

    Usage (Python):
        agent = EnhancedSummarizationAgent(model_manager)
        result = agent.summarize_file("/path/to/book.epub")
    """

    def __init__(
        self,
        model_manager,
        memory_manager=None,
        config_path: str = "config/summarization_config.yaml"
    ):
        """
        Initialize enhanced summarization agent.

        Args:
            model_manager: LazyModelManager for LLM access
            memory_manager: Optional MemoryManager for RAG integration
            config_path: Path to summarization config
        """
        super().__init__("summarization", model_manager)

        # Load config
        self.config = self._load_config(config_path)

        # Initialize components
        # Use lower min_chapter_size (50) to handle small documents/chapters
        # Default of 500 was causing small files to be skipped entirely
        # Small chapters below threshold still get processed via fallback
        self.chunker = ChapterAwareChunker(
            chunk_size=2000,
            chunk_overlap=200,
            min_chapter_size=self.config.get('chunking', {}).get('min_chapter_size', 50)
        )
        self.ephemeral_store = EphemeralSummaryStore(
            db_path=self.config.get('output', {}).get('directory', 'summaries') + '/ephemeral',
            use_lancedb=False
        )
        self.token_counter = TokenCounter()
        self.memory_manager = memory_manager

        # Thread pool for running sync LLM calls in async context
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Stats tracking (compatible with old agent)
        self.stats.update({
            'files_summarized': 0,
            'chunks_processed': 0,
            'total_tokens_processed': 0,
            'successful_summaries': 0
        })

        logger.info("EnhancedSummarizationAgent initialized")

    def _get_max_content_tokens(self, style: 'SummaryStyle' = None) -> int:
        """Get max content tokens for the current style.

        Explanatory mode uses a higher limit (3200) because it needs more
        source context for the detailed explanations it produces.
        Other styles use the default (2500) which is safe for 4096 context.
        """
        base = self.config.get('processing', {}).get('max_content_tokens', 6000)
        if style == SummaryStyle.EXPLANATORY:
            return max(base, 6000)
        return base

    def _load_config(self, config_path: str) -> Dict:
        """Load summarization config from YAML"""
        try:
            import yaml
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file) as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")

        # Return defaults
        return {
            'styles': {'default': 'narrative'},
            'compression': {'default': 'standard'},
            'output': {
                'directory': 'summaries',
                'formats': ['markdown', 'pdf', 'epub']  # Generate all 3 formats
            },
            'processing': {
                'max_concurrent_chapters': 3,
                'max_content_tokens': 6000,
                'use_rolling_context': True
            }
        }

    def chat(self, message: str, history: list = None) -> str:
        """
        Interactive chat interface for summarization.

        Parses file paths and chapter specifications, shows document structure,
        allows chapter selection, and displays progress during summarization.

        When called from Oracle's background thread, interactive prompts are
        auto-confirmed since stdin is not available to this thread.
        """
        import re
        import threading
        from pathlib import Path
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
        from rich.prompt import Confirm, Prompt

        console = Console()

        # Detect if we're running in a background thread (e.g. Oracle worker)
        # In that case, stdin is not available - auto-confirm all prompts
        is_background = threading.current_thread() is not threading.main_thread()

        # Parse for file paths
        file_match = re.search(r'[~\/][^\s]+\.(pdf|txt|md|epub)', message, re.IGNORECASE)

        if not file_match:
            return (
                "I'm the Enhanced Summarization Agent. I can summarize books and documents.\n\n"
                "**Usage:**\n"
                "• Summarize ~/path/to/file.pdf\n"
                "• Summarize chapters 5-7 from ~/books/document.pdf\n"
                "• Summarize ~/book.pdf --style narrative\n\n"
                "**Options:**\n"
                "• Styles: narrative, academic, technical, explanatory, quick\n"
                "• Compression: detailed (20%), standard (10%), condensed (5%), outline (2%)"
            )

        file_path = file_match.group(0)
        if file_path.startswith('~'):
            file_path = str(Path(file_path).expanduser())

        if not Path(file_path).exists():
            return f"File not found: {file_path}"

        # Parse chapter specification
        chapter_match = re.search(r'chapters?\s+(\d+)\s*[-–]\s*(\d+)', message, re.IGNORECASE)
        chapters = None
        if chapter_match:
            start_ch = int(chapter_match.group(1))
            end_ch = int(chapter_match.group(2))
            chapters = list(range(start_ch, end_ch + 1))

        # Parse style (--style narrative, narrative style, explanatory style, etc.)
        style = None
        style_match = re.search(r'(?:--style\s+|style\s+)?(\b(?:narrative|academic|technical|explanatory|quick)\b)', message, re.IGNORECASE)
        if style_match:
            style = style_match.group(1).lower()

        # Parse compression level (detailed, standard, condensed, outline, quick)
        compression = "standard"  # default
        comp_match = re.search(r'\b(detailed|standard|condensed|outline|quick)\b', message, re.IGNORECASE)
        if comp_match:
            compression = comp_match.group(1).lower()

        # Load and analyze document structure
        console.print(f"\n[bold cyan]📖 Analyzing: {Path(file_path).name}[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Loading document...", total=None)

            try:
                content = self._load_file_content(file_path)
                structure = self.chunker.chunk_document(content, file_path)
            except Exception as e:
                return f"Error analyzing document: {e}"

        # Display chapters
        if structure.chapters:
            console.print(f"\n[bold]Document: {structure.title or Path(file_path).stem}[/bold]")
            console.print(f"Type: {structure.doc_type}")
            console.print(f"Total chapters: {len(structure.chapters)}\n")

            # Show ALL chapters
            for ch in structure.chapters:
                tokens = ch.token_count
                title_display = ch.chapter_title[:60] if len(ch.chapter_title) > 60 else ch.chapter_title
                console.print(f"  [{ch.chapter_num:2d}] {title_display} ({tokens:,} tokens)")

            # Show current settings
            console.print(f"\n[bold]Settings:[/bold]")
            console.print(f"  Style: [cyan]{style or 'narrative (default)'}[/cyan]")
            console.print(f"  Compression: [cyan]{compression}[/cyan]")

            # Chapter selection loop - allows re-selection if user says no
            # In background mode (called from Oracle), skip interactive prompts
            while True:
                # If no chapters specified, ask user (only in foreground)
                if not chapters and not is_background:
                    console.print("\n[bold]Select chapters to summarize:[/bold]")
                    console.print("[dim]Enter: range (1-5), list (1,3,5), or 'all'[/dim]")
                    selection = Prompt.ask(
                        "Chapters",
                        default="all"
                    )

                    if selection.lower() == 'q' or selection.lower() == 'quit':
                        return "Summarization cancelled."

                    if selection.lower() != 'all':
                        range_match = re.match(r'(\d+)\s*[-–]\s*(\d+)', selection)
                        if range_match:
                            chapters = list(range(int(range_match.group(1)), int(range_match.group(2)) + 1))
                        else:
                            try:
                                chapters = [int(x.strip()) for x in selection.split(',')]
                            except:
                                chapters = None

                # Show summary
                if chapters:
                    chapter_desc = f"chapters {chapters[0]}-{chapters[-1]}" if len(chapters) > 1 else f"chapter {chapters[0]}"
                    total_tokens = sum(ch.token_count for ch in structure.chapters if ch.chapter_num in chapters or
                                       structure.chapters.index(ch) + 1 in chapters)
                else:
                    chapter_desc = "all chapters"
                    total_tokens = sum(ch.token_count for ch in structure.chapters)

                console.print(f"\n[bold yellow]Selected:[/bold yellow] {chapter_desc} ({total_tokens:,} tokens)")

                # Confirm or re-select (auto-confirm in background mode)
                if is_background or Confirm.ask("Proceed with summarization?", default=True):
                    break  # Proceed with summarization
                else:
                    # Reset chapters to allow re-selection
                    chapters = None
                    console.print("\n[dim]Re-select chapters or enter 'q' to quit[/dim]")

        # Progress callback for Rich progress bar
        progress_state = {"progress": None, "task": None, "last_total": 0}

        def progress_callback(stage: str, current: int, total: int):
            if progress_state["progress"] is None:
                return
            # Skip progress updates until we have actual chunk counts (total > 1)
            if total <= 1 and stage in ["Loading", "Analyzing"]:
                return
            if progress_state["task"] is None:
                progress_state["task"] = progress_state["progress"].add_task(stage, total=total)
                progress_state["last_total"] = total
            elif total != progress_state["last_total"]:
                # Update total if it changed
                progress_state["progress"].update(progress_state["task"], total=total)
                progress_state["last_total"] = total
            progress_state["progress"].update(progress_state["task"], completed=current, description=stage)

        # Summarize with progress
        console.print(f"\n[bold green]🚀 Starting summarization...[/bold green]")
        if chapters:
            console.print(f"Processing chapters: {chapters[0]}-{chapters[-1]}")
        else:
            console.print("Processing all chapters")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
        ) as progress:
            progress_state["progress"] = progress

            result = self.summarize_file(
                file_path=file_path,
                chapters=chapters,
                depth=compression,
                style=style,
                progress_callback=progress_callback
            )

        if result.success:
            response = f"\n**Summary of {Path(file_path).name}"
            if chapters:
                response += f" (Chapters {chapters[0]}-{chapters[-1]})"
            response += "**\n\n"
            response += result.summary + "\n\n"
            if result.key_insights:
                response += "**Key Insights:**\n"
                for insight in result.key_insights[:5]:
                    response += f"• {insight}\n"
            response += f"\n_Processed in {result.processing_time:.1f}s_"
            return response
        else:
            return f"Summarization failed: {', '.join(result.errors)}"

    def _load_file_content(self, file_path: str) -> str:
        """Load file content for analysis - handles PDF, EPUB, and text files"""
        from pathlib import Path

        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == '.pdf':
            return self._load_pdf(path)
        elif ext == '.epub':
            return self._load_epub(path)
        elif ext == '.mobi':
            return self._load_mobi(path)
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

    def summarize_file(
        self,
        file_path: str,
        depth: str = "standard",
        style: Optional[str] = None,
        chapters: Optional[List[int]] = None,
        force_refresh: bool = False,
        request_id: str = None,
        progress_callback: Optional[Callable] = None
    ):
        """
        Summarize a file (synchronous interface for interactive mode).

        Args:
            file_path: Path to file to summarize
            depth: Compression level - "quick", "standard", "detailed", "outline"
            style: Output style - "narrative", "academic", "technical", "quick"
                   If None, auto-detected from file type
            chapters: Optional list of chapter numbers to summarize
            force_refresh: Force re-processing (ignored, kept for compatibility)
            request_id: Request tracking ID
            progress_callback: Optional callback(stage, current, total)

        Returns:
            FileSummary-compatible result object
        """
        import uuid
        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        start_time = time.time()
        logger.info(f"[{request_id}] Starting summarization: {file_path}")

        # Map depth to compression level
        compression_map = {
            'quick': CompressionLevel.CONDENSED,
            'standard': CompressionLevel.STANDARD,
            'detailed': CompressionLevel.DETAILED,
            'outline': CompressionLevel.OUTLINE,
            'deep': CompressionLevel.DETAILED
        }
        compression = compression_map.get(depth, CompressionLevel.STANDARD)

        # Auto-detect or use specified style
        if style:
            style_enum = SummaryStyle[style.upper()] if style.upper() in SummaryStyle.__members__ else SummaryStyle.NARRATIVE
        else:
            style_enum = self._detect_style(file_path)

        # Run async summarization in sync context
        try:
            result = asyncio.run(self._summarize_file_async(
                file_path=file_path,
                style=style_enum,
                compression=compression,
                chapters=chapters,
                progress_callback=progress_callback
            ))

            # Update stats
            self.stats['files_summarized'] += 1
            self.stats['successful_summaries'] += 1
            self.stats['total_tokens_processed'] += result.original_tokens

            # Auto-save outputs based on config
            self._auto_save_outputs(result)

            # Convert to FileSummary-compatible format
            return self._to_file_summary(result, start_time)

        except Exception as e:
            logger.error(f"[{request_id}] Summarization failed: {e}")
            return self._create_error_summary(file_path, str(e), start_time)

    async def _summarize_file_async(
        self,
        file_path: str,
        style: SummaryStyle,
        compression: CompressionLevel,
        chapters: Optional[List[int]] = None,
        progress_callback: Optional[Callable] = None
    ) -> BookSummaryResult:
        """Internal async summarization"""
        # Load content
        content = self._load_file(file_path)

        # Pre-selected chapters to pass directly (avoids re-chunking)
        selected_chapters = None

        # If specific chapters requested, filter
        if chapters:
            structure = self.chunker.chunk_document(content, file_path)

            # First try matching by chapter number
            selected = [ch for ch in structure.chapters if ch.chapter_num in chapters]

            # If no match, use position-based selection (chapter 1 = first chapter, etc.)
            if not selected and structure.chapters:
                logger.info(f"No chapters matched by number, using position-based selection")

                # Filter out front matter (TOC, copyright, etc.) - chapters with < 500 tokens
                MIN_CHAPTER_TOKENS = 500
                content_chapters = [ch for ch in structure.chapters if ch.token_count >= MIN_CHAPTER_TOKENS]

                if content_chapters:
                    logger.info(f"Filtered to {len(content_chapters)} content chapters (>= {MIN_CHAPTER_TOKENS} tokens)")
                    max_pos = len(content_chapters)
                    selected = [content_chapters[i-1] for i in chapters if 0 < i <= max_pos]
                else:
                    # Fallback to all chapters if filtering removes everything
                    max_pos = len(structure.chapters)
                    selected = [structure.chapters[i-1] for i in chapters if 0 < i <= max_pos]

            if selected:
                logger.info(f"Selected {len(selected)} chapter(s) for summarization: {[ch.chapter_title for ch in selected]}")
                # IMPORTANT: Pass selected chapters directly instead of re-chunking
                selected_chapters = selected
            else:
                logger.warning(f"No chapters found matching {chapters}, processing full document")

        # Run summarization
        use_rolling = self.config.get('processing', {}).get('use_rolling_context', True)

        return await self.summarize_book(
            file_path=file_path,
            content=content,
            style=style,
            compression=compression,
            use_rolling_context=use_rolling,
            progress_callback=progress_callback,
            selected_chapters=selected_chapters  # Pass pre-selected chapters
        )

    def _detect_style(self, file_path: str) -> SummaryStyle:
        """Auto-detect appropriate style from file type"""
        ext = Path(file_path).suffix.lower()
        file_types = self.config.get('file_types', {})

        if ext in file_types.get('books', ['.epub', '.mobi']):
            return SummaryStyle.NARRATIVE
        elif ext in ['.pdf']:
            # Could be academic or technical - default to academic
            return SummaryStyle.ACADEMIC
        elif ext in file_types.get('data', ['.xlsx', '.xls']):
            return SummaryStyle.TECHNICAL
        else:
            style_name = self.config.get('styles', {}).get('default', 'narrative')
            return SummaryStyle[style_name.upper()]

    def _auto_save_outputs(self, result: BookSummaryResult):
        """Auto-save outputs based on config"""
        formats = self.config.get('output', {}).get('formats', ['markdown'])

        if 'markdown' in formats:
            self.save_to_markdown(result)

        if 'pdf' in formats:
            try:
                self.save_to_pdf(result)
            except Exception as e:
                logger.warning(f"PDF export failed: {e}")

        if 'epub' in formats:
            try:
                self.save_to_epub(result)
            except Exception as e:
                logger.warning(f"EPUB export failed: {e}")

        # Save to RAG if configured
        if self.config.get('integration', {}).get('save_to_rag', True):
            try:
                asyncio.run(self.save_to_rag(result))
            except Exception as e:
                logger.warning(f"RAG storage failed: {e}")

    def _to_file_summary(self, result: BookSummaryResult, start_time: float):
        """Convert BookSummaryResult to FileSummary-compatible format"""
        # Create a simple object with expected attributes
        class FileSummaryCompat:
            pass

        summary = FileSummaryCompat()
        summary.file_path = result.file_path
        summary.file_hash = ""  # Not tracked in new agent
        summary.total_tokens = result.original_tokens
        summary.chunk_count = result.total_chapters
        summary.chunks_processed = result.chapters_processed
        summary.summary = result.combined_summary
        summary.key_insights = []  # Extract from summary if needed
        summary.processing_time = time.time() - start_time
        summary.depth = "standard"
        summary.timestamp = datetime.now()
        summary.success = result.success
        summary.errors = result.errors

        return summary

    def _create_error_summary(self, file_path: str, error: str, start_time: float):
        """Create error summary for failed summarization"""
        class FileSummaryCompat:
            pass

        summary = FileSummaryCompat()
        summary.file_path = file_path
        summary.file_hash = ""
        summary.total_tokens = 0
        summary.chunk_count = 0
        summary.chunks_processed = 0
        summary.summary = ""
        summary.key_insights = []
        summary.processing_time = time.time() - start_time
        summary.depth = "standard"
        summary.timestamp = datetime.now()
        summary.success = False
        summary.errors = [error]

        return summary

    @staticmethod
    def parse_summarize_command(user_input: str) -> Dict:
        """
        Parse user summarization command.

        Examples:
            "summarize /path/to/book.epub"
            "summarize chapter 2 from /path/to/book.epub"
            "summarize /path/to/paper.pdf with academic style"
            "summarize /path/to/doc.pdf detailed"

        Returns:
            Dict with keys: file_path, chapters, style, depth
        """
        import re

        result = {
            'file_path': None,
            'chapters': None,
            'style': None,
            'depth': 'standard'
        }

        text = user_input.lower()

        # Extract chapter numbers
        chapter_match = re.search(r'chapter[s]?\s+([\d,\s-]+)', text)
        if chapter_match:
            chapter_str = chapter_match.group(1)
            chapters = []
            for part in chapter_str.replace(',', ' ').split():
                if '-' in part:
                    start, end = part.split('-')
                    chapters.extend(range(int(start), int(end) + 1))
                else:
                    chapters.append(int(part))
            result['chapters'] = chapters

        # Extract style
        for style in ['narrative', 'academic', 'technical', 'explanatory', 'quick']:
            if style in text:
                result['style'] = style
                break

        # Extract depth
        for depth in ['detailed', 'quick', 'outline', 'condensed']:
            if depth in text:
                result['depth'] = depth
                break

        # Extract file path (look for path-like strings)
        path_match = re.search(r'([/~][\w./\-_]+\.\w+)', user_input)
        if path_match:
            result['file_path'] = path_match.group(1)
            # Expand ~ to home directory
            if result['file_path'].startswith('~'):
                result['file_path'] = str(Path(result['file_path']).expanduser())

        return result

    async def summarize_book(
        self,
        file_path: str,
        content: Optional[str] = None,
        style: SummaryStyle = SummaryStyle.NARRATIVE,
        compression: CompressionLevel = CompressionLevel.STANDARD,
        compression_ratio: Optional[float] = None,  # Override if specified
        max_concurrent: int = 3,
        use_rolling_context: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        selected_chapters: Optional[List] = None  # Pre-selected chapters (skip re-chunking)
    ) -> BookSummaryResult:
        """
        Summarize a book with parallel or sequential chapter processing.

        Args:
            file_path: Path to the book file
            content: Pre-loaded content (optional, will read file if not provided)
            style: Summary style (NARRATIVE, ACADEMIC, TECHNICAL, EXPLANATORY, QUICK)
            compression: Compression level preset (DETAILED=20%, STANDARD=10%, CONDENSED=5%, OUTLINE=2%)
            compression_ratio: Override ratio if specified (0.1 = 10%)
            max_concurrent: Maximum parallel chapter summarizations (ignored if rolling_context=True)
            use_rolling_context: If True, process sequentially and pass previous summary for continuity
            progress_callback: Callback(stage, current, total) for progress updates

        Returns:
            BookSummaryResult with chapter summaries and combined output

        Example:
            # 10x reduction with parallel processing (fast, no continuity)
            result = await agent.summarize_book(
                "/path/to/book.epub",
                compression=CompressionLevel.STANDARD,
                max_concurrent=3
            )

            # 10x reduction with rolling context (slower, better continuity)
            result = await agent.summarize_book(
                "/path/to/book.epub",
                compression=CompressionLevel.STANDARD,
                use_rolling_context=True
            )
        """
        # Use compression level ratio unless explicitly overridden
        actual_ratio = compression_ratio if compression_ratio is not None else compression.ratio
        logger.info(f"Summarization: compression={compression.value} ({actual_ratio:.0%}), rolling_context={use_rolling_context}")
        start_time = time.time()
        errors = []

        # Track chunk-level progress for smoother updates
        chunk_progress = {"completed": 0, "total": 0}

        def report_progress(stage: str, current: int, total: int):
            if progress_callback:
                progress_callback(stage, current, total)
            logger.info(f"Progress: {stage} - {current}/{total}")

        def report_chunk_progress(chunks_increment: int):
            """Report progress based on chunks, not chapters (cumulative)"""
            chunk_progress["completed"] += chunks_increment
            if progress_callback and chunk_progress["total"] > 0:
                progress_callback("Summarizing", chunk_progress["completed"], chunk_progress["total"])
                logger.debug(f"Chunk progress: {chunk_progress['completed']}/{chunk_progress['total']}")

        # Create job
        job = self.ephemeral_store.create_job(file_path)
        logger.info(f"Created summarization job: {job.job_id}")

        try:
            # Use pre-selected chapters if provided, otherwise chunk the document
            if selected_chapters:
                # Use pre-selected chapters directly (preserves original chapter structure)
                report_progress("Loading", 1, 1)
                report_progress("Analyzing", 1, 1)

                chapters_to_process = selected_chapters
                total_tokens = sum(ch.token_count for ch in chapters_to_process)
                doc_title = selected_chapters[0].chapter_title.split(' (')[0] if selected_chapters else Path(file_path).stem

                logger.info(f"Using {len(chapters_to_process)} pre-selected chapters, {total_tokens} tokens")

            else:
                # Load content if not provided
                if content is None:
                    report_progress("Loading", 0, 1)
                    content = self._load_file(file_path)
                    report_progress("Loading", 1, 1)

                # Chunk document by chapters
                report_progress("Analyzing", 0, 1)
                structure = self.chunker.chunk_document(content, file_path)
                report_progress("Analyzing", 1, 1)

                chapters_to_process = structure.chapters
                total_tokens = structure.total_tokens
                doc_title = structure.title or Path(file_path).stem

                logger.info(f"Document structure: {len(chapters_to_process)} chapters, {total_tokens} tokens")

            # Calculate total chunks for progress tracking
            total_chunks = sum(len(ch.chunks) for ch in chapters_to_process)
            chunk_progress["total"] = total_chunks
            logger.info(f"Total chunks for progress: {total_chunks}")

            # Store chapters in ephemeral store
            for chapter in chapters_to_process:
                self.ephemeral_store.store_chapter(
                    job.job_id,
                    chapter.chapter_num,
                    chapter.chapter_title,
                    chapter.chunks
                )

            # Calculate summary targets based on chapters being processed
            # Create a minimal structure for target calculation
            from utilities.chapter_chunker import DocumentStructure
            temp_structure = DocumentStructure(
                doc_type='book',
                chapters=chapters_to_process,
                total_tokens=total_tokens,
                title=doc_title
            )
            targets = self.chunker.get_chapter_summary_targets(temp_structure, actual_ratio)

            # Summarize chapters - use chunk-level progress
            report_progress("Summarizing", 0, total_chunks)

            if use_rolling_context:
                # Sequential processing with context passing (better continuity)
                chapter_results = await self._summarize_chapters_sequential(
                    job=job,
                    chapters=chapters_to_process,
                    targets=targets,
                    style=style,
                    chunk_progress_callback=report_chunk_progress
                )
            else:
                # Parallel processing (faster, no continuity)
                chapter_results = await self._summarize_chapters_parallel(
                    job=job,
                    chapters=chapters_to_process,
                    targets=targets,
                    style=style,
                    max_concurrent=max_concurrent,
                    chunk_progress_callback=report_chunk_progress
                )

            # Combine chapter summaries
            report_progress("Combining", 0, 1)
            combined_summary = self._combine_chapter_summaries(
                chapter_results,
                doc_title,
                style
            )
            report_progress("Combining", 1, 1)

            # Calculate stats
            total_summary_tokens = sum(r.summary_tokens for r in chapter_results if r.success)
            actual_ratio = total_summary_tokens / total_tokens if total_tokens > 0 else 0

            # Collect errors and warnings from chapter results
            all_warnings = []
            for result in chapter_results:
                if result.error:
                    errors.append(f"Chapter {result.chapter_num}: {result.error}")
                if result.warnings:
                    all_warnings.extend(result.warnings)

            # Log warning summary
            if all_warnings:
                logger.warning(f"\n{'='*60}")
                logger.warning(f"SUMMARIZATION WARNINGS ({len(all_warnings)} issues):")
                for w in all_warnings:
                    logger.warning(f"  - {w}")
                logger.warning(f"{'='*60}\n")

            processing_time = time.time() - start_time

            # Track if user specified chapters for proper filename suffix
            user_specified = selected_chapters is not None and len(selected_chapters) > 0
            requested_nums = [c.chapter_num for c in chapters_to_process] if user_specified else None

            return BookSummaryResult(
                title=doc_title,
                file_path=file_path,
                total_chapters=len(chapters_to_process),
                chapters_processed=sum(1 for r in chapter_results if r.success),
                original_tokens=total_tokens,
                summary_tokens=total_summary_tokens,
                compression_ratio=actual_ratio,
                chapter_summaries=chapter_results,
                combined_summary=combined_summary,
                processing_time=processing_time,
                style=style,
                success=len(errors) == 0,
                errors=errors,
                user_specified_range=user_specified,
                requested_chapters=requested_nums
            )

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return BookSummaryResult(
                title=Path(file_path).stem,
                file_path=file_path,
                total_chapters=0,
                chapters_processed=0,
                original_tokens=0,
                summary_tokens=0,
                compression_ratio=0.0,
                chapter_summaries=[],
                combined_summary="",
                processing_time=time.time() - start_time,
                style=style,
                success=False,
                errors=[str(e)],
                user_specified_range=selected_chapters is not None,
                requested_chapters=None
            )

        finally:
            # Cleanup ephemeral data
            self.ephemeral_store.cleanup_job(job.job_id)

    async def _summarize_chapters_parallel(
        self,
        job: SummaryJob,
        chapters: List[ChapterChunks],
        targets: Dict[int, int],
        style: SummaryStyle,
        max_concurrent: int,
        chunk_progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[ChapterSummaryResult]:
        """
        Summarize multiple chapters in parallel.

        Args:
            job: Current summarization job
            chapters: List of chapters to summarize
            targets: Target word counts per chapter
            style: Summary style
            max_concurrent: Maximum concurrent summarizations
            chunk_progress_callback: Callback(chunks_completed) for per-chunk progress

        Returns:
            List of ChapterSummaryResult
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        chunks_completed = 0

        async def summarize_with_semaphore(chapter: ChapterChunks) -> ChapterSummaryResult:
            nonlocal chunks_completed
            async with semaphore:
                result = await self._summarize_single_chapter(
                    job=job,
                    chapter=chapter,
                    target_words=targets.get(chapter.chapter_num, 500),
                    style=style,
                    chunk_progress_callback=chunk_progress_callback
                )
                return result

        # Create tasks for all chapters
        tasks = [summarize_with_semaphore(chapter) for chapter in chapters]

        # Run all tasks concurrently (limited by semaphore)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ChapterSummaryResult(
                    chapter_num=chapters[i].chapter_num,
                    chapter_title=chapters[i].chapter_title,
                    original_tokens=chapters[i].token_count,
                    summary_tokens=0,
                    summary="",
                    processing_time=0.0,
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _summarize_chapters_sequential(
        self,
        job: SummaryJob,
        chapters: List[ChapterChunks],
        targets: Dict[int, int],
        style: SummaryStyle,
        chunk_progress_callback: Optional[Callable[[int], None]] = None
    ) -> List[ChapterSummaryResult]:
        """
        Summarize chapters sequentially with rolling context.

        Each chapter receives the previous chapter's summary for continuity.
        Slower than parallel but produces better narrative flow.

        Args:
            job: Current summarization job
            chapters: List of chapters to summarize
            targets: Target word counts per chapter
            style: Summary style
            chunk_progress_callback: Callback(chunks_completed) for per-chunk progress

        Returns:
            List of ChapterSummaryResult
        """
        results = []
        previous_summary = None

        for i, chapter in enumerate(chapters):
            is_first = (i == 0)

            result = await self._summarize_single_chapter(
                job=job,
                chapter=chapter,
                target_words=targets.get(chapter.chapter_num, 500),
                style=style,
                previous_summary=previous_summary,
                is_first_chapter=is_first,
                chunk_progress_callback=chunk_progress_callback
            )

            results.append(result)

            # Update rolling context with this chapter's summary
            if result.success:
                previous_summary = result.summary

        return results

    async def _summarize_single_chapter(
        self,
        job: SummaryJob,
        chapter: ChapterChunks,
        target_words: int,
        style: SummaryStyle,
        previous_summary: Optional[str] = None,
        is_first_chapter: bool = False,
        chunk_progress_callback: Optional[Callable[[int], None]] = None
    ) -> ChapterSummaryResult:
        """
        Summarize a single chapter.

        Args:
            job: Current job
            chapter: Chapter to summarize
            target_words: Target word count for summary
            style: Summary style
            previous_summary: Summary of previous chapter (for rolling context)
            is_first_chapter: True if this is the first chapter
            chunk_progress_callback: Callback(chunks_completed) for per-chunk progress

        Returns:
            ChapterSummaryResult
        """
        start_time = time.time()

        try:
            # Get chapter content
            content = "\n\n".join(chapter.chunks)

            # If chapter is very long, we need to summarize in stages
            content_tokens = self.token_counter.count_tokens(content)

            # Use ~50% of context for content, rest for prompt + output
            # Explanatory style gets higher limit (3200) for richer source context
            max_content_tokens = self._get_max_content_tokens(style)

            if content_tokens > max_content_tokens:  # Too long for single pass
                # Summarize chunks first, then combine
                summary = await self._hierarchical_chapter_summary(
                    chapter, target_words, style,
                    previous_summary=previous_summary,
                    is_first_chapter=is_first_chapter,
                    chunk_progress_callback=chunk_progress_callback
                )
            else:
                # Single-pass summarization - report all chunks as done
                summary = await self._generate_summary(
                    content=content,
                    chapter_num=chapter.chapter_num,
                    chapter_title=chapter.chapter_title,
                    target_words=target_words,
                    style=style,
                    previous_summary=previous_summary,
                    is_first_chapter=is_first_chapter
                )
                # Report all chunks in this chapter as processed
                if chunk_progress_callback:
                    chunk_progress_callback(len(chapter.chunks))

            summary_tokens = self.token_counter.count_tokens(summary)

            # Store summary
            self.ephemeral_store.store_chapter_summary(
                job.job_id,
                chapter.chapter_num,
                summary,
                metadata={"style": style.value, "target_words": target_words}
            )

            # Collect any warnings from this chapter
            chapter_warnings = []

            # Check for generation-level truncation warnings
            if hasattr(self, '_generation_warnings') and self._generation_warnings:
                chapter_warnings.extend(self._generation_warnings)
                self._generation_warnings = []  # Reset for next chapter

            # Run integrity check on final output
            integrity_issues = self._verify_chapter_integrity(summary, chapter.chapter_title)
            if integrity_issues:
                chapter_warnings.extend(integrity_issues)

            # Log warnings prominently
            if chapter_warnings:
                for w in chapter_warnings:
                    logger.warning(f"⚠️  Chapter {chapter.chapter_num} ({chapter.chapter_title}): {w}")

            return ChapterSummaryResult(
                chapter_num=chapter.chapter_num,
                chapter_title=chapter.chapter_title,
                original_tokens=chapter.token_count,
                summary_tokens=summary_tokens,
                summary=summary,
                processing_time=time.time() - start_time,
                success=True,
                warnings=chapter_warnings if chapter_warnings else None
            )

        except Exception as e:
            logger.error(f"Failed to summarize chapter {chapter.chapter_num}: {e}")
            return ChapterSummaryResult(
                chapter_num=chapter.chapter_num,
                chapter_title=chapter.chapter_title,
                original_tokens=chapter.token_count,
                summary_tokens=0,
                summary="",
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

    async def _hierarchical_chapter_summary(
        self,
        chapter: ChapterChunks,
        target_words: int,
        style: SummaryStyle,
        previous_summary: Optional[str] = None,
        is_first_chapter: bool = False,
        chunk_progress_callback: Optional[Callable[[int], None]] = None
    ) -> str:
        """
        Summarize a long chapter hierarchically with running context.

        Maintains a running excerpt (story-so-far) that gets passed to each
        chunk summarization, ensuring continuity and reference tracking.

        Flow:
        1. Chunk 1 → Summarize (with chapter context if available)
        2. Update running excerpt
        3. Chunk 2 → Summarize (with running excerpt)
        4. Update running excerpt
        ... repeat for all chunks
        5. Final combine pass
        """
        chunk_summaries = []
        running_excerpt = ""
        max_excerpt_tokens = 800  # Keep excerpt concise but informative

        # Summarize each chunk with running context
        for i, chunk in enumerate(chapter.chunks):
            chunk_target = max(100, target_words // len(chapter.chunks))

            # Build context for this chunk
            if i == 0:
                # First chunk: use previous chapter context if available
                chunk_context = previous_summary
                chunk_is_first = is_first_chapter
            else:
                # Subsequent chunks: use running excerpt from this chapter
                chunk_context = running_excerpt
                chunk_is_first = False

            chunk_summary = await self._generate_summary(
                content=chunk,
                chapter_num=chapter.chapter_num,
                chapter_title=f"{chapter.chapter_title} (Part {i+1}/{len(chapter.chunks)})",
                target_words=chunk_target,
                style=style,
                previous_summary=chunk_context,
                is_first_chapter=chunk_is_first
            )
            chunk_summaries.append(chunk_summary)

            # Report per-chunk progress (increment by 1)
            if chunk_progress_callback:
                chunk_progress_callback(1)

            # Update running excerpt with key information from this chunk
            running_excerpt = await self._update_running_excerpt(
                current_excerpt=running_excerpt,
                new_summary=chunk_summary,
                max_tokens=max_excerpt_tokens
            )

        # Combine chunk summaries
        combined_content = "\n\n".join(chunk_summaries)

        # Check if combined content still exceeds context window
        combined_tokens = self.token_counter.count_tokens(combined_content)
        max_content_tokens = self._get_max_content_tokens(style)

        if combined_tokens > max_content_tokens:
            # Still too large - recursively summarize the chunk summaries
            logger.info(f"Combined summaries ({combined_tokens} tokens) exceed limit, doing recursive pass")

            # Calculate how many groups needed to fit each group under context limit
            # Start with minimum needed, then verify and increase if necessary
            import math
            num_groups = max(2, math.ceil(combined_tokens / max_content_tokens))

            # Keep increasing groups until each group fits
            max_attempts = 20  # Safety limit
            for attempt in range(max_attempts):
                group_size = max(1, len(chunk_summaries) // num_groups)
                summary_groups = [
                    chunk_summaries[i:i + group_size]
                    for i in range(0, len(chunk_summaries), group_size)
                ]

                # Verify all groups fit in context
                all_fit = True
                for group in summary_groups:
                    group_content = "\n\n".join(group)
                    group_tokens = self.token_counter.count_tokens(group_content)
                    if group_tokens > max_content_tokens:
                        all_fit = False
                        break

                if all_fit:
                    logger.info(f"Split into {len(summary_groups)} groups (attempt {attempt + 1})")
                    break
                else:
                    # Need more groups
                    num_groups += 1
                    logger.debug(f"Groups too large, trying {num_groups} groups")

            # Now summarize each group
            group_summaries = []
            for group_idx, group in enumerate(summary_groups):
                group_content = "\n\n".join(group)

                group_summary = await self._generate_summary(
                    content=group_content,
                    chapter_num=chapter.chapter_num,
                    chapter_title=f"{chapter.chapter_title} (Summary {group_idx + 1}/{len(summary_groups)})",
                    target_words=max(50, target_words // len(summary_groups)),
                    style=style
                )
                group_summaries.append(group_summary)

            # Now combine the group summaries for final pass
            combined_content = "\n\n".join(group_summaries)

            # Check if we need another recursive pass (for extremely large chapters)
            final_tokens = self.token_counter.count_tokens(combined_content)
            if final_tokens > max_content_tokens:
                logger.warning(f"Group summaries still too large ({final_tokens} tokens), truncating to {max_content_tokens} tokens")
                if not hasattr(self, '_generation_warnings'):
                    self._generation_warnings = []
                self._generation_warnings.append(
                    f"Chapter content truncated from {final_tokens} to {max_content_tokens} tokens in combining step"
                )
                # Last resort: truncate to fit, but at a sentence boundary
                max_chars = int(max_content_tokens * 3.5)
                truncated = combined_content[:max_chars]
                # Find last sentence-ending punctuation to avoid mid-word cuts
                last_sentence_end = max(
                    truncated.rfind('. '),
                    truncated.rfind('.\n'),
                    truncated.rfind('."'),
                    truncated.rfind('.\u201D'),
                    truncated.rfind('! '),
                    truncated.rfind('? '),
                )
                if last_sentence_end > max_chars * 0.7:  # Only if we keep at least 70%
                    combined_content = truncated[:last_sentence_end + 1]
                else:
                    combined_content = truncated

        # Final summary pass with full context
        final_summary = await self._generate_summary(
            content=combined_content,
            chapter_num=chapter.chapter_num,
            chapter_title=chapter.chapter_title,
            target_words=target_words,
            style=style
        )

        return final_summary

    async def _update_running_excerpt(
        self,
        current_excerpt: str,
        new_summary: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Update the running excerpt with new summary content.

        Keeps track of key characters, events, and plot points mentioned so far.
        Uses LLM to intelligently merge and condense if needed.
        """
        if not current_excerpt:
            # First chunk - just use the summary, truncated if needed
            excerpt_tokens = self.token_counter.count_tokens(new_summary)
            if excerpt_tokens <= max_tokens:
                return new_summary
            # Truncate to fit
            return new_summary[:int(max_tokens * 3.5)]  # ~3.5 chars per token

        # Combine and check size
        combined = f"{current_excerpt}\n\n{new_summary}"
        combined_tokens = self.token_counter.count_tokens(combined)

        if combined_tokens <= max_tokens:
            return combined

        # Need to condense - use LLM to merge excerpts intelligently
        condense_prompt = f"""Condense these story excerpts into a single brief summary (max {max_tokens // 2} words).
Keep track of: main characters, key events, important relationships, unresolved plot points.
Remove redundancy but preserve all important narrative elements.

EXCERPTS:
{combined}

Write a condensed excerpt that captures all essential story elements:"""

        loop = asyncio.get_event_loop()
        condensed = await loop.run_in_executor(
            self._executor,
            lambda: self._sync_generate(condense_prompt, max_tokens=max_tokens)
        )

        return condensed.strip()

    async def _generate_summary(
        self,
        content: str,
        chapter_num: int,
        chapter_title: str,
        target_words: int,
        style: SummaryStyle,
        previous_summary: Optional[str] = None,
        is_first_chapter: bool = False
    ) -> str:
        """
        Generate summary using LLM.

        Runs synchronous LLM call in thread pool for async compatibility.

        Args:
            content: Chapter content to summarize
            chapter_num: Chapter number
            chapter_title: Chapter title
            target_words: Target word count (minimum)
            style: Summary style
            previous_summary: Summary of previous chapter (for continuity)
            is_first_chapter: True if this is the first chapter
        """
        # Detect language and get prompt template
        language = self._detect_language(content)
        if language == 'ru' and style in NARRATIVE_PROMPTS_RU:
            prompt_template = NARRATIVE_PROMPTS_RU[style]
            ctx_templates = CONTEXT_TEMPLATES_RU
        else:
            prompt_template = NARRATIVE_PROMPTS.get(style, NARRATIVE_PROMPTS[SummaryStyle.NARRATIVE])
            ctx_templates = CONTEXT_TEMPLATES

        # Build context note for continuity
        if is_first_chapter:
            if style == SummaryStyle.EXPLANATORY:
                context_note = ("This is the first chapter - establish the key concepts and foundational ideas. "
                               "Explain any technical terms or formulas introduced here thoroughly, as they may be "
                               "referenced in later chapters.")
            else:
                context_note = ctx_templates["first_chapter"]
        elif previous_summary:
            # Truncate previous summary if too long - EXPLANATORY gets more context
            max_prev_len = 800 if style == SummaryStyle.EXPLANATORY else 500
            prev_truncated = previous_summary[:max_prev_len] + "..." if len(previous_summary) > max_prev_len else previous_summary
            if style == SummaryStyle.EXPLANATORY:
                context_note = f"""PREVIOUSLY COVERED (remind the reader of key concepts they may have forgotten):
{prev_truncated}

When referencing concepts from previous chapters, briefly remind the reader what they mean.
For example: "Building on the wave-particle duality discussed earlier (where we saw that light behaves as both waves and particles)..."
This helps readers who may have read previous chapters weeks ago."""
            else:
                context_note = ctx_templates["with_previous"].format(previous_summary=prev_truncated)
        else:
            context_note = ctx_templates["standalone"]

        # Calculate min/max words and compression guidance based on target and style
        # This helps enforce different compression levels properly

        # EXPLANATORY mode gets special treatment - prioritize clarity over brevity
        if style == SummaryStyle.EXPLANATORY:
            # Allow 25-50% more words for thorough explanations
            min_words = target_words
            max_words = int(target_words * 1.5)  # Up to 50% more for complex explanations
            compression_guidance = ("PRIORITIZE CLARITY over brevity. Take the space needed to explain complex concepts, "
                                   "examples, and formulas thoroughly. It's better to be longer and clear than short and confusing. "
                                   "Make sure every difficult idea is properly explained.")
        elif target_words <= 150:  # Very condensed (outline level)
            min_words = max(50, target_words - 50)
            max_words = target_words + 50
            compression_guidance = "Be EXTREMELY brief - only the most essential points. Aim for maximum compression."
        elif target_words <= 300:  # Condensed
            min_words = max(100, target_words - 100)
            max_words = target_words + 100
            compression_guidance = "Be concise - cover only key points and main arguments. Skip supporting details."
        elif target_words <= 600:  # Standard
            min_words = target_words - 150
            max_words = target_words + 200
            compression_guidance = "Balance detail and brevity - cover main themes with some supporting detail."
        else:  # Detailed
            min_words = target_words - 200
            max_words = target_words + 300
            if style == SummaryStyle.NARRATIVE:
                compression_guidance = ("Include good detail - preserve the author's voice, key descriptions, atmosphere, "
                                       "and emotional beats. Keep evocative passages that define the tone.")
            else:
                compression_guidance = "Include good detail - cover arguments, evidence, and examples thoroughly."

        # Format prompt with all parameters
        prompt = prompt_template.format(
            chapter_num=chapter_num,
            chapter_title=chapter_title,
            content=content,
            target_words=target_words,
            min_words=min_words,
            max_words=max_words,
            compression_guidance=compression_guidance,
            context_note=context_note
        )

        # Calculate max_tokens - need enough room for target + buffer
        # Tokens are roughly 0.75 words, so multiply by 1.5 and add generous buffer
        # EXPLANATORY mode gets higher cap for thorough explanations
        if style == SummaryStyle.EXPLANATORY:
            # Allow up to 6144 tokens for explanatory mode
            # Higher multiplier (4x) because chat API models use tokens for
            # internal thinking (~800) plus produce richer explanatory output
            max_tokens = min(6144, max(1500, int(target_words * 4.0)))
        else:
            # Cap at 6144 to prevent truncation (reasoning models use ~800-1200 tokens
            # for thinking; wider context (16K) allows generous output budget)
            max_tokens = min(6144, max(1500, int(target_words * 4)))

        # Run LLM in thread pool (since llama-cpp-python is sync)
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            self._executor,
            lambda: self._sync_generate(prompt, max_tokens)
        )

        return summary.strip()

    def _sync_generate(self, prompt: str, max_tokens: int = 4096) -> str:
        """Synchronous LLM generation (called from thread pool)

        Enforces minimum max_tokens of 1200 to ensure chat models have room
        for actual content after internal reasoning/thinking tokens.
        Retries once with doubled max_tokens if response is empty.
        """
        import uuid
        import re

        # Some chat-finetuned models may use ~800 tokens for internal
        # thinking before producing content. Enforce minimum to avoid empty responses.
        # Reasoning models use ~800-1200 tokens for internal thinking.
        # Increase budget to ensure actual content is not truncated.
        effective_max_tokens = max(3072, max_tokens)

        request_id = f"sum_{uuid.uuid4().hex[:8]}"
        # Use very long timeout for summarization - large chapters can take 10+ minutes
        # Pass timeout=600 (10 min) instead of default 120s
        response = self.generate_with_logging(prompt, request_id=request_id, max_tokens=effective_max_tokens, timeout=600)

        # Robust handling of <think> tags for reasoning models
        # Models may use think tags in various ways:
        # 1. Wrap ALL output in think tags (nemotron-sunfall)
        # 2. Use think tags for reasoning, then output separately
        # 3. Have orphaned/unclosed tags
        # 4. Mix think content with regular content

        cleaned = self._clean_think_tags(response)
        cleaned = self._strip_plaintext_reasoning(cleaned)

        # Check if generation was truncated (finish_reason: length)
        if hasattr(self, '_last_finish_reason') and self._last_finish_reason == 'length':
            self.logger.warning(
                f"[{request_id}] TRUNCATION: Generation hit token limit "
                f"(max_tokens={effective_max_tokens}). Output may be incomplete."
            )
            if not hasattr(self, '_generation_warnings'):
                self._generation_warnings = []
            self._generation_warnings.append(
                f"Generation truncated at {effective_max_tokens} tokens"
            )

        # Retry once if empty (model may have used all tokens for thinking)
        if not cleaned.strip() and effective_max_tokens < 8192:
            retry_tokens = min(8192, effective_max_tokens * 2)
            self.logger.warning(f"[{request_id}] Empty response, retrying with max_tokens={retry_tokens}")
            retry_id = f"sum_{uuid.uuid4().hex[:8]}"
            response = self.generate_with_logging(prompt, request_id=retry_id, max_tokens=retry_tokens, timeout=600)
            cleaned = self._clean_think_tags(response)
            cleaned = self._strip_plaintext_reasoning(cleaned)

        return cleaned.strip()



    def _verify_chapter_integrity(self, text: str, chapter_title: str) -> List[str]:
        """Check if a chapter summary appears complete and flag potential issues.

        Returns a list of warning strings (empty = all good).
        """
        warnings = []
        if not text or len(text.strip()) < 50:
            warnings.append(f"Chapter '{chapter_title}': Summary too short ({len(text.strip())} chars)")
            return warnings

        stripped = text.strip()

        # Check 1: Does it end with proper punctuation?
        proper_endings = set('.!?"”»—)’]\'`*')
        last_char = stripped[-1]
        if last_char not in proper_endings:
            warnings.append(
                f"Chapter '{chapter_title}': May be truncated "
                f"(ends with '{stripped[-30:]}', last char U+{ord(last_char):04X})"
            )

        # Check 2: Suspiciously short for a chapter?
        word_count = len(stripped.split())
        if word_count < 200:
            warnings.append(
                f"Chapter '{chapter_title}': Unusually short ({word_count} words)"
            )

        return warnings

    def _strip_plaintext_reasoning(self, text: str) -> str:
        """
        Strip plain-text reasoning that reasoning models output
        without <think> tags. These appear as planning/drafting text like:
        "We need to write the summary...", "Must not include...", "Let's aim..."

        Strategy: detect contiguous reasoning blocks and remove them,
        keeping only the actual summary content.
        """
        import re
        if not text or len(text) < 100:
            return text

        # Reasoning indicators - lines that are clearly model planning, not content
        reasoning_starts = [
            r'^we need to\b', r'^we must\b', r'^we can\b', r'^we have to\b',
            r"^we'll\b", r"^let's\b", r'^must not\b', r'^we should\b',
            r'^our current draft\b', r'^draft:', r'^ensure no\b',
            r'^the user\b.*says', r'^we don.t know\b', r'^we might\b',
            r'^so we need\b', r'^safest is to\b', r'^but we\b.*don.t know',
            r'^we.ll count\b', r'^we.ll produce\b', r'^we.ll just\b',
            r'^we.ll keep\b', r'^we.ll then\b',
        ]
        reasoning_pattern = re.compile('|'.join(reasoning_starts), re.IGNORECASE)

        # Content indicators - lines that are clearly actual content
        content_starts = [
            r'^#{1,4}\s',        # Markdown headers
            r'^\*\*',            # Bold text
            r'^[-*]\s',          # Bullet points
            r'^\d+\.\s',       # Numbered lists
            r'^\|',              # Table rows
            r'^>\s',             # Blockquotes
            r'^```',              # Code blocks
        ]
        content_pattern = re.compile('|'.join(content_starts))

        lines = text.split('\n')

        # Strip leading reasoning block
        content_start_idx = 0
        consecutive_reasoning = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if reasoning_pattern.search(stripped):
                consecutive_reasoning += 1
                content_start_idx = i + 1
            elif content_pattern.search(stripped):
                # Hit actual content - stop stripping
                break
            elif consecutive_reasoning >= 2:
                # We've seen 2+ reasoning lines; keep going until we hit content
                # (reasoning blocks often have non-matching planning lines mixed in)
                if any(kw in stripped.lower() for kw in [
                    'word count', 'words.', 'approximately', 'forbidden',
                    'token', 'aim for', 'produce a', 'current draft',
                    'count after', 'not add new', 'not invent',
                ]):
                    content_start_idx = i + 1
                else:
                    break
            else:
                break

        if content_start_idx > 0:
            stripped_count = content_start_idx
            remaining = '\n'.join(lines[content_start_idx:]).strip()
            if remaining and len(remaining) > 50:
                logger.info(f"Stripped {stripped_count} lines of plain-text reasoning from start")
                text = remaining
            # If stripping would leave too little, keep original
            # (the whole response might be reasoning-as-content)

        # Remove isolated reasoning lines within the text
        # (be conservative - only strip obvious planning text mid-content)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip pure reasoning/planning lines embedded in content
            if stripped and reasoning_pattern.search(stripped):
                # But only if it looks like planning, not content that happens
                # to start with "We need" (e.g., "We need better security")
                meta_keywords = ['word', 'draft', 'summary', 'produce', 'forbidden',
                                'prefix', 'ellipsis', 'aim for', 'count', 'token',
                                'not include', 'not add', 'not invent', 'must not']
                if any(kw in stripped.lower() for kw in meta_keywords):
                    logger.debug(f"Stripped mid-content reasoning: {stripped[:60]}...")
                    continue
            cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines).strip()
        return result if result else text

    def _clean_think_tags(self, text: str) -> str:
        """
        Robustly extract useful content from text that may contain think tags.

        Strategy:
        1. Remove all complete <think>...</think> blocks and check remainder
        2. If nothing left, extract content from INSIDE think tags
        3. Handle orphaned opening/closing tags
        4. Clean up any remaining tag fragments
        """
        import re

        if not text:
            return ""

        original_len = len(text)

        # Step 1: Try to get content OUTSIDE all think tags
        # Handle multiple think blocks
        content_outside = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        if content_outside and len(content_outside) > 50:  # Meaningful content outside
            # Clean up any remaining tag fragments
            content_outside = self._remove_tag_fragments(content_outside)
            if content_outside:
                return content_outside

        # Step 2: No meaningful content outside - extract from INSIDE think tags
        # Get all think tag contents
        think_matches = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        if think_matches:
            # Combine all think content
            combined_think = "\n\n".join(m.strip() for m in think_matches if m.strip())
            if combined_think:
                logger.debug(f"Extracted {len(combined_think)} chars from {len(think_matches)} <think> tag(s)")
                return self._remove_tag_fragments(combined_think)

        # Step 3: Handle unclosed <think> at start (model started thinking, never closed)
        unclosed_start = re.search(r'^<think>(.*)', text, flags=re.DOTALL)
        if unclosed_start:
            partial = unclosed_start.group(1).strip()
            # Remove any closing think tag at the end
            partial = re.sub(r'</think>\s*$', '', partial).strip()
            if partial:
                logger.debug(f"Extracted {len(partial)} chars from unclosed <think> tag")
                return self._remove_tag_fragments(partial)

        # Step 4: Handle orphaned </think> at the start (incomplete streaming)
        if text.startswith('</think>'):
            cleaned = text.replace('</think>', '', 1).strip()
            if cleaned:
                return self._remove_tag_fragments(cleaned)

        # Step 5: Fallback - clean any remaining fragments and return
        cleaned = self._remove_tag_fragments(text)
        return cleaned if cleaned else text.strip()

    def _remove_tag_fragments(self, text: str) -> str:
        """Remove orphaned think tag fragments from text."""
        import re

        # Remove orphaned opening tags
        text = re.sub(r'<think>\s*', '', text)
        # Remove orphaned closing tags
        text = re.sub(r'\s*</think>', '', text)
        # Remove any other think-like patterns that might have slipped through
        text = re.sub(r'</?think/?>', '', text)

        return text.strip()

    def _combine_chapter_summaries(
        self,
        chapter_results: List[ChapterSummaryResult],
        title: str,
        style: SummaryStyle
    ) -> str:
        """
        Combine chapter summaries into final document.

        Args:
            chapter_results: List of chapter summary results
            title: Document title
            style: Summary style

        Returns:
            Combined summary document
        """
        # Sort by chapter number
        sorted_results = sorted(
            [r for r in chapter_results if r.success],
            key=lambda r: r.chapter_num
        )

        if not sorted_results:
            return "No chapters were successfully summarized."

        # Build combined document
        parts = []

        if style == SummaryStyle.NARRATIVE:
            # Narrative style: flowing prose with chapter breaks
            parts.append(f"# {title}\n")
            parts.append("*Condensed Narrative*\n")
            parts.append("---\n")

            for result in sorted_results:
                parts.append(f"\n## Chapter {result.chapter_num}: {result.chapter_title}\n")
                parts.append(result.summary)
                parts.append("\n")

        elif style == SummaryStyle.ACADEMIC:
            # Academic style: structured with sections
            parts.append(f"# Summary: {title}\n")

            for result in sorted_results:
                parts.append(f"\n### {result.chapter_num}. {result.chapter_title}\n")
                parts.append(result.summary)

        elif style == SummaryStyle.TECHNICAL:
            # Technical style: documentation format
            parts.append(f"# {title} - Technical Summary\n")

            for result in sorted_results:
                parts.append(f"\n## {result.chapter_num}. {result.chapter_title}\n")
                parts.append(result.summary)

        elif style == SummaryStyle.EXPLANATORY:
            # Explanatory style: educational with clear explanations
            parts.append(f"# {title}\n")
            parts.append("*Explanatory Summary - Complex concepts explained clearly*\n")
            parts.append("---\n")

            for result in sorted_results:
                parts.append(f"\n## Chapter {result.chapter_num}: {result.chapter_title}\n")
                parts.append(result.summary)
                parts.append("\n")

        else:
            # Quick style: bullet points
            parts.append(f"# {title} - Quick Summary\n")

            for result in sorted_results:
                parts.append(f"\n**Chapter {result.chapter_num}: {result.chapter_title}**\n")
                parts.append(result.summary)

        return "\n".join(parts)

    def get_chapter_info(self, file_path: str) -> Dict[str, Any]:
        """Get chapter information without summarizing.

        Args:
            file_path: Path to document

        Returns:
            Dict with:
                - chapters: List of {num, title, tokens}
                - total_chapters: int
                - total_tokens: int
                - file_type: str
        """
        from pathlib import Path
        MIN_CHAPTER_TOKENS = 500

        path = Path(file_path)
        if not path.exists():
            return {'error': f"File not found: {file_path}", 'chapters': []}

        try:
            content = self._load_file(file_path)
            structure = self.chunker.chunk_document(content, file_path)

            # Filter content chapters (skip TOC, front matter)
            all_chapters = structure.chapters
            content_chapters = [ch for ch in all_chapters if ch.token_count >= MIN_CHAPTER_TOKENS]

            chapter_info = []
            for i, ch in enumerate(content_chapters):
                chapter_info.append({
                    'num': ch.chapter_num if ch.chapter_num else i + 1,
                    'title': ch.chapter_title or f"Chapter {i + 1}",
                    'tokens': ch.token_count,
                    'position': i + 1  # 1-indexed position in content chapters
                })

            return {
                'chapters': chapter_info,
                'total_chapters': len(content_chapters),
                'skipped_chapters': len(all_chapters) - len(content_chapters),
                'total_tokens': sum(ch.token_count for ch in content_chapters),
                'file_type': path.suffix.lower()
            }
        except Exception as e:
            logger.error(f"Failed to get chapter info: {e}")
            return {'error': str(e), 'chapters': []}

    def _detect_language(self, content: str) -> str:
        """Detect document language from content sample.

        Returns 'ru' for Russian (Cyrillic-dominant), 'en' otherwise.
        """
        import re
        sample = content[:10000]
        cyrillic = len(re.findall(r'[а-яёА-ЯЁ]', sample))
        latin = len(re.findall(r'[a-zA-Z]', sample))
        total = cyrillic + latin
        if total == 0:
            return 'en'
        return 'ru' if cyrillic / total > 0.3 else 'en'

    def _load_file(self, file_path: str) -> str:
        """Load file content with format detection

        Supported formats:
        - EPUB (.epub) - via ebooklib
        - MOBI (.mobi) - via mobi library
        - PDF (.pdf) - via PyPDF2
        - DOCX (.docx) - via python-docx
        - FB2 (.fb2) - via xml.etree (FictionBook2)
        - Excel (.xlsx, .xls) - via openpyxl/xlrd
        - Text (.txt, .md)
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == '.epub':
            return self._load_epub(path)
        elif ext == '.mobi':
            return self._load_mobi(path)
        elif ext == '.pdf':
            return self._load_pdf(path)
        elif ext == '.docx':
            return self._load_docx(path)
        elif ext == '.fb2':
            return self._load_fb2(path)
        elif ext in ['.xlsx', '.xls']:
            return self._load_excel(path)
        elif ext in ['.txt', '.md']:
            return path.read_text(encoding='utf-8')
        else:
            # Try as text
            return path.read_text(encoding='utf-8')

    def _load_epub(self, path: Path) -> str:
        """Load EPUB file content with structure-aware extraction.

        Properly handles:
        - Multi-book compilations (trilogies, etc.)
        - Chapter boundaries from EPUB spine/NCX structure
        - Title extraction from EPUB metadata
        - Spine order (correct reading sequence)
        """
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            import re

            # Read EPUB with NCX for table of contents
            book = epub.read_epub(str(path))

            # Extract title from metadata
            title = None
            if book.get_metadata('DC', 'title'):
                title = book.get_metadata('DC', 'title')[0][0]

            # Build item ID to item mapping
            items_by_id = {}
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    items_by_id[item.get_id()] = item

            # Get spine order (reading sequence)
            spine_ids = [item[0] for item in book.spine]

            # Try to get TOC for chapter names
            toc_map = {}  # Maps item href to chapter title
            try:
                def extract_toc(toc_items, depth=0):
                    for item in toc_items:
                        if isinstance(item, tuple):
                            # Nested TOC
                            section, children = item
                            if hasattr(section, 'href') and hasattr(section, 'title'):
                                href = section.href.split('#')[0]  # Remove anchor
                                toc_map[href] = section.title
                            extract_toc(children, depth + 1)
                        elif hasattr(item, 'href') and hasattr(item, 'title'):
                            href = item.href.split('#')[0]
                            toc_map[href] = item.title

                extract_toc(book.toc)
            except Exception as e:
                logger.debug(f"Could not extract TOC: {e}")

            text_parts = []
            chapter_num = 0

            # Process items in spine order
            for spine_id in spine_ids:
                if spine_id not in items_by_id:
                    continue

                item = items_by_id[spine_id]

                try:
                    content = item.get_content()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='ignore')

                    soup = BeautifulSoup(content, 'html.parser')

                    # Remove non-content elements
                    for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                        tag.decompose()

                    # Check if this document has chapter markers (h1, h2)
                    # This helps detect chapter boundaries within EPUB items
                    chapter_headers = soup.find_all(['h1', 'h2'])

                    # Get chapter title from TOC or headers
                    item_href = item.get_name()
                    chapter_title = toc_map.get(item_href)

                    if not chapter_title and chapter_headers:
                        # Use first heading as chapter title
                        chapter_title = chapter_headers[0].get_text(strip=True)

                    # Extract text preserving some structure
                    text = soup.get_text(separator='\n', strip=True)
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    text = '\n'.join(lines)

                    # Skip very short fragments (navigation, copyright, etc.)
                    if len(text) < 100:
                        continue

                    # Skip items that look like front/back matter
                    text_lower = text.lower()[:500]
                    skip_patterns = ['table of contents', 'copyright', 'all rights reserved',
                                   'about the author', 'also by', 'other books by']
                    if any(p in text_lower for p in skip_patterns) and len(text) < 2000:
                        continue

                    # For LARGE items (>50KB), check for internal chapter/section markers
                    # This handles EPUBs where multiple chapters are combined (like Three Body Problem trilogy)
                    if len(text) > 50000:
                        # Pattern to split on era-based chapter markers (Three Body Problem style)
                        # Matches: "Year 3, Crisis Era", "Deterrence Era, Year 12", etc.
                        era_pattern = r'(?:^|\n\s*)((?:Year \d+,?\s*)?(?:Crisis|Deterrence|Broadcast|Bunker|Post-Deterrence|Galaxy|Common)\s+Era(?:,?\s*Year \d+)?[^\n]*)'

                        # Try era-based splitting first (most effective for this book)
                        parts = re.split(era_pattern, text, flags=re.IGNORECASE)

                        if len(parts) > 2:  # Successfully split
                            # re.split with capturing group returns: [before, match1, after1, match2, after2, ...]
                            # Reconstruct: pair each header with its following content
                            reconstructed = []

                            # First part (before any era marker)
                            if parts[0].strip() and len(parts[0].strip()) > 500:
                                reconstructed.append(parts[0].strip())

                            # Pair headers with content: parts[1]=header1, parts[2]=content1, parts[3]=header2, etc.
                            for i in range(1, len(parts) - 1, 2):
                                header = parts[i].strip() if i < len(parts) else ""
                                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                                if header and content:
                                    combined = f"{header}\n\n{content}"
                                    if len(combined) > 500:
                                        reconstructed.append(combined)
                                elif header and len(header) > 500:
                                    reconstructed.append(header)

                            if len(reconstructed) > 1:
                                logger.info(f"Split large EPUB item on era markers into {len(reconstructed)} sections")
                                for split_text in reconstructed:
                                    chapter_num += 1
                                    first_line = split_text.split('\n')[0][:100]
                                    text_parts.append(f"Chapter {chapter_num}: {first_line}\n\n{split_text}")
                                continue

                        # Fallback: try other chapter patterns
                        fallback_patterns = [
                            r'(?:^|\n\s*)(Part\s+[IVX]+[:\s][^\n]+)',  # Part I: Title
                            r'(?:^|\n\s*)(Chapter\s+\d+[:\s][^\n]+)',  # Chapter 1: Title
                            r'(?:^|\n\s*)(\d{3,4}\s*(?:C\.E\.|A\.D\.|BCE)[^\n]*)',  # Historical dates
                        ]

                        for pattern in fallback_patterns:
                            parts = re.split(pattern, text, flags=re.IGNORECASE)
                            if len(parts) > 2:
                                reconstructed = []
                                if parts[0].strip() and len(parts[0].strip()) > 500:
                                    reconstructed.append(parts[0].strip())
                                for i in range(1, len(parts) - 1, 2):
                                    header = parts[i].strip() if i < len(parts) else ""
                                    content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                                    if header and content:
                                        combined = f"{header}\n\n{content}"
                                        if len(combined) > 500:
                                            reconstructed.append(combined)

                                if len(reconstructed) > 1:
                                    logger.info(f"Split large EPUB item with fallback pattern into {len(reconstructed)} sections")
                                    for split_text in reconstructed:
                                        chapter_num += 1
                                        first_line = split_text.split('\n')[0][:100]
                                        text_parts.append(f"Chapter {chapter_num}: {first_line}\n\n{split_text}")
                                    break  # Exit pattern loop on success
                        else:
                            # No pattern worked - split on line breaks for very large items
                            # This handles EPUBs with continuous narrative (no chapter markers)
                            if len(text) > 80000:  # ~20K tokens
                                # Try double newlines first, fall back to single newlines
                                paragraphs = re.split(r'\n{2,}', text)
                                if len(paragraphs) < 3:
                                    # No double newlines - split on single newlines
                                    paragraphs = text.split('\n')

                                # Group paragraphs into ~30K char sections
                                MAX_SECTION_SIZE = 30000
                                sections = []
                                current_section = []
                                current_size = 0

                                for para in paragraphs:
                                    para = para.strip()
                                    if not para:
                                        continue
                                    if current_size + len(para) > MAX_SECTION_SIZE and current_section:
                                        sections.append('\n\n'.join(current_section))
                                        current_section = [para]
                                        current_size = len(para)
                                    else:
                                        current_section.append(para)
                                        current_size += len(para)

                                if current_section:
                                    sections.append('\n\n'.join(current_section))

                                if len(sections) > 1:
                                    logger.info(f"Split large EPUB item into {len(sections)} sections (~{MAX_SECTION_SIZE} chars each)")
                                    for i, section in enumerate(sections):
                                        if len(section) > 500:
                                            chapter_num += 1
                                            # Use first line or generate title
                                            first_line = section.split('\n')[0][:80]
                                            if len(first_line) < 10:
                                                first_line = f"Section {i+1}"
                                            text_parts.append(f"Chapter {chapter_num}: {first_line}\n\n{section}")
                                    continue  # Skip normal processing

                    # Insert chapter marker if we have a chapter title
                    # This helps ChapterAwareChunker detect boundaries
                    if chapter_title and len(text) > 500:
                        # Clean chapter title
                        chapter_title = chapter_title.strip()

                        # Check if chapter already starts with "Chapter X" marker
                        if not re.match(r'^(Chapter|CHAPTER|Part|PART)\s+', text[:50]):
                            chapter_num += 1
                            # Insert normalized chapter marker
                            text = f"Chapter {chapter_num}: {chapter_title}\n\n{text}"

                    text_parts.append(text)

                except Exception as e:
                    logger.warning(f"Failed to extract EPUB item {item.get_id()}: {e}")
                    continue

            if not text_parts:
                raise ValueError(f"No readable content found in EPUB: {path.name}")

            # Join with clear separation
            result = "\n\n---\n\n".join(text_parts)

            # Prepend title if found
            if title:
                result = f"{title}\n\n{result}"

            logger.info(f"EPUB loaded: {len(text_parts)} sections, {len(result)} chars, title='{title}'")
            return result

        except ImportError:
            raise ValueError("EPUB support requires ebooklib: pip install ebooklib beautifulsoup4")
        except Exception as e:
            logger.error(f"EPUB loading failed: {e}")
            raise ValueError(f"Failed to load EPUB: {e}")

    def _load_pdf(self, path: Path) -> str:
        """Load PDF file content. Prefers pymupdf for better chapter marker
        extraction (handles spaced-out letters, OCR artifacts), falls back to PyPDF2."""
        # Try pymupdf first (better at preserving chapter headings)
        try:
            import pymupdf
            doc = pymupdf.open(str(path))
            text_parts = []
            for page in doc:
                text = page.get_text()
                if text:
                    text_parts.append(text)
            doc.close()
            logger.info(f"PDF extracted with pymupdf: {len(text_parts)} pages")
            return "\n\n".join(text_parts)
        except ImportError:
            pass

        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            with open(path, 'rb') as f:
                reader = PdfReader(f)
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            logger.info(f"PDF extracted with PyPDF2: {len(text_parts)} pages")
            return "\n\n".join(text_parts)
        except ImportError:
            raise ValueError("PDF support requires pymupdf or PyPDF2")

    def _load_mobi(self, path: Path) -> str:
        """Load MOBI file content"""
        try:
            import mobi

            # Extract MOBI to temp directory and read
            tempdir, filepath = mobi.extract(str(path))
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Clean up HTML if present
            if '<html' in content.lower():
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                content = soup.get_text(separator='\n', strip=True)

            return content

        except ImportError:
            raise ValueError("MOBI support requires mobi: pip install mobi")

    def _load_fb2(self, path: Path) -> str:
        """Load FB2 (FictionBook2) file content.

        FB2 is an XML-based ebook format popular in Russian-speaking countries.
        Structure: <FictionBook> → <body> → <section> (chapters) → <title> + <p>
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(str(path))
        root = tree.getroot()

        # Handle FB2 XML namespace
        ns_match = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
        ns = {'fb': ns_match.strip('{}')} if ns_match else {}

        def find(el, tag):
            """Find element with or without namespace."""
            if ns:
                result = el.find(f'fb:{tag}', ns)
                if result is not None:
                    return result
            return el.find(tag)

        def findall(el, tag):
            if ns:
                result = el.findall(f'fb:{tag}', ns)
                if result:
                    return result
            return el.findall(tag)

        # Extract title
        title_el = root.find(f'.//{ns_match}book-title') if ns_match else root.find('.//book-title')
        book_title = title_el.text if title_el is not None else path.stem

        # Find body element
        body = root.find(f'.//{ns_match}body') if ns_match else root.find('.//body')
        if body is None:
            raise ValueError(f"No <body> found in FB2: {path.name}")

        text_parts = []

        def extract_section(section, depth=0):
            """Recursively extract sections, handling nested structure."""
            # Get section title
            title_elem = find(section, 'title')
            section_title = ""
            if title_elem is not None:
                title_parts = []
                for p in findall(title_elem, 'p'):
                    t = ''.join(p.itertext()).strip()
                    if t:
                        title_parts.append(t)
                if not title_parts:
                    t = ''.join(title_elem.itertext()).strip()
                    if t:
                        title_parts.append(t)
                section_title = ' '.join(title_parts)

            # Get direct paragraphs (not in nested sections)
            paragraphs = []
            for p in findall(section, 'p'):
                text = ''.join(p.itertext()).strip()
                if text:
                    paragraphs.append(text)

            # Check for nested sections
            nested = findall(section, 'section')

            if nested:
                # This is a container section (like "ОГЛАВЛЕНИЕ" with chapters inside)
                for child in nested:
                    extract_section(child, depth + 1)
            elif paragraphs:
                # Leaf section with content
                section_text = '\n\n'.join(paragraphs)
                if section_title:
                    text_parts.append(f"{section_title}\n\n{section_text}")
                else:
                    text_parts.append(section_text)

        # Process top-level sections
        for section in findall(body, 'section'):
            extract_section(section)

        if not text_parts:
            raise ValueError(f"No readable content in FB2: {path.name}")

        result = '\n\n---\n\n'.join(text_parts)
        logger.info(f"FB2 loaded: {len(text_parts)} sections, {len(result)} chars, title='{book_title}'")
        return result

    def _load_docx(self, path: Path) -> str:
        """Load DOCX file content"""
        try:
            from docx import Document

            doc = Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)

        except ImportError:
            raise ValueError("DOCX support requires python-docx: pip install python-docx")

    def _load_excel(self, path: Path) -> str:
        """Load Excel file content (converts sheets to text)"""
        try:
            ext = path.suffix.lower()

            if ext == '.xlsx':
                from openpyxl import load_workbook
                wb = load_workbook(str(path), read_only=True)
                text_parts = []

                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    text_parts.append(f"## Sheet: {sheet_name}\n")
                    for row in sheet.iter_rows(values_only=True):
                        row_text = " | ".join(str(cell) if cell else "" for cell in row)
                        if row_text.strip():
                            text_parts.append(row_text)
                    text_parts.append("")

                return "\n".join(text_parts)

            else:  # .xls
                import xlrd
                wb = xlrd.open_workbook(str(path))
                text_parts = []

                for sheet in wb.sheets():
                    text_parts.append(f"## Sheet: {sheet.name}\n")
                    for row_idx in range(sheet.nrows):
                        row = sheet.row_values(row_idx)
                        row_text = " | ".join(str(cell) if cell else "" for cell in row)
                        if row_text.strip():
                            text_parts.append(row_text)
                    text_parts.append("")

                return "\n".join(text_parts)

        except ImportError as e:
            raise ValueError(f"Excel support requires openpyxl/xlrd: {e}")

    def save_to_pdf(
        self,
        result: BookSummaryResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Save summary result to a PDF file.

        Args:
            result: BookSummaryResult from summarize_book()
            output_path: Optional output path

        Returns:
            Path to the saved PDF file
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.units import inch
        except ImportError:
            raise ValueError("PDF export requires reportlab: pip install reportlab")

        if output_path is None:
            source_name = Path(result.file_path).stem
            output_dir = Path("summaries/pdf")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_path = output_dir / f"{source_name}_summary_{timestamp}.pdf"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create PDF
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=12
        )
        chapter_style = ParagraphStyle(
            'ChapterTitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=12
        )

        story = []

        # Title
        story.append(Paragraph(f"{result.title} - Summary", title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Metadata
        meta_text = f"Source: {result.file_path}<br/>"
        meta_text += f"Style: {result.style.value}<br/>"
        meta_text += f"Compression: {result.compression_ratio:.1%}<br/>"
        meta_text += f"Chapters: {result.chapters_processed}/{result.total_chapters}"
        story.append(Paragraph(meta_text, styles['Italic']))
        story.append(Spacer(1, 0.3 * inch))

        # Chapter summaries
        for chapter_result in sorted(result.chapter_summaries, key=lambda x: x.chapter_num):
            if not chapter_result.success:
                continue

            # Chapter title
            story.append(Paragraph(
                f"Chapter {chapter_result.chapter_num}: {chapter_result.chapter_title}",
                chapter_style
            ))

            # Chapter content - split into paragraphs
            for para in chapter_result.summary.split('\n\n'):
                if para.strip():
                    # Escape special characters for reportlab
                    safe_para = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(safe_para, body_style))

        doc.build(story)
        logger.info(f"PDF saved to {output_path}")

        return str(output_path)

    def _generate_chapter_suffix(self, result: BookSummaryResult) -> str:
        """
        Generate chapter suffix for filename based on summarized chapters.

       Fix: Only use '_full' when user did NOT specify a chapter range.
        If user explicitly requested chapters 2-5, output should be 'ch2-5' even
        if the document happens to only have chapters 2-5.
        """
        if not result.chapter_summaries:
            return "full"

        # Get sorted list of successfully summarized chapter numbers
        chapter_nums = sorted([
            r.chapter_num for r in result.chapter_summaries if r.success
        ])

        if not chapter_nums:
            return "full"

        # Only use 'full' if user didn't specify a range AND all chapters processed
        if not result.user_specified_range and result.chapters_processed == result.total_chapters:
            return "full"

        # Single chapter
        if len(chapter_nums) == 1:
            return f"ch{chapter_nums[0]}"

        # Check if consecutive
        is_consecutive = all(
            chapter_nums[i] + 1 == chapter_nums[i + 1]
            for i in range(len(chapter_nums) - 1)
        )

        if is_consecutive:
            return f"ch{chapter_nums[0]}-{chapter_nums[-1]}"
        else:
            # Non-consecutive: ch1_ch3_ch5
            return "_".join(f"ch{n}" for n in chapter_nums)

    def save_to_epub(
        self,
        result: BookSummaryResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Save summary result to an EPUB file.

        Args:
            result: BookSummaryResult from summarize_book()
            output_path: Optional output path

        Returns:
            Path to the saved EPUB file
        """
        try:
            from ebooklib import epub
        except ImportError:
            raise ValueError("EPUB export requires ebooklib: pip install ebooklib")

        if output_path is None:
            source_name = Path(result.file_path).stem
            output_dir = Path("summaries/epub")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_path = output_dir / f"{source_name}_summary_{timestamp}.epub"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create EPUB book
        book = epub.EpubBook()
        book.set_identifier(f'summary-{result.file_path}-{datetime.now().isoformat()}')
        book.set_title(f"{result.title} - Summary")
        book.set_language('en')
        book.add_author('LLM-Agent-System Summarizer')

        # Add metadata
        book.add_metadata('DC', 'description', f'Summary of {result.file_path}')
        book.add_metadata('DC', 'source', result.file_path)

        # Create chapters
        chapters = []
        spine = ['nav']

        # Add summary chapter
        summary_chapter = epub.EpubHtml(title='Summary', file_name='summary.xhtml', lang='en')

        # Convert markdown to simple HTML
        html_content = f'''<html>
<head><title>{result.title} - Summary</title></head>
<body>
<h1>{result.title}</h1>
<p><em>Style: {result.style.value}</em></p>
<p><em>Original: {result.original_tokens:,} tokens | Summary: {result.summary_tokens:,} tokens | Compression: {result.compression_ratio:.1%}</em></p>
<hr/>
{self._markdown_to_html(result.combined_summary)}
</body>
</html>'''
        summary_chapter.content = html_content
        book.add_item(summary_chapter)
        chapters.append(summary_chapter)
        spine.append(summary_chapter)

        # Add navigation
        book.toc = tuple(chapters)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # Set spine
        book.spine = spine

        # Write EPUB file
        epub.write_epub(str(output_path), book)
        logger.info(f"EPUB saved to {output_path}")

        return str(output_path)

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert simple markdown to HTML"""
        import re
        html = markdown_text

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Line breaks for paragraphs
        html = re.sub(r'\n\n', '</p><p>', html)
        html = f'<p>{html}</p>'

        return html

    def save_to_markdown(
        self,
        result: BookSummaryResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Save summary result to a Markdown file.

        Args:
            result: BookSummaryResult from summarize_book()
            output_path: Optional output path. If not provided, generates
                        based on source filename in summaries/ directory.

        Returns:
            Path to the saved file
        """
        if output_path is None:
            # Generate output path with document name and chapter info
            source_name = Path(result.file_path).stem
            # Clean source name (remove special chars, lowercase, replace spaces)
            clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_name)
            clean_name = clean_name.strip("_").lower()

            chapter_suffix = self._generate_chapter_suffix(result)

            output_dir = Path("summaries/markdown")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Format: documentname_chapters_summary.md
            output_path = output_dir / f"{clean_name}_{chapter_suffix}_summary.md"

            # If file exists, add timestamp to avoid overwrite
            if output_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                output_path = output_dir / f"{clean_name}_{chapter_suffix}_summary_{timestamp}.md"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Detect language from content for metadata
        content_language = self._detect_language(result.combined_summary) if result.combined_summary else 'en'

        # Get model name from config
        model_name = "unknown"
        try:
            if hasattr(self, 'config') and self.config:
                model_name = getattr(self.config, 'model_name', None) or self.config.get('model', {}).get('name', 'unknown')
            if model_name == "unknown" and hasattr(self, 'model_config'):
                model_name = self.model_config.get('name', 'unknown')
        except Exception:
            pass

        # Build markdown content with metadata header
        content_parts = [
            "---",
            f"title: \"{result.title}\"",
            f"source: \"{result.file_path}\"",
            f"style: {result.style.value}",
            f"depth: detailed",
            f"created: {datetime.now().isoformat()}",
            f"model: {model_name}",
            f"language: {content_language}",
            f"original_tokens: {result.original_tokens:,}",
            f"summary_tokens: {result.summary_tokens:,}",
            f"compression_ratio: {result.compression_ratio:.1%}",
            f"chapters: {result.chapters_processed}/{result.total_chapters}",
        ]

        # Add warnings to frontmatter if any chapters had issues
        all_warnings = []
        if hasattr(result, 'chapter_summaries') and result.chapter_summaries:
            for ch_result in result.chapter_summaries:
                if hasattr(ch_result, 'warnings') and ch_result.warnings:
                    all_warnings.extend(ch_result.warnings)

        if all_warnings:
            content_parts.append(f"warnings: {len(all_warnings)}")
            for w in all_warnings:
                content_parts.append(f"  - \"{w}\"")

        content_parts.extend([
            "---\n",
            result.combined_summary
        ])

        content = "\n".join(content_parts)

        # Write file
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"Summary saved to {output_path}")

        return str(output_path)

    async def save_to_rag(
        self,
        result: BookSummaryResult,
        collection_name: str = "book_summaries"
    ) -> bool:
        """
        Save summary to permanent RAG storage for Knowledge Agent queries.

        Args:
            result: BookSummaryResult from summarize_book()
            collection_name: RAG collection name

        Returns:
            True if successful
        """
        try:
            # Use memory_manager passed during init, or skip if not available
            if self.memory_manager is None:
                logger.info("RAG not configured - skipping RAG storage (pass memory_manager to enable)")
                return False

            memory_mgr = self.memory_manager

            # Store each chapter summary as a separate memory entry
            for chapter_result in result.chapter_summaries:
                if not chapter_result.success:
                    continue

                metadata = {
                    "source": result.file_path,
                    "book_title": result.title,
                    "chapter_num": chapter_result.chapter_num,
                    "chapter_title": chapter_result.chapter_title,
                    "doc_type": "book_summary",
                    "style": result.style.value,
                    "original_tokens": chapter_result.original_tokens,
                    "summary_tokens": chapter_result.summary_tokens
                }

                memory_mgr.store_semantic_memory(
                    knowledge=chapter_result.summary,
                    metadata=metadata
                )

            # Also store combined summary for overview queries
            combined_metadata = {
                "source": result.file_path,
                "book_title": result.title,
                "doc_type": "book_summary_combined",
                "style": result.style.value,
                "chapters_count": result.total_chapters,
                "compression_ratio": result.compression_ratio
            }

            memory_mgr.store_semantic_memory(
                knowledge=result.combined_summary,
                metadata=combined_metadata
            )

            logger.info(f"Summary saved to RAG collection '{collection_name}'")
            return True

        except ImportError:
            logger.warning("RAG components not available - skipping RAG storage")
            return False
        except Exception as e:
            logger.error(f"Failed to save to RAG: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=False)
        self.ephemeral_store.cleanup_old_jobs()


# Convenience function for synchronous usage
def summarize_book_sync(
    model_manager,
    file_path: str,
    style: SummaryStyle = SummaryStyle.NARRATIVE,
    compression_ratio: float = 0.1,
    max_concurrent: int = 3,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> BookSummaryResult:
    """
    Synchronous wrapper for book summarization.

    Usage:
        from agents.enhanced_summarization import summarize_book_sync, SummaryStyle

        result = summarize_book_sync(
            model_manager,
            "/path/to/book.epub",
            style=SummaryStyle.NARRATIVE,
            compression_ratio=0.1
        )

        print(result.combined_summary)
    """
    agent = EnhancedSummarizationAgent(model_manager)

    try:
        # Run async function in new event loop
        result = asyncio.run(agent.summarize_book(
            file_path=file_path,
            style=style,
            compression_ratio=compression_ratio,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback
        ))
        return result

    finally:
        agent.cleanup()


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("Enhanced Summarization Agent")
    print("=" * 50)
    print("\nUsage:")
    print("  from agents.enhanced_summarization import EnhancedSummarizationAgent, SummaryStyle")
    print("  agent = EnhancedSummarizationAgent(model_manager)")
    print("  result = await agent.summarize_book('/path/to/book.epub')")
    print("\nStyles: NARRATIVE, ACADEMIC, TECHNICAL, EXPLANATORY, QUICK")
