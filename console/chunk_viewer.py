#!/usr/bin/env python3
"""
Chunk Viewer - Interactive Document Chunk Browser

Provides a terminal UI for browsing and searching stored document chunks.
Designed to run in a tmux pane alongside the Console.

Usage:
    python -m console.chunk_viewer
    python -m console.chunk_viewer --project llm-agent-system
    python -m console.chunk_viewer --query "authentication"
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rich.console import Console as RichConsole
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

# Import LLM-Agent-System's LanceDB store
LLM_ENGINE_AVAILABLE = False
LanceDBStore = None
MemoryEntry = None
try:
    from rag.vector_stores.lancedb_store import LanceDBStore, LANCEDB_AVAILABLE
    from rag.vector_stores.base_store import MemoryEntry
    LLM_ENGINE_AVAILABLE = LANCEDB_AVAILABLE
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('chunk_viewer.log')]
)
logger = logging.getLogger(__name__)


class ChunkViewer:
    """
    Interactive chunk browser for the Console

    Features:
    - List documents and their chunks
    - Search chunks by content or metadata
    - View chunk details with syntax highlighting
    - Navigate through chunk pages
    - Filter by document, topic, or entity

    Commands:
    - /list [doc_id]     - List documents or chunks in a document
    - /search <query>    - Search chunks
    - /view <chunk_id>   - View chunk details
    - /topics            - List all memory types
    - /recent [n]        - Show recent chunks
    - /stats             - Show storage statistics
    - /help              - Show help
    - /back              - Return to Console
    """

    def __init__(self, project: str = "default", backend: str = "auto"):
        self.console = RichConsole()
        self.project = project
        self.running = True

        # Storage backend
        self._llm_storage = None
        self._active_backend = None
        self._init_storage()

        # Current view state
        self.current_chunks: List[Dict] = []
        self.current_page = 0
        self.page_size = 10

    def _init_storage(self):
        """Initialize storage backend"""
        if LLM_ENGINE_AVAILABLE and LanceDBStore is not None:
            try:
                db_path = Path.home() / ".llm_engine" / "vector_db" / "private"
                if not db_path.exists():
                    db_path = Path.home() / ".llm_engine" / "lance_db"

                self._llm_storage = LanceDBStore(
                    db_path=str(db_path),
                    collection_prefix=self.project
                )
                self._active_backend = "llm-agent-system"
                logger.info(f"LanceDB available at {db_path}")
            except Exception as e:
                logger.warning(f"Storage init failed: {e}")

        if not self._active_backend:
            logger.error("No storage backend available")

    def display_welcome(self):
        """Display welcome banner"""
        backend_status = f"[green]{self._active_backend}[/green]" if self._active_backend else "[red]none[/red]"
        banner = f"""
[bold cyan]{'='*50}
   Chunk Viewer
   Project: {self.project}
   Backend: {backend_status}
{'='*50}[/bold cyan]

Browse and search document chunks.
Type /help for commands.
"""
        self.console.print(banner)

    def get_prompt(self) -> str:
        """Get user input"""
        try:
            return Prompt.ask("[bold cyan]Chunks>[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            return "/back"

    def run(self):
        """Main viewer loop"""
        self.display_welcome()

        if not self._active_backend:
            self.console.print("[red]No storage backend available[/red]")
            self.console.print("[dim]Install: pip install lancedb pyarrow[/dim]")
            return

        # Show initial stats
        self.cmd_stats()

        while self.running:
            try:
                user_input = self.get_prompt()

                if not user_input.strip():
                    continue

                self.handle_input(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Use /back to exit[/dim]")

        self.console.print("\n[dim]Chunk viewer closed[/dim]")

    def handle_input(self, input_str: str):
        """Handle user input"""
        input_str = input_str.strip()

        if input_str.startswith('/'):
            parts = input_str[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            handlers = {
                'back': self.cmd_back,
                'exit': self.cmd_back,
                'quit': self.cmd_back,
                'help': self.cmd_help,
                'list': self.cmd_list,
                'search': self.cmd_search,
                'view': self.cmd_view,
                'topics': self.cmd_topics,
                'recent': self.cmd_recent,
                'stats': self.cmd_stats,
                'next': self.cmd_next,
                'prev': self.cmd_prev,
            }

            handler = handlers.get(cmd)
            if handler:
                handler(args)
            else:
                self.console.print(f"[red]Unknown command: /{cmd}[/red]")
        else:
            # Default to search
            self.cmd_search(input_str)

    def cmd_back(self, args: str = ""):
        """Exit viewer"""
        self.running = False

    def cmd_help(self, args: str = ""):
        """Show help"""
        help_text = """
[bold]Commands:[/bold]

  /list              List collections/tables
  /search <query>    Search chunks by content
  /view <id>         View chunk details
  /topics            List all memory types
  /recent [n]        Show n most recent chunks (default: 10)
  /stats             Show storage statistics
  /next              Next page of results
  /prev              Previous page of results
  /help              Show this help
  /back              Return to Console

[dim]Tip: Just type text to search (no /search needed)[/dim]
"""
        self.console.print(Panel(help_text, title="Chunk Viewer Help"))

    def cmd_list(self, args: str = ""):
        """List collections and their contents"""
        if not self._active_backend:
            self.console.print("[red]No storage backend available[/red]")
            return

        try:
            collections = self._llm_storage.list_collections()

            if not collections:
                self.console.print("[dim]No collections found[/dim]")
                return

            table = Table(title=f"Collections in {self.project}")
            table.add_column("Collection", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Memory Types")

            for collection in collections:
                stats = self._llm_storage.get_stats(collection)
                count = stats.get('count', 0)
                memory_types = stats.get('memory_types', {})
                types_str = ", ".join(memory_types.keys()) if memory_types else "-"

                table.add_row(collection[:40], str(count), types_str[:30])

            self.console.print(table)

        except Exception as e:
            logger.error(f"List error: {e}")
            self.console.print(f"[red]Error: {e}[/red]")

    def cmd_search(self, query: str):
        """Search chunks by text content"""
        if not query.strip():
            self.console.print("[yellow]Usage: /search <query>[/yellow]")
            return

        if not self._active_backend:
            self.console.print("[red]No storage backend available[/red]")
            return

        try:
            self.console.print(f"[dim]Searching for: {query}...[/dim]")
            results = []
            query_lower = query.lower()

            collections = self._llm_storage.list_collections()

            for collection in collections:
                docs = self._llm_storage.get_all_documents(collection, limit=100)
                for i, (doc_id, content) in enumerate(zip(docs.get('ids', []), docs.get('documents', []))):
                    if content and query_lower in content.lower():
                        metadatas = docs.get('metadatas', [])
                        metadata = {}
                        if i < len(metadatas):
                            meta = metadatas[i]
                            if isinstance(meta, str):
                                try:
                                    metadata = json.loads(meta)
                                except:
                                    metadata = {}
                            else:
                                metadata = meta or {}

                        results.append({
                            'id': doc_id,
                            'content': content,
                            'collection': collection,
                            'metadata': metadata
                        })

            if not results:
                self.console.print("[dim]No results found[/dim]")
                return

            self.current_chunks = results[:50]  # Limit results
            self.current_page = 0
            self._display_results()

        except Exception as e:
            logger.error(f"Search error: {e}")
            self.console.print(f"[red]Search error: {e}[/red]")

    def _display_results(self):
        """Display current page of results"""
        if not self.current_chunks:
            self.console.print("[dim]No results[/dim]")
            return

        start = self.current_page * self.page_size
        end = start + self.page_size
        page_chunks = self.current_chunks[start:end]

        total_pages = (len(self.current_chunks) + self.page_size - 1) // self.page_size

        table = Table(title=f"Results (Page {self.current_page + 1}/{total_pages})")
        table.add_column("#", style="dim", width=4)
        table.add_column("Content", max_width=60)
        table.add_column("Collection", max_width=15)
        table.add_column("Type", max_width=12)

        for i, chunk in enumerate(page_chunks, start=start + 1):
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                collection = chunk.get('collection', '-')
                metadata = chunk.get('metadata', {})
            else:
                content = getattr(chunk, 'content', '')
                collection = '-'
                metadata = getattr(chunk, 'metadata', {})

            # Truncate content
            content_display = content[:100] + "..." if len(content) > 100 else content
            content_display = content_display.replace('\n', ' ')

            mem_type = metadata.get('memory_type', '-') if isinstance(metadata, dict) else '-'

            table.add_row(str(i), content_display, collection[:15], mem_type[:12])

        self.console.print(table)
        self.console.print(f"[dim]/view <#> to see details, /next /prev to navigate[/dim]")

    def cmd_view(self, args: str):
        """View chunk details"""
        if not args.strip():
            self.console.print("[yellow]Usage: /view <number>[/yellow]")
            return

        try:
            idx = int(args.strip()) - 1
            if idx < 0 or idx >= len(self.current_chunks):
                self.console.print(f"[red]Invalid chunk number. Range: 1-{len(self.current_chunks)}[/red]")
                return

            chunk = self.current_chunks[idx]

            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                chunk_id = chunk.get('id', 'unknown')
            else:
                content = getattr(chunk, 'content', '')
                metadata = getattr(chunk, 'metadata', {})
                chunk_id = getattr(chunk, 'id', 'unknown')

            # Detect if it's code
            lang = metadata.get('language', '') if isinstance(metadata, dict) else ''

            if lang in ['python', 'javascript', 'java', 'go', 'rust']:
                display = Syntax(content, lang, theme="monokai", line_numbers=True)
            else:
                display = content

            # Build metadata panel
            meta_lines = [f"[cyan]ID:[/cyan] {chunk_id}"]

            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    if key not in ['content', 'embedding', 'vector']:
                        meta_lines.append(f"[cyan]{key}:[/cyan] {value}")

            self.console.print(Panel(display, title=f"Chunk #{idx + 1}"))

            if meta_lines:
                self.console.print(Panel("\n".join(meta_lines), title="Metadata"))

        except ValueError:
            self.console.print("[red]Invalid chunk number[/red]")
        except Exception as e:
            logger.error(f"View error: {e}")
            self.console.print(f"[red]Error: {e}[/red]")

    def cmd_topics(self, args: str = ""):
        """List memory types across collections"""
        if not self._active_backend:
            self.console.print("[red]No storage backend available[/red]")
            return

        try:
            type_counts: Dict[str, int] = {}

            collections = self._llm_storage.list_collections()
            for collection in collections:
                stats = self._llm_storage.get_stats(collection)
                memory_types = stats.get('memory_types', {})
                for mem_type, count in memory_types.items():
                    type_counts[mem_type] = type_counts.get(mem_type, 0) + count

            if not type_counts:
                self.console.print("[dim]No types found[/dim]")
                return

            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

            table = Table(title="Memory Types")
            table.add_column("Type", style="cyan")
            table.add_column("Count", justify="right")

            for mem_type, count in sorted_types[:20]:
                table.add_row(str(mem_type), str(count))

            self.console.print(table)

        except Exception as e:
            logger.error(f"Topics error: {e}")
            self.console.print(f"[red]Error: {e}[/red]")

    def cmd_recent(self, args: str = ""):
        """Show recent chunks from all collections"""
        n = 10
        if args.strip():
            try:
                n = int(args.strip())
            except ValueError:
                pass

        if not self._active_backend:
            self.console.print("[red]No storage backend available[/red]")
            return

        try:
            results = []

            collections = self._llm_storage.list_collections()
            for collection in collections:
                docs = self._llm_storage.get_all_documents(collection, limit=n)
                for i, (doc_id, content) in enumerate(zip(docs.get('ids', []), docs.get('documents', []))):
                    metadatas = docs.get('metadatas', [])
                    metadata = {}
                    if i < len(metadatas):
                        meta = metadatas[i]
                        if isinstance(meta, str):
                            try:
                                metadata = json.loads(meta)
                            except:
                                metadata = {}
                        else:
                            metadata = meta or {}

                    results.append({
                        'id': doc_id,
                        'content': content,
                        'collection': collection,
                        'metadata': metadata
                    })

            if not results:
                self.console.print("[dim]No chunks found[/dim]")
                return

            self.current_chunks = results[:n]
            self.current_page = 0
            self._display_results()

        except Exception as e:
            logger.error(f"Recent error: {e}")
            self.console.print(f"[red]Error: {e}[/red]")

    def cmd_stats(self, args: str = ""):
        """Show storage statistics"""
        if not self._active_backend:
            self.console.print("[red]No storage backend available[/red]")
            return

        try:
            table = Table(title=f"Storage Stats - {self._active_backend}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")

            table.add_row("Backend", self._active_backend)

            collections = self._llm_storage.list_collections()
            table.add_row("Collections", str(len(collections)))

            total_count = 0
            for collection in collections:
                count = self._llm_storage.get_collection_count(collection)
                total_count += count
                table.add_row(f"  {collection}", str(count))

            table.add_row("Total Items", str(total_count))
            table.add_row("DB Path", str(self._llm_storage.persist_directory))

            self.console.print(table)

        except Exception as e:
            logger.error(f"Stats error: {e}")
            self.console.print(f"[red]Error: {e}[/red]")

    def cmd_next(self, args: str = ""):
        """Next page"""
        total_pages = (len(self.current_chunks) + self.page_size - 1) // self.page_size
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self._display_results()
        else:
            self.console.print("[dim]Already on last page[/dim]")

    def cmd_prev(self, args: str = ""):
        """Previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self._display_results()
        else:
            self.console.print("[dim]Already on first page[/dim]")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Chunk Viewer - Browse document chunks'
    )

    parser.add_argument(
        '--project', '-p',
        type=str,
        default='default',
        help='Project/collection to browse'
    )

    parser.add_argument(
        '--query', '-q',
        type=str,
        default=None,
        help='Initial search query'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    viewer = ChunkViewer(project=args.project)

    if args.query:
        viewer.cmd_search(args.query)

    viewer.run()


if __name__ == "__main__":
    main()
