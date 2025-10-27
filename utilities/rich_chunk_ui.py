#!/usr/bin/env python3
"""
Rich Chunk UI - Beautiful Terminal Interface

Interactive chunk viewer using Rich library
Supports keyboard navigation, search, and context adjustment
"""

import sys
from typing import Optional, Dict
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import Progress, BarColumn, TextColumn
from rich import box

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("⚠️  pynput not available. Install for better keyboard controls: pip install pynput")


class RichChunkUI:
    """
    Interactive Rich terminal UI for chunk viewing
    
    Controls:
    - ↑↓ or j/k: Navigate chunks
    - +/-: Adjust context size
    - g<n>: Jump to chunk number
    - /: Search (coming soon)
    - q: Quit
    - ?: Help
    """
    
    def __init__(self, viewer):
        """
        Initialize Rich UI
        
        Args:
            viewer: ChunkViewer instance
        """
        self.viewer = viewer
        self.console = Console()
        self.current_chunk_num = 0
        self.context_size = 5
        self.doc_id = None
        self.collection = "private"
        self.quit_requested = False
        
    def show_interactive(self, doc_id: str, initial_chunk: int = 0, 
                        collection: str = "private",
                        relevance_scores: Dict[int, float] = None):
        """
        Start interactive chunk viewing session
        
        Args:
            doc_id: Document ID to view
            initial_chunk: Starting chunk number
            collection: Collection name
            relevance_scores: Optional relevance scores for chunks
        """
        self.doc_id = doc_id
        self.current_chunk_num = initial_chunk
        self.collection = collection
        self.relevance_scores = relevance_scores or {}
        self.quit_requested = False
        
        # Show initial view
        self._display_current_view()
        
        # Enter command loop
        self._command_loop()
    
    def _display_current_view(self):
        """Display current chunk with context"""
        # Get chunk data
        data = self.viewer.get_chunk_with_context(
            doc_id=self.doc_id,
            chunk_number=self.current_chunk_num,
            context_size=self.context_size,
            collection=self.collection,
            relevance_scores=self.relevance_scores
        )
        
        if 'error' in data:
            self.console.print(f"[red]❌ Error: {data['error']}[/red]")
            if 'available_range' in data:
                min_chunk, max_chunk = data['available_range']
                self.console.print(f"[yellow]Available range: {min_chunk}-{max_chunk}[/yellow]")
            return
        
        # Clear screen
        self.console.clear()
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="content"),
            Layout(name="footer", size=3)
        )
        
        # Render components
        layout["header"].update(self._render_header(data))
        layout["content"].update(self._render_content(data))
        layout["footer"].update(self._render_footer(data))
        
        # Display
        self.console.print(layout)
    
    def _render_header(self, data: Dict) -> Panel:
        """Render document header"""
        doc_info = data['doc_info']
        progress = data.get('progress', 0.0)
        
        title = doc_info.get('title', 'Unknown Document')
        chunk_num = data['target_chunk_number']
        total_chunks = data['total_chunks']
        
        # Add chapter info if available
        chapter_text = ""
        target_chunk = next((c for c in data['chunks'] if c.is_target), None)
        if target_chunk and target_chunk.chapter_title:
            chapter_text = f" (Chapter {target_chunk.chapter_num}: {target_chunk.chapter_title})"
        
        header_text = Text()
        header_text.append("📄 ", style="bold blue")
        header_text.append(f"{title}", style="bold cyan")
        header_text.append(chapter_text, style="dim cyan")
        header_text.append(f"\n   Chunk {chunk_num}/{total_chunks} ", style="")
        header_text.append(f"({progress*100:.0f}%)", style="dim")
        
        return Panel(header_text, border_style="cyan", box=box.ROUNDED)
    
    def _render_content(self, data: Dict) -> Panel:
        """Render chunks with context"""
        chunks = data['chunks']
        context_start, context_end = data['context_range']
        
        # Build content
        content = Text()
        
        for chunk in chunks:
            # Chunk number and separator
            if chunk.is_target:
                # Target chunk - highlighted
                content.append("\n")
                content.append("╔" + "═" * 78 + "╗\n", style="bold yellow")
                content.append(f"║ [{chunk.chunk_number}] TARGET CHUNK", style="bold yellow")
                content.append(" " * (78 - len(f"[{chunk.chunk_number}] TARGET CHUNK") - 2), style="bold yellow")
                content.append("║\n", style="bold yellow")
                content.append("╚" + "═" * 78 + "╝\n", style="bold yellow")
                
                # Show relevance if available
                if chunk.relevance_score:
                    content.append(f"✨ Relevance: {chunk.relevance_score:.2f}  ", style="green")
                content.append(f"📊 Tokens: {chunk.token_count}  ", style="blue")
                if chunk.has_code:
                    content.append("💻 Contains Code  ", style="magenta")
                if chunk.chapter_title:
                    content.append(f"📖 {chunk.chapter_title}", style="cyan")
                content.append("\n\n")
                
                # Chunk content - with syntax highlighting if code
                if chunk.has_code and self._looks_like_code(chunk.content):
                    # Try to render as code
                    try:
                        syntax = Syntax(chunk.content, "python", theme="monokai", 
                                      line_numbers=False, word_wrap=True)
                        # Can't directly add Syntax to Text, so just add raw
                        content.append(chunk.content, style="")
                    except:
                        content.append(chunk.content, style="")
                else:
                    content.append(chunk.content, style="")
                
                content.append("\n\n")
            
            else:
                # Context chunk - dimmed
                content.append(f"\n[{chunk.chunk_number}] ", style="dim white")
                
                # Show brief preview (first 150 chars)
                preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                content.append(preview, style="dim white")
                content.append("\n", style="dim white")
        
        title = f"Context: Chunks {context_start}-{context_end}"
        return Panel(content, title=title, border_style="blue", box=box.ROUNDED)
    
    def _render_footer(self, data: Dict) -> Panel:
        """Render control hints"""
        controls = Text()
        
        # Navigation
        controls.append("↑↓/j/k", style="bold cyan")
        controls.append(": Navigate  ", style="")
        
        # Context
        controls.append("+/-", style="bold cyan")
        controls.append(": Context  ", style="")
        
        # Jump
        controls.append("g<n>", style="bold cyan")
        controls.append(": Jump  ", style="")
        
        # Search (future)
        # controls.append("/", style="bold cyan")
        # controls.append(": Search  ", style="")
        
        # Help
        controls.append("?", style="bold cyan")
        controls.append(": Help  ", style="")
        
        # Quit
        controls.append("q", style="bold cyan")
        controls.append(": Quit", style="")
        
        return Panel(controls, border_style="green", box=box.ROUNDED)
    
    def _command_loop(self):
        """Interactive command loop"""
        while not self.quit_requested:
            # Get command
            try:
                cmd = self.console.input("\n[bold cyan]Command (v for view mode):[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            
            if not cmd:
                continue
            
            # Process command
            if cmd in ['q', 'quit', 'exit']:
                break
            
            # ✅ NEW: View mode
            elif cmd == 'v':
                self._enter_view_mode()
            
            elif cmd in ['j', 'down']:
                self._move_next()
            
            elif cmd in ['k', 'up']:
                self._move_prev()
            
            elif cmd == '+':
                self.context_size = min(self.context_size + 2, 20)
                self._display_current_view()
            
            elif cmd == '-':
                self.context_size = max(self.context_size - 2, 0)
                self._display_current_view()
            
            elif cmd.startswith('g') and len(cmd) > 1:
                try:
                    target = int(cmd[1:])
                    self.current_chunk_num = target
                    self._display_current_view()
                except ValueError:
                    self.console.print("[red]Invalid chunk number[/red]")
            
            elif cmd == '?':
                self._show_help()
            
            else:
                self.console.print(f"[red]Unknown command: {cmd}[/red]")
                self.console.print("[dim]Type ? for help, v for view mode[/dim]")

    def _enter_view_mode(self):
        """
        Enter view mode with instant navigation
        
        Uses pynput for keyboard capture
        """
        if not PYNPUT_AVAILABLE:
            self.console.print("[yellow]⚠️  View mode requires pynput[/yellow]")
            self.console.print("[dim]Install with: pip install pynput[/dim]")
            self.console.print("[dim]Continuing with command mode...[/dim]")
            return
        
        from pynput import keyboard
        
        self.console.print("\n[bold green]📖 VIEW MODE[/bold green]")
        self.console.print("[dim]Use arrow keys or j/k to navigate, q to exit view mode[/dim]\n")
        
        view_mode_active = True
        
        def on_press(key):
            nonlocal view_mode_active
            
            try:
                # Handle special keys
                if key == keyboard.Key.down or (hasattr(key, 'char') and key.char == 'j'):
                    self._move_next()
                    return
                
                elif key == keyboard.Key.up or (hasattr(key, 'char') and key.char == 'k'):
                    self._move_prev()
                    return
                
                elif hasattr(key, 'char'):
                    if key.char == 'q':
                        self.console.print("\n[dim]Exiting view mode...[/dim]")
                        view_mode_active = False
                        return False  # Stop listener
                    
                    elif key.char == '+':
                        self.context_size = min(self.context_size + 2, 20)
                        self._display_current_view()
                        return
                    
                    elif key.char == '-':
                        self.context_size = max(self.context_size - 2, 0)
                        self._display_current_view()
                        return
                    
                    elif key.char == '?':
                        self._show_help()
                        return
            
            except AttributeError:
                pass
        
        # Start keyboard listener
        with keyboard.Listener(on_press=on_press) as listener:
            while view_mode_active:
                listener.join(timeout=0.1)
                if not listener.running:
                    break
        
        self.console.print("[dim]Returned to command mode[/dim]")
    
    def _move_next(self):
        """Move to next chunk"""
        data = self.viewer.get_chunk_with_context(
            self.doc_id, self.current_chunk_num, 
            self.context_size, self.collection
        )
        
        if data.get('can_go_next'):
            self.current_chunk_num += 1
            self._display_current_view()
        else:
            self.console.print("[yellow]⚠️  Already at last chunk[/yellow]")
    
    def _move_prev(self):
        """Move to previous chunk"""
        data = self.viewer.get_chunk_with_context(
            self.doc_id, self.current_chunk_num,
            self.context_size, self.collection
        )
        
        if data.get('can_go_prev'):
            self.current_chunk_num -= 1
            self._display_current_view()
        else:
            self.console.print("[yellow]⚠️  Already at first chunk[/yellow]")
    
    def _show_help(self):
        """Display help"""
        self.console.print("\n[bold cyan]Chunk Viewer Help[/bold cyan]\n")
        
        self.console.print("[bold]Navigation (Command Mode):[/bold]")
        self.console.print("  j, down    - Next chunk")
        self.console.print("  k, up      - Previous chunk")
        self.console.print("  +          - Increase context")
        self.console.print("  -          - Decrease context")
        self.console.print("  g<n>       - Jump to chunk number")
        self.console.print("  v          - Enter view mode (instant navigation)")
        self.console.print("  ?          - Show this help")
        self.console.print("  q          - Quit")
        
        self.console.print("\n[bold]View Mode:[/bold]")
        self.console.print("  j/k or ↑↓  - Navigate (instant, no Enter needed)")
        self.console.print("  +/-        - Adjust context")
        self.console.print("  q          - Exit view mode")
        
    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code"""
        code_indicators = ['def ', 'class ', 'import ', 'function', 'const ', 
                          'var ', 'let ', '#include', 'public ', 'private ']
        return any(indicator in text for indicator in code_indicators)