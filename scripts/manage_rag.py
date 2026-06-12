#!/usr/bin/env python3
"""
Enhanced RAG Database Management Script - FIXED IMPORTS

Provides comprehensive database management for vector stores

This script now uses LanceDB as the default backend via DualStoreManager.
"""

import argparse
import shutil
import sys
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# ============================================================
# CRITICAL: Document Format Support - Must be at top level
# ============================================================

# PDF Support
try:
    import PyPDF2
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    PyPDF2 = None
    PdfReader = None
    print("⚠️  Warning: PyPDF2 not installed. PDF support disabled.")
    print("   Install with: pip install PyPDF2")

# EPUB Support
try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False
    ebooklib = None
    epub = None

# DOCX Support
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    docx = None

# Excel Support (XLSX/XLS)
try:
    import openpyxl
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False
    openpyxl = None

try:
    import xlrd
    XLS_AVAILABLE = True
except ImportError:
    XLS_AVAILABLE = False
    xlrd = None

# Add project root to path BEFORE any local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import local modules
try:
    from utilities.metadata_store import MetadataStore
    from rag.vector_stores.dual_store_manager import DualStoreManager
    from rag.embedding.embedding_engine import EmbeddingEngine
    from rag.memory.memory_manager import MemoryManager
    from rag.query_parser import QueryParser
    from rag.intelligent_retriever import IntelligentRetriever, QualityMode
    from agents.knowledge_agent import KnowledgeAgent
    IMPORTS_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"⚠️  Warning: Could not import RAG modules: {e}")

console = Console()


# ============================================================
# Helper Functions
# ============================================================

def extract_pdf_metadata(pdf_path: Path) -> dict:
    """
    Extract metadata from PDF file

    Returns dict with: title, author, subject, creator, producer
    """
    if not PDF_AVAILABLE:
        return {}

    def clean_text(text):
        """Clean text by removing problematic characters and fixing encoding"""
        if not text:
            return text
        # Replace common encoding issues
        replacements = {
            '???': '—',  # Em dash
            '\ufffd': '?',  # Replacement character
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '—',  # Em dash
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        # Remove any remaining non-ASCII characters that might cause issues
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text

    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            metadata = pdf_reader.metadata

            if metadata:
                return {
                    'title': clean_text(metadata.get('/Title', pdf_path.stem)),
                    'author': clean_text(metadata.get('/Author', 'Unknown')),
                    'subject': clean_text(metadata.get('/Subject')),
                    'creator': clean_text(metadata.get('/Creator')),
                    'producer': clean_text(metadata.get('/Producer')),
                    'creation_date': metadata.get('/CreationDate')
                }
    except Exception as e:
        console.print(f"[yellow]⚠ Could not extract PDF metadata: {e}[/yellow]")

    return {'title': clean_text(pdf_path.stem), 'author': 'Unknown'}


def detect_chapters(text: str, file_name: str) -> list:
    """
    Detect chapter markers in text
    
    Returns list of dicts with chapter info: {number, title, start_pos}
    """
    import re
    
    chapters = []
    
    # Common chapter patterns
    patterns = [
        r'Chapter\s+(\d+)[:\s]+(.+?)(?:\n|$)',
        r'CHAPTER\s+(\d+)[:\s]+(.+?)(?:\n|$)',
        r'Part\s+(\d+)[:\s]+(.+?)(?:\n|$)',
        r'Section\s+(\d+)[:\s]+(.+?)(?:\n|$)',
        r'^(\d+)\.\s+(.+?)(?:\n|$)',  # "1. Introduction"
        r'(0x[0-9a-fA-F]+)([A-Z].+?)(?:\n|$)',  # Hex chapters like "0x100Introduction"
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.MULTILINE):
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip()
            
            # Clean up title
            chapter_title = chapter_title.rstrip('.')
            
            chapters.append({
                'number': chapter_num,
                'title': chapter_title,
                'start_pos': match.start()
            })
    
    # Deduplicate and sort
    seen = set()
    unique_chapters = []
    for chapter in chapters:
        key = (chapter['number'], chapter['title'])
        if key not in seen:
            seen.add(key)
            unique_chapters.append(chapter)
    
    return sorted(unique_chapters, key=lambda x: x['start_pos'])


def detect_code(text: str) -> bool:
    """Detect if text contains code"""
    code_indicators = [
        'def ', 'class ', 'import ', 'function',
        '() {', '};', 'const ', 'var ', 'let ',
        '#include', 'public static', 'private void',
        'if (', 'for (', 'while ('
    ]
    return any(indicator in text for indicator in code_indicators)


def detect_document_type(file_path: Path, content: str) -> str:
    """
    Detect document type based on file extension and content
    
    Returns: 'book', 'article', 'code', 'log', 'documentation', 'other'
    """
    ext = file_path.suffix.lower()

    # Code files
    code_extensions = [
        '.py', '.pyw',                           # Python
        '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx',  # C/C++
        '.js', '.jsx', '.mjs', '.ts', '.tsx',   # JavaScript/TypeScript
        '.rs',                                   # Rust
        '.rb',                                   # Ruby
        '.gd',                                   # GDScript
        '.sh', '.bash',                          # Shell
        '.go',                                   # Go
        '.java',                                 # Java
    ]
    if ext in code_extensions:
        return 'code'
    
    # Log files
    if ext in ['.log'] or 'ERROR' in content[:1000] or 'INFO' in content[:1000]:
        return 'log'
    
    # Check content for book indicators
    content_lower = content.lower()[:2000]
    if any(word in content_lower for word in ['chapter ', 'preface', 'table of contents']):
        return 'book'
    
    # Academic papers
    if any(word in content_lower for word in ['abstract', 'introduction', 'methodology', 'references']):
        return 'article'
    
    # Documentation
    if ext in ['.md'] or 'README' in file_path.name.upper():
        return 'documentation'
    
    return 'other'


def read_file(file_path: Path) -> str:
    """
    Read file content, supporting multiple formats

    Args:
        file_path: Path to file

    Returns:
        File content as string

    Raises:
        ValueError: If file format not supported or cannot be read
    """
    file_ext = file_path.suffix.lower()

    # Handle PDF files
    if file_ext == '.pdf':
        if not PDF_AVAILABLE:
            raise ValueError(
                "PDF support not available. Install PyPDF2: pip install PyPDF2"
            )

        try:
            console.print(f"[dim]Reading PDF file...[/dim]")

            # Use the imported PdfReader
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)

                # Extract text from all pages
                text_parts = []
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text)

                content = "\n\n".join(text_parts)

                if not content.strip():
                    raise ValueError("PDF appears to be empty or contains only images")

                console.print(f"[green]✓[/green] Extracted text from {len(reader.pages)} pages")
                return content

        except Exception as e:
            raise ValueError(f"Failed to read PDF: {e}")

    # Handle EPUB files
    elif file_ext == '.epub':
        if not EPUB_AVAILABLE:
            raise ValueError(
                "EPUB support not available. Install ebooklib: pip install ebooklib beautifulsoup4"
            )

        try:
            console.print(f"[dim]Reading EPUB file...[/dim]")

            book = epub.read_epub(str(file_path))
            text_parts = []
            chapter_count = 0

            # Extract text from all items
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    if text:
                        text_parts.append(text)
                        chapter_count += 1

            content = "\n\n".join(text_parts)

            if not content.strip():
                raise ValueError("EPUB appears to be empty")

            console.print(f"[green]✓[/green] Extracted text from {chapter_count} chapters")
            return content

        except Exception as e:
            raise ValueError(f"Failed to read EPUB: {e}")

    # Handle DOCX files
    elif file_ext == '.docx':
        if not DOCX_AVAILABLE:
            raise ValueError(
                "DOCX support not available. Install python-docx: pip install python-docx"
            )

        try:
            console.print(f"[dim]Reading DOCX file...[/dim]")

            doc = docx.Document(str(file_path))
            text_parts = []

            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)

            # Also extract from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        text_parts.append(row_text)

            content = "\n\n".join(text_parts)

            if not content.strip():
                raise ValueError("DOCX appears to be empty")

            console.print(f"[green]✓[/green] Extracted {len(text_parts)} paragraphs/rows")
            return content

        except Exception as e:
            raise ValueError(f"Failed to read DOCX: {e}")

    # Handle XLSX files (Excel 2007+)
    elif file_ext == '.xlsx':
        if not XLSX_AVAILABLE:
            raise ValueError(
                "XLSX support not available. Install openpyxl: pip install openpyxl"
            )

        try:
            console.print(f"[dim]Reading XLSX file...[/dim]")

            wb = openpyxl.load_workbook(str(file_path), data_only=True)
            text_parts = []

            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text_parts.append(f"=== Sheet: {sheet_name} ===")

                for row in sheet.iter_rows():
                    row_values = []
                    for cell in row:
                        if cell.value is not None:
                            row_values.append(str(cell.value))
                    if row_values:
                        text_parts.append(' | '.join(row_values))

            content = "\n".join(text_parts)

            if not content.strip():
                raise ValueError("XLSX appears to be empty")

            console.print(f"[green]✓[/green] Extracted from {len(wb.sheetnames)} sheets")
            return content

        except Exception as e:
            raise ValueError(f"Failed to read XLSX: {e}")

    # Handle XLS files (legacy Excel)
    elif file_ext == '.xls':
        if not XLS_AVAILABLE:
            raise ValueError(
                "XLS support not available. Install xlrd: pip install xlrd"
            )

        try:
            console.print(f"[dim]Reading XLS file...[/dim]")

            wb = xlrd.open_workbook(str(file_path))
            text_parts = []

            for sheet_idx in range(wb.nsheets):
                sheet = wb.sheet_by_index(sheet_idx)
                text_parts.append(f"=== Sheet: {sheet.name} ===")

                for row_idx in range(sheet.nrows):
                    row_values = []
                    for col_idx in range(sheet.ncols):
                        cell_value = sheet.cell_value(row_idx, col_idx)
                        if cell_value:
                            row_values.append(str(cell_value))
                    if row_values:
                        text_parts.append(' | '.join(row_values))

            content = "\n".join(text_parts)

            if not content.strip():
                raise ValueError("XLS appears to be empty")

            console.print(f"[green]✓[/green] Extracted from {wb.nsheets} sheets")
            return content

        except Exception as e:
            raise ValueError(f"Failed to read XLS: {e}")

    # Handle text files (default)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                raise ValueError("File is empty")

            return content

        except UnicodeDecodeError:
            try:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                return content
            except Exception:
                raise ValueError("Cannot read file (encoding issue)")
        except Exception as e:
            raise ValueError(f"Cannot read file: {e}")


def get_db_path() -> Path:
    """Get database base path"""
    return Path.home() / ".llm_engine" / "vector_db"


def show_stats():
    """Show database statistics"""
    base_path = get_db_path()
    
    table = Table(title="📊 RAG Database Statistics")
    table.add_column("Database", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Files", justify="right")
    
    for db_type in ["private", "public"]:
        db_path = base_path / db_type
        
        if db_path.exists():
            # Calculate size
            total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            # Count files
            file_count = len([f for f in db_path.rglob('*') if f.is_file()])
            
            table.add_row(
                db_type.capitalize(),
                "✓ Exists",
                f"{size_mb:.1f} MB",
                str(file_count)
            )
        else:
            table.add_row(
                db_type.capitalize(),
                "✗ Not found",
                "0 MB",
                "0"
            )
    
    console.print(table)


def show_metadata():
    """Show metadata database stats"""
    try:
        metadata_store = MetadataStore()
        stats = metadata_store.get_stats()
        
        console.print("\n[bold cyan]📊 Metadata Database Statistics:[/bold cyan]\n")
        
        table = Table(show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("Documents", str(stats['documents']))
        table.add_row("Chapters", str(stats['chapters']))
        table.add_row("Chunks", str(stats['chunks']))
        table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
        
        console.print(table)
        
        # Show recent documents
        docs = metadata_store.list_documents(limit=5)
        
        if docs:
            console.print("\n[bold cyan]📚 Recent Documents:[/bold cyan]\n")
            
            doc_table = Table(show_header=True)
            doc_table.add_column("Title", style="cyan")
            doc_table.add_column("Author")
            doc_table.add_column("Type", style="yellow")
            doc_table.add_column("Chunks", justify="right")
            
            for doc in docs:
                doc_table.add_row(
                    doc['title'][:40],
                    doc['author'][:20],
                    doc['doc_type'],
                    str(doc['total_chunks'])
                )
            
            console.print(doc_table)
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()


def list_collections(db_type: str = "all"):
    """List all collections in databases"""
    if not IMPORTS_AVAILABLE:
        console.print(f"[red]❌ Cannot import RAG modules: {IMPORT_ERROR}[/red]")
        return
    
    base_path = get_db_path()
    
    try:
        store_manager = DualStoreManager(
            private_db_path=str(base_path / "private"),
            public_db_path=str(base_path / "public")
        )
        
        stores_to_check = []
        if db_type in ["all", "private"]:
            stores_to_check.append(("Private", store_manager.private_store))
        if db_type in ["all", "public"]:
            stores_to_check.append(("Public", store_manager.public_store))
        
        for name, store in stores_to_check:
            console.print(f"\n[bold cyan]{name} Database:[/bold cyan]")

            try:
                # Use BaseVectorStore interface
                collections = store.list_collections()

                if not collections:
                    console.print("  [yellow]No collections found[/yellow]")
                    continue

                table = Table(show_header=True)
                table.add_column("Collection", style="cyan")
                table.add_column("Documents", justify="right", style="green")
                table.add_column("Status", style="yellow")

                for coll_name in collections:
                    try:
                        count = store.get_collection_count(coll_name)

                        # Get status from stats
                        stats = store.get_stats(coll_name)
                        status = "Active" if stats.get('exists', False) else "Empty"

                        table.add_row(coll_name, str(count), status)
                    except Exception as e:
                        table.add_row(coll_name, "Error", str(e))
                
                console.print(table)
                
            except Exception as e:
                console.print(f"  [red]❌ Error listing collections: {e}[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize stores: {e}[/red]")


def inspect_collection(collection_name: str, db_type: str = "private", limit: int = 5):
    """
    Enhanced inspect: Shows vector store data and SQLite metadata
    
    FIXES:
    - ValueError with embeddings array check
    - Shows metadata from SQLite
    - Better error handling
    """
    if not IMPORTS_AVAILABLE:
        console.print(f"[red]❌ Cannot import RAG modules: {IMPORT_ERROR}[/red]")
        return
    
    base_path = get_db_path()
    
    try:
        # Initialize components
        store_manager = DualStoreManager(
            private_db_path=str(base_path / "private"),
            public_db_path=str(base_path / "public")
        )
        metadata_store = MetadataStore()
        
        store = store_manager.private_store if db_type == "private" else store_manager.public_store
        
        console.print(f"\n[bold cyan]🔍 Inspecting Collection: {collection_name}[/bold cyan]")
        
        # Get documents from vector store
        try:
            # LanceDB returns all fields by default (no 'include' parameter needed)
            results = store.get_all_documents(collection_name, limit=limit)
            
            if not results['ids']:
                console.print("[yellow]Collection is empty[/yellow]")
                return
            
            console.print(f"\n[green]Found {len(results['ids'])} documents (showing first {limit}):[/green]\n")
            
            # Track document IDs to get metadata
            doc_ids_seen = set()
            
            for i, chunk_id in enumerate(results['ids']):
                # ✅ FIX: Safely get metadata (handle both dict and JSON string)
                metadata = {}
                if results.get('metadatas') and i < len(results['metadatas']):
                    raw_metadata = results['metadatas'][i]
                    if raw_metadata:
                        if isinstance(raw_metadata, str):
                            try:
                                import json
                                metadata = json.loads(raw_metadata)
                            except (json.JSONDecodeError, TypeError):
                                metadata = {'raw': raw_metadata}
                        elif isinstance(raw_metadata, dict):
                            metadata = raw_metadata
                        else:
                            metadata = {'raw': str(raw_metadata)}
                
                # ✅ FIX: Safely get document
                document = "N/A"
                if results.get('documents') and i < len(results['documents']):
                    document = results['documents'][i] if results['documents'][i] else "N/A"
                
                # ✅ FIX: Check embeddings properly (embeddings is a list of arrays)
                embedding_dims = 0
                if results.get('embeddings') is not None and len(results['embeddings']) > i:
                    try:
                        embedding_dims = len(results['embeddings'][i])
                    except:
                        embedding_dims = 0
                
                # Build display panel
                panel_content = f"[cyan]ID:[/cyan] {chunk_id}\n"
                panel_content += f"[cyan]Metadata:[/cyan] {metadata}\n"
                
                if embedding_dims > 0:
                    panel_content += f"[cyan]Embedding:[/cyan] {embedding_dims} dimensions\n"
                else:
                    panel_content += f"[yellow]⚠️ No embedding found[/yellow]\n"
                
                panel_content += f"[cyan]Content Preview:[/cyan]\n{document[:200]}{'...' if len(document) > 200 else ''}"
                
                panel = Panel(
                    panel_content,
                    title=f"Document {i+1}",
                    border_style="green"
                )
                console.print(panel)
                
                # Track doc_id for metadata lookup
                doc_id = metadata.get('doc_id')
                if doc_id:
                    doc_ids_seen.add(doc_id)
            
            # ✅ Show SQLite metadata for these documents
            if doc_ids_seen:
                console.print(f"\n[bold cyan]📊 Document Metadata (from SQLite):[/bold cyan]\n")
                
                for doc_id in doc_ids_seen:
                    try:
                        doc_info = metadata_store.get_document(doc_id=doc_id)
                        
                        if doc_info:
                            table = Table(show_header=True, title=f"Document: {doc_id}")
                            table.add_column("Field", style="cyan")
                            table.add_column("Value", style="green")
                            
                            table.add_row("Title", doc_info.get('title', 'N/A'))
                            table.add_row("Author", doc_info.get('author', 'N/A'))
                            table.add_row("Type", doc_info.get('doc_type', 'N/A'))
                            table.add_row("File Path", doc_info.get('file_path', 'N/A'))
                            table.add_row("Total Chunks", str(doc_info.get('total_chunks', 0)))
                            table.add_row("Total Tokens", f"{doc_info.get('total_tokens', 0):,}")
                            table.add_row("Created", doc_info.get('created_at', 'N/A'))
                            
                            console.print(table)
                            console.print()
                            
                            # Show chapters if any
                            chapters = metadata_store.get_chapters(doc_id=doc_id)
                            if chapters:
                                console.print(f"[cyan]Chapters:[/cyan] {len(chapters)} found")
                                ch_table = Table(show_header=True)
                                ch_table.add_column("Number", style="yellow")
                                ch_table.add_column("Title", style="cyan")
                                ch_table.add_column("Chunks", style="green")
                                
                                for ch in chapters[:10]:  # Show first 10
                                    ch_table.add_row(
                                        str(ch.get('chapter_num', 'N/A')),
                                        ch.get('chapter_title', 'N/A')[:50],
                                        f"{ch.get('start_chunk_num', 0)}-{ch.get('end_chunk_num', 0)}"
                                    )
                                
                                console.print(ch_table)
                                console.print()
                            
                            # ✅ NEW: Show chunk metadata
                            chunk_meta = metadata_store.get_chunk_metadata(doc_id=doc_id)
                            if chunk_meta:
                                console.print(f"[cyan]Chunk Metadata:[/cyan] {len(chunk_meta)} chunks tracked")
                            else:
                                console.print(f"[yellow]⚠️ No chunk metadata (context expansion won't work)[/yellow]")
                            
                        else:
                            console.print(f"[yellow]⚠️ No metadata found for doc_id: {doc_id}[/yellow]")
                    
                    except Exception as e:
                        console.print(f"[yellow]⚠️ Error getting metadata for {doc_id}: {e}[/yellow]")
        
        except Exception as e:
            console.print(f"[red]❌ Error inspecting collection: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize: {e}[/red]")
        import traceback
        traceback.print_exc()

# Also add this helper to show all metadata
def show_metadata_detailed():
    """Show detailed metadata from SQLite database"""
    try:
        metadata_store = MetadataStore()
        
        console.print("\n[bold cyan]📊 Detailed Metadata Database:[/bold cyan]\n")
        
        # Get all documents
        docs = metadata_store.list_documents(limit=100)
        
        if not docs:
            console.print("[yellow]No documents in metadata database[/yellow]")
            return
        
        # Create table
        table = Table(show_header=True, title=f"Documents ({len(docs)} total)")
        table.add_column("Title", style="cyan", max_width=30)
        table.add_column("Author", style="green", max_width=20)
        table.add_column("Type", style="yellow")
        table.add_column("Chunks", justify="right", style="magenta")
        table.add_column("Tokens", justify="right", style="blue")
        table.add_column("Created", style="dim")
        
        for doc in docs:
            table.add_row(
                doc['title'][:30],
                doc['author'][:20],
                doc['doc_type'],
                str(doc['total_chunks']),
                f"{doc['total_tokens']:,}" if doc['total_tokens'] else "0",
                doc['created_at'][:10] if doc.get('created_at') else 'N/A'
            )
        
        console.print(table)
        
        # Show statistics
        stats = metadata_store.get_stats()
        
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Documents: {stats['documents']}")
        console.print(f"  Chapters: {stats['chapters']}")
        console.print(f"  Chunks: {stats['chunks']}")
        console.print(f"  Total Tokens: {stats['total_tokens']:,}")
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()


def query_collection(collection_name: str, query: str, db_type: str = "private", top_k: int = 3):
    """Query a specific collection"""
    if not IMPORTS_AVAILABLE:
        console.print(f"[red]❌ Cannot import RAG modules: {IMPORT_ERROR}[/red]")
        return
    
    base_path = get_db_path()
    
    try:
        # Initialize components
        store_manager = DualStoreManager(
            private_db_path=str(base_path / "private"),
            public_db_path=str(base_path / "public")
        )

        embedder = EmbeddingEngine()  # Use config backend (respects rag_config.yaml)
        store = store_manager.private_store if db_type == "private" else store_manager.public_store
        
        console.print(f"\n[bold cyan]🔍 Querying Collection: {collection_name}[/bold cyan]")
        console.print(f"[yellow]Query:[/yellow] {query}\n")
        
        try:
            # Generate query embedding
            query_embedding = embedder.embed_text(query)

            # Search using BaseVectorStore interface
            memories = store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                collection_name=collection_name
            )

            if not memories:
                console.print("[yellow]No results found[/yellow]")
                return

            console.print(f"[green]Found {len(memories)} results:[/green]\n")

            for i, memory in enumerate(memories):
                # MemoryEntry objects don't have direct similarity scores
                # You can calculate from embedding similarity if needed
                similarity = 0.85  # Placeholder - would need actual calculation

                document = memory.content
                metadata = memory.metadata
                
                panel = Panel(
                    f"[green]Similarity:[/green] {similarity:.3f}\n"
                    f"[cyan]ID:[/cyan] {memory.id}\n"
                    f"[cyan]Type:[/cyan] {memory.memory_type}\n"
                    f"[cyan]Metadata:[/cyan] {metadata}\n"
                    f"[cyan]Content:[/cyan]\n{document[:300]}{'...' if len(document) > 300 else ''}",
                    title=f"Result {i+1}",
                    border_style="green"
                )
                console.print(panel)
        
        except Exception as e:
            console.print(f"[red]❌ Query failed: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize: {e}[/red]")


def test_query(query: str, quality: str = "balanced"):
    """Test a knowledge query"""
    if not IMPORTS_AVAILABLE:
        console.print(f"[red]❌ Cannot import RAG modules: {IMPORT_ERROR}[/red]")
        return
    
    try:
        console.print(f"\n[bold cyan]🔍 Testing Query:[/bold cyan] {query}")
        console.print(f"[yellow]Quality Mode:[/yellow] {quality.upper()}\n")
        
        # Initialize components
        metadata_store = MetadataStore()
        parser = QueryParser()
        
        # Parse query
        intent = parser.parse(query)
        console.print(f"[green]✓[/green] Query parsed:")
        console.print(f"  Intent: {intent.intent_type.value}")
        console.print(f"  Confidence: {intent.confidence:.2f}")
        if intent.author:
            console.print(f"  Author: {intent.author}")
        if intent.topic:
            console.print(f"  Topic: {intent.topic}")
        if intent.chapter:
            console.print(f"  Chapter: {intent.chapter}")
        
        # Initialize retriever
        base_path = get_db_path()
        store_manager = DualStoreManager(
            private_db_path=str(base_path / "private"),
            public_db_path=str(base_path / "public")
        )

        embedder = EmbeddingEngine()  # Use config backend (respects rag_config.yaml)

        retriever = IntelligentRetriever(
            metadata_store=metadata_store,
            store_manager=store_manager,
            embedding_engine=embedder
        )
        
        # Map quality string to enum
        quality_map = {
            'fast': QualityMode.FAST,
            'balanced': QualityMode.BALANCED,
            'accurate': QualityMode.ACCURATE,
            'thorough': QualityMode.THOROUGH
        }
        quality_mode = quality_map.get(quality.lower(), QualityMode.BALANCED)
        
        # Retrieve
        console.print(f"\n[yellow]Retrieving...[/yellow]")
        result = retriever.retrieve(
            query=query,
            intent=intent,
            quality=quality_mode,
            is_private=True
        )
        
        # Display results
        console.print(f"\n[bold green]✓ Retrieved {len(result.chunks)} chunks in {result.retrieval_time:.2f}s[/bold green]\n")
        
        if result.chunks:
            table = Table(show_header=True, title="Top Results")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Score", justify="right", style="green", width=6)
            table.add_column("Source", style="yellow")
            table.add_column("Preview")
            
            for i, chunk in enumerate(result.chunks[:5], 1):
                score = f"{chunk.final_score:.3f}"
                source = chunk.title or "Unknown"
                preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
                
                table.add_row(str(i), score, source, preview)
            
            console.print(table)
        else:
            console.print("[yellow]No results found[/yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()


def reset_db(db_type: str = "all", confirm: bool = False):
    """Reset vector databases"""
    base_path = get_db_path()
    
    if not confirm:
        console.print("[red]⚠️  Use --confirm flag to actually reset databases[/red]")
        return
    
    if db_type in ["all", "private"]:
        private_path = base_path / "private"
        if private_path.exists():
            shutil.rmtree(private_path)
            console.print("[green]✅ Private database reset[/green]")
        else:
            console.print("[yellow]ℹ️  Private database doesn't exist[/yellow]")
    
    if db_type in ["all", "public"]:
        public_path = base_path / "public"
        if public_path.exists():
            shutil.rmtree(public_path)
            console.print("[green]✅ Public database reset[/green]")
        else:
            console.print("[yellow]ℹ️  Public database doesn't exist[/yellow]")


def health_check():
    """Perform system health check"""
    console.print("\n[bold cyan]🏥 System Health Check[/bold cyan]\n")
    
    checks = []
    
    # Check imports
    if IMPORTS_AVAILABLE:
        checks.append(("RAG Modules", True, "All modules importable"))
    else:
        checks.append(("RAG Modules", False, f"Import error: {IMPORT_ERROR}"))
    
    # Check database paths
    base_path = get_db_path()
    checks.append(("Base Path", base_path.parent.exists(), str(base_path.parent)))
    checks.append(("Private DB", (base_path / "private").exists(), "Exists"))
    checks.append(("Public DB", (base_path / "public").exists(), "Exists"))
    
    # Check metadata DB
    try:
        metadata_store = MetadataStore()
        stats = metadata_store.get_stats()
        checks.append(("Metadata DB", True, f"{stats['documents']} documents"))
    except Exception as e:
        checks.append(("Metadata DB", False, str(e)))
    
    # Display results
    table = Table(show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    passed = 0
    failed = 0
    
    for name, status, details in checks:
        if status:
            table.add_row(name, "✓ PASS", details)
            passed += 1
        else:
            table.add_row(name, "✗ FAIL", details)
            failed += 1
    
    console.print(table)
    
    console.print(f"\n[bold]Results: {passed} passed, {failed} failed[/bold]")
    
    if failed == 0:
        console.print(f"\n[bold green]✅ All checks passed![/bold green]")
    else:
        console.print(f"\n[bold red]⚠️  {failed} checks failed[/bold red]")
        

def index_folder(folder_path: str, collection: str = "private", 
                recursive: bool = True, force: bool = False,
                extensions: list = None):
    """
    Index all files in a folder
    
    Args:
        folder_path: Path to folder
        collection: Collection name
        recursive: Search subfolders
        force: Re-index existing files
        extensions: List of file extensions to index
    
    FIXED: Removed duplicate checking - let index_file handle it
    """
    if not IMPORTS_AVAILABLE:
        console.print(f"[red]❌ Cannot import RAG modules: {IMPORT_ERROR}[/red]")
        return
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        console.print(f"[red]❌ Folder not found: {folder_path}[/red]")
        return
    
    if not folder_path.is_dir():
        console.print(f"[red]❌ Not a folder: {folder_path}[/red]")
        return
    
    # Default extensions if none specified
    if extensions is None:
        extensions = ['.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.log', '.pdf',
                      '.epub', '.docx', '.xlsx', '.xls']
    
    console.print(f"\n[bold cyan]📁 Indexing Folder[/bold cyan]")
    console.print(f"Folder: {folder_path}")
    console.print(f"Recursive: {recursive}")
    console.print(f"Extensions: {', '.join(extensions)}")
    console.print(f"Collection: {collection}\n")
    
    # Find all files
    if recursive:
        files = []
        for ext in extensions:
            files.extend(folder_path.rglob(f"*{ext}"))
    else:
        files = []
        for ext in extensions:
            files.extend(folder_path.glob(f"*{ext}"))
    
    if not files:
        console.print(f"[yellow]⚠️  No files found with extensions: {', '.join(extensions)}[/yellow]")
        return
    
    console.print(f"[bold]Found {len(files)} files to index[/bold]\n")
    
    # Track results
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # Index each file
    for i, file_path in enumerate(files, 1):
        console.print(f"\n[bold cyan]File {i}/{len(files)}:[/bold cyan] {file_path.name}")
        
        try:
            # ✅ FIXED: Just call index_file - it handles duplicate detection properly
            result = index_file(str(file_path), collection, force)
            
            # Check result to track stats
            if result == "skipped":
                skip_count += 1
            elif result == "success":
                success_count += 1
            
        except Exception as e:
            console.print(f"[red]  ❌ Error: {e}[/red]")
            error_count += 1
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    # Final summary
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold]Indexing Complete![/bold]\n")
    console.print(f"[green]✓ Successfully indexed:[/green] {success_count}")
    console.print(f"[yellow]⊘ Skipped (already indexed):[/yellow] {skip_count}")
    console.print(f"[red]✗ Errors:[/red] {error_count}")
    console.print(f"\nTotal files processed: {len(files)}")

def index_file(file_path: str, collection: str = "private", force: bool = False):
    """
    Enhanced file indexing with proper metadata extraction
    
    FIXED: Use BaseVectorStore.add_memories() instead of non-existent add_texts()
    
    Returns:
        "success", "skipped", or raises Exception
    """
    if not IMPORTS_AVAILABLE:
        console.print(f"[red]❌ Cannot import RAG modules: {IMPORT_ERROR}[/red]")
        raise ImportError("RAG modules not available")
    
    from utilities.semantic_chunker import SemanticChunker
    from utilities.metadata_store import MetadataStore
    from utilities.token_counter import TokenCounter
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        console.print(f"[red]❌ File not found: {file_path}[/red]")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check if file type is supported
    supported_extensions = [
        # Text and documentation
        '.txt', '.md', '.log', '.pdf',
        # E-books and documents
        '.epub', '.docx', '.doc',
        # Spreadsheets
        '.xlsx', '.xls',
        # Config and data
        '.json', '.yaml', '.yml',
        # Python
        '.py', '.pyw',
        # C/C++
        '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx',
        # JavaScript/TypeScript
        '.js', '.jsx', '.mjs', '.ts', '.tsx',
        # Other languages
        '.rs',    # Rust
        '.rb',    # Ruby
        '.gd',    # GDScript (Godot)
        '.sh', '.bash',  # Bash/Shell
        '.go',    # Go
        '.java',  # Java
    ]
    if file_path.suffix.lower() not in supported_extensions:
        console.print(f"[red]❌ Unsupported file type: {file_path.suffix}[/red]")
        console.print(f"[yellow]Supported types: {', '.join(sorted(set(supported_extensions)))}[/yellow]")
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    console.print(f"\n[bold cyan]📄 Indexing File[/bold cyan]")
    console.print(f"File: {file_path}")
    console.print(f"Type: {file_path.suffix}")
    console.print(f"Collection: {collection}\n")
    
    try:
        # Initialize components
        base_path = get_db_path()
        # Use dual-backend for reliability
        embedder = EmbeddingEngine(lazy_load=False)  # Use config backend
        import torch

        console.print("[bold yellow]🔍 DEBUG: Checking GPU usage...[/bold yellow]")
        console.print(f"   CUDA Available: {torch.cuda.is_available()}")
        console.print(f"   Config Device: {embedder.device}")
        console.print(f"   Config Batch Size: {embedder.batch_size}")
        console.print(f"   Model Name: {embedder.model_name}")
        console.print(f"   Model Format: {embedder.model_format}")

        if embedder.model is not None:
            # Only access .device for HuggingFace models (not GGUF)
            if embedder.model_format == 'hf':
                console.print(f"   Model Device (actual): {embedder.model.device}")
                # Check GPU memory for HF models
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    console.print(f"   GPU Memory Allocated: {mem_allocated:.2f} GB")
            else:
                console.print(f"   Model Type: GGUF (llama-cpp-python)")
                # For GGUF models, use nvidia-smi to check GPU memory
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True
                    )
                    gpu_mem_mb = int(result.stdout.strip())
                    console.print(f"   GPU Memory (nvidia-smi): {gpu_mem_mb} MB")

                    # Check config for GPU layers
                    import yaml
                    with open('config/rag_config.yaml') as f:
                        cfg = yaml.safe_load(f)
                        n_gpu_layers = cfg.get('embedding', {}).get('gguf', {}).get('n_gpu_layers', 0)

                    if n_gpu_layers > 0:
                        console.print(f"   [green]✅ GPU enabled (n_gpu_layers={n_gpu_layers})[/green]")
                    else:
                        console.print(f"   [yellow]⚠️  CPU only (n_gpu_layers=0)[/yellow]")
                except:
                    console.print(f"   [dim]Could not check GPU memory[/dim]")
        else:
            console.print("   [yellow]⚠️  Model not loaded yet (lazy loading)[/yellow]")

        console.print("")
        store_manager = DualStoreManager(
            private_db_path=str(base_path / "private"),
            public_db_path=str(base_path / "public")
        )
        metadata_store = MetadataStore()
        
        # Get the store
        store = store_manager.private_store if collection == "private" else store_manager.public_store
        
        # Generate document ID
        doc_id = f"doc_{file_path.stem}_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}"
        
        # Check if already indexed (unless force)
        if not force:
            try:
                collection_name = f"{store.collection_prefix}_documents"
                # Check if document exists using LanceDB-compatible method
                if store.collection_exists(collection_name):
                    results = store.get_all_documents(collection_name, limit=1000)
                    existing = {'ids': results.get('ids', [])}
                else:
                    existing = {'ids': []}
                
                if existing and existing.get('ids'):
                    console.print(f"[yellow]⚠️  File already indexed: {doc_id}[/yellow]")
                    console.print(f"[yellow]   Use --force to re-index[/yellow]")
                    return "skipped"
            except Exception as e:
                console.print(f"[dim]Could not check existing: {e}[/dim]")
        
        # Audio / video / image processing was removed in v0.2 — only text-like
        # formats (and PDF, handled separately below) flow through this script.
        is_multimodal = False

        # Extract PDF metadata if applicable
        pdf_metadata = {}
        if file_path.suffix.lower() == '.pdf':
            console.print("[dim]Extracting PDF metadata...[/dim]")
            pdf_metadata = extract_pdf_metadata(file_path)
            if pdf_metadata and pdf_metadata.get('title') != file_path.stem:
                console.print(f"[green]✓[/green] Found: {pdf_metadata.get('title', 'Unknown Title')}")

        console.print("[dim]Reading file content...[/dim]")
        content = read_file(file_path)
        console.print(f"[green]✓[/green] Read {len(content)} characters")
        
        # Detect chapters
        console.print("[dim]Detecting document structure...[/dim]")
        chapters = detect_chapters(content, file_path.name)
        if chapters:
            console.print(f"[green]✓[/green] Found {len(chapters)} chapters")
        
        # Chunk the text using adaptive smart chunker
        console.print("[dim]Creating adaptive chunks (code-aware)...[/dim]")
        try:
            from utilities.adaptive_chunker import AdaptiveSmartChunker

            # Load chunker from config
            adaptive_chunker = AdaptiveSmartChunker.from_config('config/rag_config.yaml')

            # Chunk with content awareness
            adaptive_chunks = adaptive_chunker.chunk(content, file_path=str(file_path))
            console.print(f"[green]✓[/green] Created {len(adaptive_chunks)} chunks (adaptive)")

            # Display metadata summary
            if len(adaptive_chunks) > 0:
                console.print("\n[bold cyan]📊 Metadata Summary:[/bold cyan]")

                # Count chunks by content type
                content_types = {}
                languages = {}
                domains_set = set()
                topics_set = set()

                for chunk in adaptive_chunks:
                    ct = chunk.metadata.content_type
                    content_types[ct] = content_types.get(ct, 0) + 1

                    lang = chunk.metadata.language
                    if lang and lang != 'unknown':
                        languages[lang] = languages.get(lang, 0) + 1

                if content_types:
                    type_str = ", ".join([f"{k}: {v}" for k, v in content_types.items()])
                    console.print(f"  Content Types: {type_str}")

                if languages:
                    lang_str = ", ".join([f"{k}: {v}" for k, v in languages.items()])
                    console.print(f"  Languages: {lang_str}")

                # Show functions from first few chunks
                sample_functions = []
                for chunk in adaptive_chunks[:3]:
                    if chunk.metadata.functions:
                        sample_functions.extend(chunk.metadata.functions[:2])
                if sample_functions:
                    func_str = ", ".join(sample_functions[:5])
                    if len(sample_functions) > 5:
                        func_str += f" (+{len(sample_functions)-5} more)"
                    console.print(f"  Functions: {func_str}")

                console.print()

            # Extract enhanced metadata
            try:
                from utilities.metadata_extractor import MetadataExtractor
                meta_extractor = MetadataExtractor()
                console.print("[dim]Extracting enhanced metadata (domains, topics, complexity)...[/dim]")
            except ImportError as e:
                console.print(f"[yellow]⚠ Metadata extractor not available: {e}[/yellow]")
                meta_extractor = None

            # Convert to expected format
            chunks = []
            for chunk in adaptive_chunks:
                chunk_dict = {
                    'text': chunk.content,
                    'token_count': chunk.token_count,
                    'chunk_number': chunk.chunk_id,
                    # Stage 1 metadata (from adaptive chunker)
                    'content_type': chunk.metadata.content_type,
                    'language': chunk.metadata.language,
                    'is_complete': chunk.metadata.is_complete,
                    'functions': chunk.metadata.functions,
                    'chunk_strategy': chunk.metadata.chunk_strategy
                }

                # Stage 2 metadata (enhanced extraction)
                if meta_extractor and chunk.metadata.content_type == 'code':
                    enhanced_meta = meta_extractor.extract(
                        chunk.content,
                        file_path=str(file_path),
                        existing_language=chunk.metadata.language
                    )

                    # Add enhanced metadata
                    chunk_dict.update({
                        'classes': enhanced_meta.classes,
                        'imports': enhanced_meta.imports,
                        'keywords': enhanced_meta.keywords,
                        'domains': enhanced_meta.domains,
                        'topics': enhanced_meta.topics,
                        'complexity': enhanced_meta.complexity,
                        'has_examples': enhanced_meta.has_examples,
                        'has_comments': enhanced_meta.has_comments,
                        'has_documentation': enhanced_meta.has_documentation,
                        'code_patterns': enhanced_meta.code_patterns,
                        'language_confidence': enhanced_meta.language_confidence,
                    })

                chunks.append(chunk_dict)

            # Displayenhanced metadata summary
            code_chunks = [c for c in chunks if meta_extractor and c.get('content_type') == 'code']
            if code_chunks and meta_extractor:
                console.print("\n[bold cyan]✨Enhanced Metadata:[/bold cyan]")

                # Collect Stage 2 metadata
                all_domains = []
                all_topics = []
                complexities = []
                quality_counts = {'examples': 0, 'comments': 0, 'documentation': 0}

                for chunk in code_chunks:
                    if chunk.get('domains'):
                        all_domains.extend(chunk['domains'])
                    if chunk.get('topics'):
                        all_topics.extend(chunk['topics'])
                    if chunk.get('complexity'):
                        complexities.append(chunk['complexity'])
                    if chunk.get('has_examples'):
                        quality_counts['examples'] += 1
                    if chunk.get('has_comments'):
                        quality_counts['comments'] += 1
                    if chunk.get('has_documentation'):
                        quality_counts['documentation'] += 1

                # Display domains
                if all_domains:
                    domain_counts = {}
                    for d in all_domains:
                        domain_counts[d] = domain_counts.get(d, 0) + 1
                    domain_icons = {
                        'security': '🔒',
                        'systems': '⚙️',
                        'networking': '🌐',
                        'physics': '🔬',
                        'neuroscience': '🧠',
                        'ai_ml': '🤖',
                        'graphics': '🎨',
                        'web': '🌍',
                    }
                    domain_strs = []
                    for d, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                        icon = domain_icons.get(d, '📦')
                        domain_strs.append(f"{icon} {d.replace('_', ' ').title()} ({count})")
                    console.print(f"  Domains: {', '.join(domain_strs)}")

                # Display topics
                if all_topics:
                    topic_counts = {}
                    for t in all_topics:
                        topic_counts[t] = topic_counts.get(t, 0) + 1
                    topic_strs = []
                    for t, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                        topic_strs.append(f"{t.replace('_', ' ').title()} ({count})")
                    console.print(f"  Topics: {', '.join(topic_strs)}")

                # Display complexity distribution
                if complexities:
                    comp_counts = {}
                    for c in complexities:
                        comp_counts[c] = comp_counts.get(c, 0) + 1
                    comp_icons = {'simple': '🟢', 'medium': '🟡', 'complex': '🔴'}
                    comp_strs = []
                    for c in ['simple', 'medium', 'complex']:
                        if c in comp_counts:
                            icon = comp_icons[c]
                            comp_strs.append(f"{icon} {c}: {comp_counts[c]}")
                    console.print(f"  Complexity: {', '.join(comp_strs)}")

                # Display quality indicators
                quality_strs = []
                if quality_counts['examples'] > 0:
                    quality_strs.append(f"📚 {quality_counts['examples']} with examples")
                if quality_counts['comments'] > 0:
                    quality_strs.append(f"💬 {quality_counts['comments']} with comments")
                if quality_counts['documentation'] > 0:
                    quality_strs.append(f"📝 {quality_counts['documentation']} documented")
                if quality_strs:
                    console.print(f"  Quality: {', '.join(quality_strs)}")

                console.print()

        except Exception as e:
            # Fallback to old chunker if adaptive fails
            console.print(f"[yellow]⚠ Adaptive chunker failed: {e}, using fallback[/yellow]")
            chunker = SemanticChunker()
            token_counter = TokenCounter()

            chunk_strings = chunker.chunk_by_tokens(content, chunk_size=1024, overlap=128)
            console.print(f"[green]✓[/green] Created {len(chunk_strings)} chunks (fallback)")

            chunks = []
            for i, chunk_text in enumerate(chunk_strings):
                chunk_dict = {
                    'text': chunk_text,
                    'token_count': token_counter.count_tokens(chunk_text),
                    'chunk_number': i
                }
                chunks.append(chunk_dict)
        
        # Generate embeddings
        console.print(f"[dim]Generating embeddings (batch_size={embedder.batch_size})...[/dim]")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = embedder.embed_texts(
            chunk_texts,
            show_progress=True
        )
        console.print(f"[green]✓[/green] Generated {len(embeddings)} embeddings")
        
        # Prepare metadata for each chunk
        metadata_list = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                'chunk_id': f"{doc_id}_chunk_{i}",
                'chunk_number': i,
                'doc_id': doc_id,
                'file_path': str(file_path),
                'file_name': file_path.name,
                'token_count': chunk['token_count'],
                'has_code': chunk.get('has_code') or detect_code(chunk['text']),
                'indexed_at': datetime.now().isoformat(),
                # Enhanced metadata from adaptive chunker
                'content_type': chunk.get('content_type', 'unknown'),
                'language': chunk.get('language', 'unknown'),
                'is_complete': chunk.get('is_complete', False),
                'chunk_strategy': chunk.get('chunk_strategy', 'token_based')
            }

            # Add function names if available
            if chunk.get('functions'):
                chunk_meta['functions'] = ','.join(chunk['functions'])  # Store as comma-separated string

            # Add enhanced metadata if available
            if chunk.get('classes'):
                chunk_meta['classes'] = ','.join(chunk['classes'])
            if chunk.get('imports'):
                chunk_meta['imports'] = ','.join(chunk['imports'])
            if chunk.get('keywords'):
                chunk_meta['keywords'] = ','.join(chunk['keywords'])
            if chunk.get('domains'):
                chunk_meta['domains'] = ','.join(chunk['domains'])
            if chunk.get('topics'):
                chunk_meta['topics'] = ','.join(chunk['topics'])
            if chunk.get('complexity'):
                chunk_meta['complexity'] = chunk['complexity']
            if 'has_examples' in chunk:
                chunk_meta['has_examples'] = chunk['has_examples']
            if 'has_comments' in chunk:
                chunk_meta['has_comments'] = chunk['has_comments']
            if 'has_documentation' in chunk:
                chunk_meta['has_documentation'] = chunk['has_documentation']
            if chunk.get('code_patterns'):
                chunk_meta['code_patterns'] = ','.join(chunk['code_patterns'])
            if 'language_confidence' in chunk:
                chunk_meta['language_confidence'] = chunk['language_confidence']

            # Add chapter info if available
            if chapters:
                # Find which chapter this chunk belongs to
                chunk_position = i / len(chunks)
                for j, chapter in enumerate(chapters):
                    chapter_start = j / len(chapters)
                    chapter_end = (j + 1) / len(chapters) if j < len(chapters) - 1 else 1.0
                    
                    if chapter_start <= chunk_position < chapter_end:
                        chunk_meta['chapter_num'] = chapter['number']
                        chunk_meta['chapter_title'] = chapter['title']
                        break
            
            metadata_list.append(chunk_meta)
        
        # Store in vector database using BaseVectorStore interface
        console.print("[dim]Storing in vector database...[/dim]")

        collection_name = "documents"  # Collection name without prefix

        # Create MemoryEntry objects for BaseVectorStore interface
        from rag.vector_stores.base_store import MemoryEntry

        memories = []
        for i, (chunk_text, embedding, metadata) in enumerate(zip(chunk_texts, embeddings, metadata_list)):
            memory = MemoryEntry(
                id=metadata['chunk_id'],
                content=chunk_text,
                embedding=embedding if isinstance(embedding, list) else embedding.tolist(),
                metadata=metadata,
                memory_type="document",
                timestamp=datetime.now().isoformat(),
                importance_score=0.5,  # Default importance
                access_count=0
            )
            memories.append(memory)

        # Use BaseVectorStore.add_memories()
        store.add_memories(memories, collection_name=collection_name)

        console.print(f"[green]✓[/green] Stored in vector database ({store.collection_prefix}_{collection_name})")
        
        # Store document metadata in MetadataStore
        console.print("[dim]Storing document metadata...[/dim]")
        
        # Get title and author from PDF metadata or use filename
        title = pdf_metadata.get('title', file_path.stem)
        author = pdf_metadata.get('author', 'Unknown')
        doc_type = detect_document_type(file_path, content)
        
        # Calculate file hash
        file_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Calculate total tokens
        total_tokens = sum(chunk['token_count'] for chunk in chunks)

        # Check if document already exists in metadata store
        existing_doc = metadata_store.get_document(file_path=str(file_path))

        if existing_doc:
            if force:
                console.print(f"[dim]Removing old metadata for re-indexing...[/dim]")
                metadata_store.delete_document(doc_id=existing_doc['doc_id'])
            else:
                # Document exists but force=False, skip to avoid UNIQUE constraint error
                console.print(f"[yellow]⚠️  Document already in metadata store: {existing_doc['doc_id']}[/yellow]")
                console.print(f"[yellow]   Use --force to re-index[/yellow]")
                return "skipped"

        metadata_store.add_document(
            doc_id=doc_id,
            title=title,
            author=author,
            doc_type=doc_type,
            file_path=str(file_path),
            file_hash=file_hash,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            file_type=file_path.suffix,
            indexed_at=datetime.now().isoformat()
        )
        
        # Add chapter metadata
        if chapters:
            console.print("[dim]Storing chapter information...[/dim]")
            for chapter in chapters:
                # Find which chunks belong to this chapter
                chapter_chunks = [m for m in metadata_list 
                                if m.get('chapter_num') == chapter['number']]
                
                if chapter_chunks:
                    metadata_store.add_chapter(
                        doc_id=doc_id,
                        chapter_num=chapter['number'],
                        chapter_title=chapter['title'],
                        start_chunk_num=min(c['chunk_number'] for c in chapter_chunks),
                        end_chunk_num=max(c['chunk_number'] for c in chapter_chunks),
                        summary=None  # Could be generated later
                    )
            console.print(f"[green]✓[/green] Stored {len(chapters)} chapters")
        
        # Add chunk metadata to database
        console.print("[dim]Populating chunk metadata...[/dim]")
        for meta in metadata_list:
            try:
                metadata_store.add_chunk_metadata(
                    doc_id=doc_id,
                    chunk_number=meta['chunk_number'],
                    vector_id=meta['chunk_id'],
                    token_count=meta['token_count'],
                    has_code=meta.get('has_code', False),
                    has_math=False,  # Could detect LaTeX later
                    language='en',  # Could use langdetect
                    chapter_id=None  # Link to chapter if exists
                )
            except Exception as e:
                console.print(f"[yellow]⚠ Error adding chunk metadata: {e}[/yellow]")
        
        console.print(f"[green]✓[/green] Metadata stored")
        
        # Success summary
        console.print(f"\n[bold green]✅ Successfully indexed![/bold green]")
        console.print(f"Document ID: {doc_id}")
        console.print(f"Title: {title}")
        console.print(f"Author: {author}")
        console.print(f"Type: {doc_type}")
        console.print(f"Chunks: {len(chunks)}")
        console.print(f"Total Tokens: {total_tokens:,}")
        if chapters:
            console.print(f"Chapters: {len(chapters)}")
        
        return "success"
        
    except Exception as e:
        console.print(f"\n[red]❌ Error indexing file: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise

def smart_chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list:
    """
    Intelligently chunk text while preserving paragraphs
    
    Returns list of chunk texts
    """
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        para_tokens = len(para.split())  # Rough token count
        
        # If paragraph alone is too big, split by sentences
        if para_tokens > chunk_size:
            sentences = para.split('. ')
            for sent in sentences:
                sent_tokens = len(sent.split())
                
                if current_tokens + sent_tokens > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + '. '
                    current_tokens = sent_tokens
                else:
                    current_chunk += sent + '. '
                    current_tokens += sent_tokens
        
        # Normal case: add paragraph if it fits
        elif current_tokens + para_tokens <= chunk_size:
            current_chunk += para + '\n\n'
            current_tokens += para_tokens
        
        # Paragraph doesn't fit, start new chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + '\n\n'
            current_tokens = para_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def redetect_languages(collection: str = "private", min_confidence: float = 0.5, dry_run: bool = False):
    """
    Re-detect languages for chunks with low confidence or 'generic' language.

    Uses enhanced language detection for better accuracy on PDF code blocks.

    Args:
        collection: Collection name ('private' or 'public')
        min_confidence: Re-detect if language_confidence < this (default 0.5)
        dry_run: Show what would be updated without actually updating
    """
    if not IMPORTS_AVAILABLE:
        console.print(f"[red]❌ Cannot import RAG modules: {IMPORT_ERROR}[/red]")
        return

    console.print(f"\n[bold cyan]🔍 Re-detecting Languages for Low-Confidence Chunks[/bold cyan]")
    console.print(f"Collection: {collection}")
    console.print(f"Min Confidence: {min_confidence}")
    console.print(f"Dry Run: {'Yes' if dry_run else 'No'}\n")

    try:
        # Import enhanced detection tools
        try:
            from utilities.enhanced_language_detector import EnhancedLanguageDetector
            from utilities.code_block_extractor import CodeBlockExtractor
            enhanced_detector = EnhancedLanguageDetector()
            code_extractor = CodeBlockExtractor(min_confidence=0.5)
            console.print("[green]✓[/green] Enhanced language detection loaded")
        except ImportError as e:
            console.print(f"[red]❌ Enhanced detection not available: {e}[/red]")
            console.print("[yellow]  Install required: utilities/enhanced_language_detector.py[/yellow]")
            return

        # Initialize stores
        base_path = get_db_path()
        store_manager = DualStoreManager(
            private_db_path=str(base_path / "private"),
            public_db_path=str(base_path / "public")
        )

        store = store_manager.private_store if collection == "private" else store_manager.public_store
        collection_name = "documents"

        # Check if collection exists
        if not store.collection_exists(collection_name):
            console.print(f"[red]❌ Collection not found: {collection_name}[/red]")
            return

        # Get all chunks using LanceDB-compatible method
        console.print(f"[dim]Fetching all chunks from {collection_name}...[/dim]")
        results = store.get_all_documents(collection_name, limit=10000)  # Large limit to get all

        if not results['ids']:
            console.print("[yellow]Collection is empty[/yellow]")
            return

        total_chunks = len(results['ids'])
        console.print(f"[green]✓[/green] Found {total_chunks} chunks total")

        # Filter chunks needing re-detection
        chunks_to_update = []
        for i, chunk_id in enumerate(results['ids']):
            metadata = results['metadatas'][i] if results.get('metadatas') else {}
            language = metadata.get('language', 'unknown')
            lang_confidence = metadata.get('language_confidence', 0.0)
            content_type = metadata.get('content_type', 'unknown')

            # Re-detect if:
            # 1. Language is 'generic'
            # 2. Language confidence is below threshold
            # 3. Content type is 'code' but language is unknown
            needs_redetection = (
                language == 'generic' or
                (isinstance(lang_confidence, (int, float)) and lang_confidence < min_confidence) or
                (content_type == 'code' and language in ['unknown', 'generic'])
            )

            if needs_redetection:
                chunks_to_update.append({
                    'id': chunk_id,
                    'content': results['documents'][i],
                    'metadata': metadata,
                    'old_language': language,
                    'old_confidence': lang_confidence
                })

        console.print(f"\n[bold yellow]Found {len(chunks_to_update)} chunks needing re-detection[/bold yellow]")
        console.print(f"  ({len(chunks_to_update) / total_chunks * 100:.1f}% of total)\n")

        if not chunks_to_update:
            console.print("[green]✅ All chunks have good language detection![/green]")
            return

        if dry_run:
            console.print("[yellow]DRY RUN - Showing what would be updated:[/yellow]\n")

        # Track statistics
        updated_count = 0
        improved_count = 0
        failed_count = 0
        language_changes = {}

        # Process each chunk
        for i, chunk_info in enumerate(chunks_to_update, 1):
            chunk_id = chunk_info['id']
            content = chunk_info['content']
            old_metadata = chunk_info['metadata']
            old_language = chunk_info['old_language']
            old_confidence = chunk_info['old_confidence']

            console.print(f"[cyan]{i}/{len(chunks_to_update)}[/cyan] {chunk_id}")
            console.print(f"  Old: {old_language} (confidence: {old_confidence})")

            try:
                # Try enhanced detection
                detection = enhanced_detector.detect(content[:2000])  # Use first 2000 chars

                new_language = detection.language
                new_confidence = detection.confidence

                console.print(f"  New: {new_language} (confidence: {new_confidence:.2f})")

                # Check if this is an improvement
                is_improvement = (
                    new_confidence >= min_confidence and
                    new_language != 'generic' and
                    (old_language == 'generic' or new_confidence > old_confidence)
                )

                if is_improvement:
                    console.print(f"  [green]✓ Improvement detected[/green]")
                    improved_count += 1

                    # Track language changes
                    change_key = f"{old_language} → {new_language}"
                    language_changes[change_key] = language_changes.get(change_key, 0) + 1

                    if not dry_run:
                        # Update metadata using LanceDB-compatible method
                        metadata_updates = {
                            'language': new_language,
                            'language_confidence': new_confidence,
                            'redetected_at': datetime.now().isoformat()
                        }

                        # Use BaseVectorStore.update_document_metadata()
                        success = store.update_document_metadata(
                            doc_id=chunk_id,
                            metadata_updates=metadata_updates,
                            collection_name=collection_name
                        )

                        if success:
                            updated_count += 1
                            console.print(f"  [green]✓ Updated in database[/green]")
                        else:
                            console.print(f"  [yellow]⚠️  Failed to update[/yellow]")
                            failed_count += 1

                else:
                    console.print(f"  [dim]No improvement, skipping[/dim]")

            except Exception as e:
                console.print(f"  [red]✗ Error: {e}[/red]")
                failed_count += 1

            console.print()

        # Summary
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Re-detection Complete![/bold]\n")

        if dry_run:
            console.print(f"[yellow]DRY RUN - No changes made[/yellow]\n")

        console.print(f"Total chunks analyzed: {len(chunks_to_update)}")
        console.print(f"[green]✓ Improvements found:[/green] {improved_count}")
        if not dry_run:
            console.print(f"[green]✓ Successfully updated:[/green] {updated_count}")
        console.print(f"[red]✗ Failed:[/red] {failed_count}")

        if language_changes:
            console.print(f"\n[bold cyan]Language Changes:[/bold cyan]")
            table = Table(show_header=True)
            table.add_column("Change", style="cyan")
            table.add_column("Count", justify="right", style="green")

            for change, count in sorted(language_changes.items(), key=lambda x: x[1], reverse=True):
                table.add_row(change, str(count))

            console.print(table)

        if dry_run:
            console.print(f"\n[yellow]Run without --dry-run to apply changes[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()


def summarize_document(doc_id: str = None, list_docs: bool = False,
                       analyze: bool = False, style: str = "narrative",
                       ratio: float = 0.1):
    """
    Summarize ingested documents usingEnhanced Summarizer.

    Args:
        doc_id: Document ID to summarize
        list_docs: List available documents
        analyze: Analyze document structure (chapters, tokens)
        style: Summary style (narrative, academic, technical, quick)
        ratio: Compression ratio (0.1 = 10%)
    """
    if not IMPORTS_AVAILABLE:
        console.print(f"[red]❌ Cannot import RAG modules: {IMPORT_ERROR}[/red]")
        return

    try:
        metadata_store = MetadataStore()

        # List documents mode
        if list_docs:
            console.print("\n[bold cyan]📚 Documents Available for Summarization[/bold cyan]\n")

            docs = metadata_store.list_documents(limit=50)

            if not docs:
                console.print("[yellow]No documents found in database.[/yellow]")
                console.print("[dim]Use 'manage_rag.py index <file>' to add documents.[/dim]")
                return

            table = Table(show_header=True, title=f"Ingested Documents ({len(docs)} total)")
            table.add_column("Doc ID", style="cyan")
            table.add_column("Title", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Chunks", justify="right")
            table.add_column("Tokens", justify="right")

            for doc in docs:
                table.add_row(
                    doc['doc_id'],
                    doc['title'][:40] + "..." if len(doc['title']) > 40 else doc['title'],
                    doc['doc_type'],
                    str(doc['total_chunks']),
                    f"{doc['total_tokens']:,}" if doc['total_tokens'] else "0"
                )

            console.print(table)
            console.print("\n[dim]Use --doc-id <id> --analyze to see document structure[/dim]")
            return

        # Analyze mode
        if analyze and doc_id:
            console.print(f"\n[bold cyan]📊 Analyzing Document: {doc_id}[/bold cyan]\n")

            # Get document info
            doc_info = metadata_store.get_document(doc_id=doc_id)
            if not doc_info:
                console.print(f"[red]❌ Document not found: {doc_id}[/red]")
                return

            # Display document info
            console.print(f"[green]Title:[/green] {doc_info['title']}")
            console.print(f"[green]Author:[/green] {doc_info['author']}")
            console.print(f"[green]Type:[/green] {doc_info['doc_type']}")
            console.print(f"[green]Total Chunks:[/green] {doc_info['total_chunks']}")
            console.print(f"[green]Total Tokens:[/green] {doc_info['total_tokens']:,}")

            # Get chapters if any
            chapters = metadata_store.get_chapters(doc_id=doc_id)
            if chapters:
                console.print(f"\n[bold cyan]Chapters ({len(chapters)}):[/bold cyan]")

                ch_table = Table(show_header=True)
                ch_table.add_column("Num", style="yellow", width=5)
                ch_table.add_column("Title", style="cyan")
                ch_table.add_column("Chunks", justify="right")

                for ch in chapters:
                    start = ch.get('start_chunk_num', 0)
                    end = ch.get('end_chunk_num', 0)
                    chunk_range = f"{start}-{end}" if start != end else str(start)

                    ch_table.add_row(
                        str(ch.get('chapter_num', 'N/A')),
                        ch.get('chapter_title', 'Untitled')[:50],
                        chunk_range
                    )

                console.print(ch_table)
            else:
                console.print("\n[yellow]No chapters detected in this document.[/yellow]")

            # Calculate summary targets
            console.print(f"\n[bold cyan]Summary Targets (ratio={ratio:.0%}):[/bold cyan]")
            total_tokens = doc_info['total_tokens'] or 0
            target_tokens = int(total_tokens * ratio)
            target_words = int(target_tokens * 0.75)

            console.print(f"  Original: ~{total_tokens:,} tokens")
            console.print(f"  Target:   ~{target_tokens:,} tokens (~{target_words:,} words)")
            console.print(f"  Style:    {style.upper()}")

            console.print("\n[dim]Summarization engine not yet implemented.[/dim]")
            console.print("[dim]Use 'manage_rag.py summarize --doc-id <id>' when available.[/dim]")
            return

        # Summarize mode (not yet implemented)
        if doc_id and not analyze:
            console.print(f"\n[bold cyan]📝 Summarizing Document: {doc_id}[/bold cyan]")
            console.print(f"[yellow]Style:[/yellow] {style.upper()}")
            console.print(f"[yellow]Compression:[/yellow] {ratio:.0%}")

            # Check document exists
            doc_info = metadata_store.get_document(doc_id=doc_id)
            if not doc_info:
                console.print(f"[red]❌ Document not found: {doc_id}[/red]")
                return

            console.print(f"\n[green]Document:[/green] {doc_info['title']}")
            console.print(f"[green]Tokens:[/green] {doc_info['total_tokens']:,}")

            console.print("\n[yellow]⚠️  Summarization engine not yet implemented.[/yellow]")
            console.print("[dim]Parallel Summarization Engine (Parallel Summarization Engine) is needed.[/dim]")
            console.print("[dim]Current progress: M1 (Infrastructure) complete.[/dim]")
            console.print("\n[cyan]Coming soon:[/cyan]")
            console.print("  - Parallel chapter-by-chapter summarization")
            console.print("  - Narrative style output (not bullet points)")
            console.print("  - ~5-10 minute processing for large books")
            return

        # No action specified
        console.print("[yellow]Please specify an action:[/yellow]")
        console.print("  --list           List documents available for summarization")
        console.print("  --doc-id <id>    Summarize a specific document")
        console.print("  --analyze        Analyze document structure before summarizing")
        console.print("\n[dim]Example: manage_rag.py summarize --list[/dim]")
        console.print("[dim]Example: manage_rag.py summarize --doc-id doc_xyz --analyze[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced RAG Database Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stats                          # Show database statistics
  %(prog)s metadata                       # Show metadata database
  %(prog)s list                           # List all collections
  %(prog)s list --db private              # List private collections only
  %(prog)s inspect my_collection          # Inspect a collection
  %(prog)s inspect my_collection --limit 3
  %(prog)s query my_collection "search term"
  %(prog)s test-query "quantum physics"   # Test knowledge query
  %(prog)s test-query "auth" --quality accurate
  %(prog)s redetect-languages             # Re-detect low-confidence chunks
  %(prog)s redetect-languages --dry-run   # Preview changes
  %(prog)s reset --confirm                # Reset all databases
  %(prog)s health                         # System health check
        """
    )
    
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    # Metadata command
    subparsers.add_parser('metadata', help='Show metadata statistics')

    # Index file command (NEW)
    index_parser = subparsers.add_parser('index', help='Index a single file')
    index_parser.add_argument('file', help='File path to index')
    index_parser.add_argument('--collection', choices=['private', 'public'], 
                             default='private', help='Collection name')
    index_parser.add_argument('--force', action='store_true', 
                             help='Re-index even if already indexed')
    
    # Index folder command (NEW)
    folder_parser = subparsers.add_parser('index-folder', help='Index all files in folder')
    folder_parser.add_argument('folder', help='Folder path to index')
    folder_parser.add_argument('--collection', choices=['private', 'public'], 
                              default='private', help='Collection name')
    folder_parser.add_argument('--recursive', action='store_true', default=True,
                              help='Search subfolders (default: True)')
    folder_parser.add_argument('--no-recursive', dest='recursive', action='store_false',
                              help='Do not search subfolders')
    folder_parser.add_argument('--force', action='store_true', 
                              help='Re-index even if already indexed')
    folder_parser.add_argument('--extensions', nargs='+', 
                              help='File extensions to index (e.g., .txt .md .py)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List collections')
    list_parser.add_argument('--db', choices=['all', 'private', 'public'], 
                           default='all', help='Which database')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect collection contents')
    inspect_parser.add_argument('collection', help='Collection name')
    inspect_parser.add_argument('--db', choices=['private', 'public'], 
                              default='private', help='Which database')
    inspect_parser.add_argument('--limit', type=int, default=5, 
                              help='Number of documents to show')
    subparsers.add_parser('metadata-detailed', help='Show detailed metadata from SQLite')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query collection')
    query_parser.add_argument('collection', help='Collection name')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('--db', choices=['private', 'public'], 
                            default='private', help='Which database')
    query_parser.add_argument('--top-k', type=int, default=3, 
                            help='Number of results')
    
    # Test query command
    query_test_parser = subparsers.add_parser('test-query', help='Test a knowledge query')
    query_test_parser.add_argument('query', help='Query to test')
    query_test_parser.add_argument('--quality', choices=['fast', 'balanced', 'accurate', 'thorough'],
                                    default='balanced', help='Quality mode')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset databases')
    reset_parser.add_argument('--db', choices=['all', 'private', 'public'], 
                            default='all', help='Which database')
    reset_parser.add_argument('--confirm', action='store_true', 
                            help='Confirm reset (required)')
    
    # Health command
    subparsers.add_parser('health', help='System health check')

    # Re-detect languages command
    redetect_parser = subparsers.add_parser('redetect-languages',
                                            help='Re-detect languages for low-confidence chunks')
    redetect_parser.add_argument('--collection', choices=['private', 'public'],
                                default='private', help='Collection to process')
    redetect_parser.add_argument('--min-confidence', type=float, default=0.5,
                                help='Re-detect if language_confidence < this (default: 0.5)')
    redetect_parser.add_argument('--dry-run', action='store_true',
                                help='Preview changes without updating')

    # Summarize command
    summarize_parser = subparsers.add_parser('summarize', help='Summarize ingested documents')
    summarize_parser.add_argument('--doc-id', help='Document ID to summarize (use --list to see available)')
    summarize_parser.add_argument('--list', action='store_true', dest='list_docs',
                                 help='List documents available for summarization')
    summarize_parser.add_argument('--analyze', action='store_true',
                                 help='Analyze document structure (chapters, tokens)')
    summarize_parser.add_argument('--style', choices=['narrative', 'academic', 'technical', 'quick'],
                                 default='narrative', help='Summary style (default: narrative)')
    summarize_parser.add_argument('--ratio', type=float, default=0.1,
                                 help='Compression ratio (default: 0.1 = 10%%)')

    args = parser.parse_args()
    
    if not args.action:
        parser.print_help()
        return
    
    if args.action == 'stats':
        show_stats()
    elif args.action == 'metadata':
        show_metadata()
    if args.action == 'metadata-detailed':
        show_metadata_detailed()
    elif args.action == 'list':
        list_collections(args.db)
    elif args.action == 'index':
        index_file(args.file, args.collection, args.force)
    elif args.action == 'index-folder':
        index_folder(args.folder, args.collection, args.recursive, 
                    args.force, args.extensions)
    elif args.action == 'inspect':
        inspect_collection(args.collection, args.db, args.limit)
    elif args.action == 'query':
        query_collection(args.collection, args.query, args.db, args.top_k)
    elif args.action == 'test-query':
        test_query(args.query, args.quality)
    elif args.action == 'reset':
        reset_db(args.db, args.confirm)
    elif args.action == 'health':
        health_check()
    elif args.action == 'redetect-languages':
        redetect_languages(args.collection, args.min_confidence, args.dry_run)
    elif args.action == 'summarize':
        summarize_document(
            doc_id=args.doc_id,
            list_docs=args.list_docs,
            analyze=args.analyze,
            style=args.style,
            ratio=args.ratio
        )


if __name__ == "__main__":
    main()