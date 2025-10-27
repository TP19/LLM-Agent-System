#!/usr/bin/env python3
"""
Enhanced RAG Database Management Script - FIXED IMPORTS

Provides comprehensive database management for vector stores
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
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            metadata = pdf_reader.metadata
            
            if metadata:
                return {
                    'title': metadata.get('/Title', pdf_path.stem),
                    'author': metadata.get('/Author', 'Unknown'),
                    'subject': metadata.get('/Subject'),
                    'creator': metadata.get('/Creator'),
                    'producer': metadata.get('/Producer'),
                    'creation_date': metadata.get('/CreationDate')
                }
    except Exception as e:
        console.print(f"[yellow]⚠ Could not extract PDF metadata: {e}[/yellow]")
    
    return {'title': pdf_path.stem, 'author': 'Unknown'}


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
    if ext in ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs']:
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
    
    # Handle text files
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
                collections = store.client.list_collections()
                
                if not collections:
                    console.print("  [yellow]No collections found[/yellow]")
                    continue
                
                table = Table(show_header=True)
                table.add_column("Collection", style="cyan")
                table.add_column("Documents", justify="right", style="green")
                table.add_column("Created", style="yellow")
                
                for coll in collections:
                    try:
                        count = coll.count()
                        metadata = coll.metadata
                        
                        if metadata and isinstance(metadata, dict):
                            created = metadata.get('created_at', 'Unknown')
                        else:
                            created = 'Unknown'
                        
                        table.add_row(coll.name, str(count), created)
                    except Exception as e:
                        table.add_row(coll.name, "Error", str(e))
                
                console.print(table)
                
            except Exception as e:
                console.print(f"  [red]❌ Error listing collections: {e}[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize stores: {e}[/red]")


def inspect_collection(collection_name: str, db_type: str = "private", limit: int = 5):
    """
    Enhanced inspect: Shows both ChromaDB data AND SQLite metadata
    
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
        
        # Get ChromaDB data
        try:
            collection = store.client.get_collection(collection_name)
            
            # Get sample documents
            results = collection.get(limit=limit, include=['documents', 'metadatas', 'embeddings'])
            
            if not results['ids']:
                console.print("[yellow]Collection is empty[/yellow]")
                return
            
            console.print(f"\n[green]Found {len(results['ids'])} documents (showing first {limit}):[/green]\n")
            
            # Track document IDs to get metadata
            doc_ids_seen = set()
            
            for i, chunk_id in enumerate(results['ids']):
                # ✅ FIX: Safely get metadata
                metadata = {}
                if results.get('metadatas') and i < len(results['metadatas']):
                    metadata = results['metadatas'][i] if results['metadatas'][i] else {}
                
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
        
        embedder = EmbeddingEngine()
        store = store_manager.private_store if db_type == "private" else store_manager.public_store
        
        console.print(f"\n[bold cyan]🔍 Querying Collection: {collection_name}[/bold cyan]")
        console.print(f"[yellow]Query:[/yellow] {query}\n")
        
        try:
            collection = store.client.get_collection(collection_name)
            
            # Generate query embedding
            query_embedding = embedder.embed_text(query)
            
            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['ids'] or not results['ids'][0]:
                console.print("[yellow]No results found[/yellow]")
                return
            
            console.print(f"[green]Found {len(results['ids'][0])} results:[/green]\n")
            
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results.get('distances') else 0.0
                similarity = 1.0 - distance  # Convert distance to similarity
                
                document = results['documents'][0][i] if results.get('documents') else "N/A"
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                
                panel = Panel(
                    f"[green]Similarity:[/green] {similarity:.3f}\n"
                    f"[cyan]ID:[/cyan] {doc_id}\n"
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
        
        embedder = EmbeddingEngine()
        
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
        extensions = ['.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.log', '.pdf']
    
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
    
    FIXED: Use ChromaDB collection.add() instead of non-existent add_texts()
    
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
    supported_extensions = ['.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.log', '.pdf']
    if file_path.suffix.lower() not in supported_extensions:
        console.print(f"[red]❌ Unsupported file type: {file_path.suffix}[/red]")
        console.print(f"[yellow]Supported types: {', '.join(supported_extensions)}[/yellow]")
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    console.print(f"\n[bold cyan]📄 Indexing File[/bold cyan]")
    console.print(f"File: {file_path}")
    console.print(f"Type: {file_path.suffix}")
    console.print(f"Collection: {collection}\n")
    
    try:
        # Initialize components
        base_path = get_db_path()
        embedder = EmbeddingEngine(lazy_load=False)
        import torch

        console.print("[bold yellow]🔍 DEBUG: Checking GPU usage...[/bold yellow]")
        console.print(f"   CUDA Available: {torch.cuda.is_available()}")
        console.print(f"   Config Device: {embedder.device}")
        console.print(f"   Config Batch Size: {embedder.batch_size}")
        console.print(f"   Model Name: {embedder.model_name}")

        if embedder.model is not None:
            console.print(f"   Model Device (actual): {embedder.model.device}")
            
            # Check GPU memory
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                console.print(f"   GPU Memory Allocated: {mem_allocated:.2f} GB")
                console.print(f"   GPU Memory Reserved: {mem_reserved:.2f} GB")
                
                if mem_allocated > 0.5:
                    console.print(f"   [green]✅ GPU is being used![/green]")
                else:
                    console.print(f"   [red]⚠️  GPU might not be in use (low memory)[/red]")
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
                coll = store.client.get_or_create_collection(name=collection_name)
                existing = coll.get(ids=[doc_id])
                
                if existing and existing.get('ids'):
                    console.print(f"[yellow]⚠️  File already indexed: {doc_id}[/yellow]")
                    console.print(f"[yellow]   Use --force to re-index[/yellow]")
                    return "skipped"
            except Exception as e:
                console.print(f"[dim]Could not check existing: {e}[/dim]")
        
        # Extract PDF metadata if applicable
        pdf_metadata = {}
        if file_path.suffix.lower() == '.pdf':
            console.print("[dim]Extracting PDF metadata...[/dim]")
            pdf_metadata = extract_pdf_metadata(file_path)
            if pdf_metadata and pdf_metadata.get('title') != file_path.stem:
                console.print(f"[green]✓[/green] Found: {pdf_metadata.get('title', 'Unknown Title')}")
        
        # Read file content
        console.print("[dim]Reading file content...[/dim]")
        content = read_file(file_path)
        console.print(f"[green]✓[/green] Read {len(content)} characters")
        
        # Detect chapters
        console.print("[dim]Detecting document structure...[/dim]")
        chapters = detect_chapters(content, file_path.name)
        if chapters:
            console.print(f"[green]✓[/green] Found {len(chapters)} chapters")
        
        # Chunk the text using semantic chunker
        console.print("[dim]Creating semantic chunks...[/dim]")
        chunker = SemanticChunker()
        token_counter = TokenCounter()
        
        # chunk_by_tokens returns List[str], not List[dict]
        chunk_strings = chunker.chunk_by_tokens(content, chunk_size=500, overlap=50)
        console.print(f"[green]✓[/green] Created {len(chunk_strings)} chunks")
        
        # Convert strings to proper format with metadata
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
                'has_code': detect_code(chunk['text']),
                'indexed_at': datetime.now().isoformat()
            }
            
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
        
        # ✅ FIX: Store in vector database using ChromaDB collection API
        console.print("[dim]Storing in vector database...[/dim]")
        
        collection_name = f"{store.collection_prefix}_documents"
        chroma_collection = store.client.get_or_create_collection(
            name=collection_name
        )
        
        # Add to ChromaDB using the correct API
        chroma_collection.add(
            ids=[m['chunk_id'] for m in metadata_list],
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadata_list
        )
        
        console.print(f"[green]✓[/green] Stored in vector database ({collection_name})")
        
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

        existing_doc = metadata_store.get_document(file_path=str(file_path))

        if force:
            existing_doc = metadata_store.get_document(file_path=str(file_path))
            if existing_doc:
                console.print(f"[dim]Removing old metadata for re-indexing...[/dim]")
                metadata_store.delete_document(doc_id=existing_doc['doc_id'])
        
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


if __name__ == "__main__":
    main()