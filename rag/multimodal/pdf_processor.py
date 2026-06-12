"""
PDF Processing Module - Text and Table Extraction

Supports PDF text extraction with layout preservation, table extraction,
and multi-column handling.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Process PDF files to extract text and tables.

    Features:
    - Text extraction with layout preservation
    - Table extraction (markdown format)
    - Multi-column handling
    - Page number tracking
    - Encrypted PDF support (with password)
    """

    SUPPORTED_FORMATS = ['.pdf']

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PDF processor with optional config.

        Args:
            config: Configuration dict with keys:
                - preserve_layout: Preserve text layout (default: True)
                - extract_tables: Extract tables in markdown format (default: True)
                - extract_images: Extract images (default: False, future enhancement)
        """
        self.config = config or {}
        self.preserve_layout = self.config.get('preserve_layout', True)
        self.extract_tables = self.config.get('extract_tables', True)
        self.extract_images = self.config.get('extract_images', False)

    def is_supported(self, file_path: str) -> bool:
        """Check if file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_FORMATS

    def process(self, pdf_path: str, password: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text and tables from PDF.

        Args:
            pdf_path: Path to PDF file
            password: Optional password for encrypted PDFs

        Returns:
            Dict with keys:
                - success: bool
                - text: Extracted text
                - pages: List of page texts
                - tables: List of extracted tables (if extract_tables=True)
                - metadata: PDF metadata (title, author, pages, etc.)
                - error: Error message if failed
        """
        result = {
            'success': False,
            'text': '',
            'pages': [],
            'tables': [],
            'metadata': {},
            'error': None
        }

        # Validate file
        if not os.path.exists(pdf_path):
            result['error'] = f"PDF file not found: {pdf_path}"
            logger.error(result['error'])
            return result

        if not self.is_supported(pdf_path):
            result['error'] = f"Not a PDF file: {pdf_path}"
            logger.error(result['error'])
            return result

        try:
            # Try pdfplumber first (better for tables and layout)
            if self.extract_tables:
                result = self._process_with_pdfplumber(pdf_path, password)
            else:
                # Fallback to PyPDF2 (faster, simpler)
                result = self._process_with_pypdf2(pdf_path, password)

            if result['success']:
                logger.info(
                    f"PDF processing successful: {len(result['pages'])} pages, "
                    f"{len(result['text'])} chars, {len(result['tables'])} tables"
                )

        except Exception as e:
            result['error'] = f"PDF processing failed: {str(e)}"
            logger.error(result['error'], exc_info=True)

        return result

    def _process_with_pdfplumber(self, pdf_path: str, password: Optional[str] = None) -> Dict[str, Any]:
        """Process PDF with pdfplumber (better for tables and layout)."""
        result = {
            'success': False,
            'text': '',
            'pages': [],
            'tables': [],
            'metadata': {},
            'error': None
        }

        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed, falling back to PyPDF2")
            return self._process_with_pypdf2(pdf_path, password)

        try:
            with pdfplumber.open(pdf_path, password=password) as pdf:
                # Extract metadata
                result['metadata'] = {
                    'file_path': pdf_path,
                    'file_name': Path(pdf_path).name,
                    'file_size': os.path.getsize(pdf_path),
                    'num_pages': len(pdf.pages),
                    'pdf_metadata': pdf.metadata or {}
                }

                # Extract text and tables from each page
                all_text_parts = []

                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text
                    page_text = page.extract_text() or ''

                    # Add page marker
                    page_header = f"\n--- Page {page_num} ---\n"
                    all_text_parts.append(page_header)
                    all_text_parts.append(page_text)

                    result['pages'].append({
                        'page_num': page_num,
                        'text': page_text,
                        'char_count': len(page_text)
                    })

                    # Extract tables if enabled
                    if self.extract_tables:
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables or []):
                            if table:
                                # Convert table to markdown
                                md_table = self._table_to_markdown(table)
                                all_text_parts.append(f"\n[Table {table_idx + 1}]\n")
                                all_text_parts.append(md_table)

                                result['tables'].append({
                                    'page_num': page_num,
                                    'table_num': table_idx + 1,
                                    'markdown': md_table,
                                    'rows': len(table),
                                    'cols': len(table[0]) if table else 0
                                })

                # Combine all text
                result['text'] = '\n'.join(all_text_parts).strip()
                result['success'] = True

        except Exception as e:
            result['error'] = f"pdfplumber processing failed: {str(e)}"
            logger.error(result['error'])
            # Try fallback
            return self._process_with_pypdf2(pdf_path, password)

        return result

    def _process_with_pypdf2(self, pdf_path: str, password: Optional[str] = None) -> Dict[str, Any]:
        """Process PDF with PyPDF2 (simpler, faster)."""
        result = {
            'success': False,
            'text': '',
            'pages': [],
            'tables': [],
            'metadata': {},
            'error': None
        }

        try:
            from PyPDF2 import PdfReader
        except ImportError:
            result['error'] = "PyPDF2 not installed. Install with: pip install PyPDF2"
            logger.error(result['error'])
            return result

        try:
            reader = PdfReader(pdf_path)

            # Handle encryption
            if reader.is_encrypted:
                if password:
                    reader.decrypt(password)
                else:
                    result['error'] = "PDF is encrypted but no password provided"
                    return result

            # Extract metadata
            metadata = reader.metadata or {}
            result['metadata'] = {
                'file_path': pdf_path,
                'file_name': Path(pdf_path).name,
                'file_size': os.path.getsize(pdf_path),
                'num_pages': len(reader.pages),
                'pdf_metadata': {
                    'title': metadata.get('/Title', ''),
                    'author': metadata.get('/Author', ''),
                    'subject': metadata.get('/Subject', ''),
                    'creator': metadata.get('/Creator', '')
                }
            }

            # Extract text from each page
            all_text_parts = []

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ''

                # Add page marker
                page_header = f"\n--- Page {page_num} ---\n"
                all_text_parts.append(page_header)
                all_text_parts.append(page_text)

                result['pages'].append({
                    'page_num': page_num,
                    'text': page_text,
                    'char_count': len(page_text)
                })

            # Combine all text
            result['text'] = '\n'.join(all_text_parts).strip()
            result['success'] = True

        except Exception as e:
            result['error'] = f"PyPDF2 processing failed: {str(e)}"
            logger.error(result['error'])

        return result

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """
        Convert table data to markdown format.

        Args:
            table: 2D list of table cells

        Returns:
            Markdown-formatted table string
        """
        if not table or not table[0]:
            return ""

        # Clean cells (replace None with empty string)
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else '' for cell in row]
            cleaned_table.append(cleaned_row)

        # Calculate column widths
        num_cols = len(cleaned_table[0])
        col_widths = [0] * num_cols

        for row in cleaned_table:
            for i, cell in enumerate(row[:num_cols]):
                col_widths[i] = max(col_widths[i], len(cell))

        # Build markdown table
        lines = []

        # Header row
        if cleaned_table:
            header = cleaned_table[0]
            header_line = '| ' + ' | '.join(
                cell.ljust(col_widths[i]) for i, cell in enumerate(header[:num_cols])
            ) + ' |'
            lines.append(header_line)

            # Separator
            separator = '| ' + ' | '.join('-' * w for w in col_widths) + ' |'
            lines.append(separator)

            # Data rows
            for row in cleaned_table[1:]:
                row_line = '| ' + ' | '.join(
                    cell.ljust(col_widths[i]) for i, cell in enumerate(row[:num_cols])
                ) + ' |'
                lines.append(row_line)

        return '\n'.join(lines)

    def chunk_for_rag(self, pdf_path: str, chunk_size: int = 500) -> List[str]:
        """
        Process PDF and split into chunks for RAG indexing.

        Args:
            pdf_path: Path to PDF file
            chunk_size: Target chunk size in characters (default: 500)

        Returns:
            List of text chunks
        """
        result = self.process(pdf_path)

        if not result['success']:
            logger.warning(f"Cannot chunk PDF - processing failed: {result['error']}")
            return []

        # Split text into chunks
        text = result['text']
        chunks = []

        # Try to chunk by pages first
        if result['pages']:
            for page_info in result['pages']:
                page_text = page_info['text']

                # If page is small enough, keep as one chunk
                if len(page_text) <= chunk_size:
                    if page_text.strip():
                        chunks.append(page_text.strip())
                else:
                    # Split large pages into smaller chunks
                    page_chunks = self._split_text(page_text, chunk_size)
                    chunks.extend(page_chunks)
        else:
            # Fallback: simple text chunking
            chunks = self._split_text(text, chunk_size)

        logger.info(f"Created {len(chunks)} chunks from PDF ({len(result['pages'])} pages)")
        return chunks

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size."""
        chunks = []
        paragraphs = text.split('\n\n')

        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_length = len(para)

            # If adding this paragraph exceeds chunk_size, finalize current chunk
            if current_length + para_length > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0

            # If single paragraph is too large, split by sentences
            if para_length > chunk_size * 1.5:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_length = len(sent)

                    if current_length + sent_length > chunk_size and current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_length = 0

                    current_chunk.append(sent)
                    current_length += sent_length
            else:
                current_chunk.append(para)
                current_length += para_length

        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks


# Utility function for quick testing
def test_pdf_processor(pdf_path: str):
    """
    Quick test function for PDF processor.

    Args:
        pdf_path: Path to PDF file
    """
    processor = PDFProcessor()
    result = processor.process(pdf_path)

    print(f"\n{'='*60}")
    print(f"PDF Processing Results")
    print(f"{'='*60}")
    print(f"Success: {result['success']}")
    print(f"Pages: {len(result['pages'])}")
    print(f"Tables: {len(result['tables'])}")
    print(f"Text length: {len(result['text'])} chars")

    if result['metadata']:
        print(f"\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")

    print(f"\nFirst 500 chars of text:\n{result['text'][:500]}")

    if result['tables']:
        print(f"\nFirst table:\n{result['tables'][0]['markdown']}")

    if result['error']:
        print(f"\nError: {result['error']}")

    return result


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        test_pdf_processor(pdf_file)
    else:
        print("Usage: python pdf_processor.py <pdf_file>")
        print("Example: python pdf_processor.py sample.pdf")
