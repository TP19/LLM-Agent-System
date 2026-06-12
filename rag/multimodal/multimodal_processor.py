"""
Multimodal Processor - Main Orchestrator

Auto-detects file types and routes to appropriate processor.
Currently supports PDF files only (audio / video / image processors were
removed in v0.2 to keep the install lean).
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from .pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class MultiModalProcessor:
    """
    Main processor for multimodal file types.

    v0.2 ships PDF only. The router keeps the legacy "multimodal" API shape
    so adding more processors later is a drop-in change.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize multimodal processor.

        Args:
            config_path: Path to config file (default: config/rag_config.yaml)
        """
        self.config_path = config_path or "config/rag_config.yaml"
        self.config = self._load_config()

        # Initialize processors
        pdf_config = self.config.get('multimodal', {}).get('pdf', {})

        self.pdf_processor = PDFProcessor(pdf_config)

        # File type mappings
        self.file_type_map = {
            'pdf': PDFProcessor.SUPPORTED_FORMATS,
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    logger.info(f"Loaded config from {self.config_path}")
                    return config
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def detect_file_type(self, file_path: str) -> Optional[str]:
        """
        Auto-detect file type based on extension.

        Args:
            file_path: Path to file

        Returns:
            File type: 'audio', 'pdf', 'video', 'image', or None if unsupported
        """
        ext = Path(file_path).suffix.lower()

        for file_type, extensions in self.file_type_map.items():
            if ext in extensions:
                return file_type

        return None

    def is_supported(self, file_path: str) -> bool:
        """Check if file type is supported."""
        return self.detect_file_type(file_path) is not None

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Auto-detect file type and process accordingly.

        Args:
            file_path: Path to file

        Returns:
            Dict with keys:
                - file_type: 'audio'|'video'|'pdf'|'image'
                - text: Extracted text content
                - metadata: Format-specific metadata
                - chunks: Pre-chunked for RAG indexing
                - success: bool
                - error: Optional[str]
        """
        result = {
            'file_type': None,
            'text': '',
            'metadata': {},
            'chunks': [],
            'success': False,
            'error': None
        }

        # Validate file existence
        if not os.path.exists(file_path):
            result['error'] = f"File not found: {file_path}"
            logger.error(result['error'])
            return result

        # Detect file type
        file_type = self.detect_file_type(file_path)

        if file_type is None:
            ext = Path(file_path).suffix.lower()
            result['error'] = f"Unsupported file type: {ext}"
            logger.error(result['error'])
            return result

        result['file_type'] = file_type

        try:
            # Route to appropriate processor
            if file_type == 'pdf':
                processor_result = self.process_pdf(file_path)
            else:
                result['error'] = f"Unknown file type: {file_type}"
                return result

            # Merge processor result
            result['text'] = processor_result.get('text', '')
            result['metadata'] = processor_result.get('metadata', {})
            result['success'] = processor_result.get('success', False)
            result['error'] = processor_result.get('error')

            # Generate chunks for RAG
            if result['success']:
                result['chunks'] = self._generate_chunks(file_path, file_type)

            logger.info(
                f"Processed {file_type} file: {Path(file_path).name} - "
                f"success={result['success']}, "
                f"text_length={len(result['text'])}, "
                f"chunks={len(result['chunks'])}"
            )

        except Exception as e:
            result['error'] = f"Processing failed: {str(e)}"
            logger.error(result['error'], exc_info=True)

        return result

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF file using PDFProcessor."""
        logger.info(f"Processing PDF: {pdf_path}")
        return self.pdf_processor.process(pdf_path)

    def _generate_chunks(self, file_path: str, file_type: str, chunk_size: int = 500) -> List[str]:
        """
        Generate text chunks for RAG indexing.

        Args:
            file_path: Path to file
            file_type: Type of file
            chunk_size: Target chunk size in characters

        Returns:
            List of text chunks
        """
        try:
            if file_type == 'pdf':
                return self.pdf_processor.chunk_for_rag(file_path, chunk_size)
            return []
        except Exception as e:
            logger.error(f"Failed to generate chunks: {e}")
            return []

    def process_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple files in batch.

        Args:
            file_paths: List of file paths

        Returns:
            List of results (same format as process_file)
        """
        results = []

        for file_path in file_paths:
            result = self.process_file(file_path)
            results.append(result)

        return results

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get all supported file formats.

        Returns:
            Dict mapping file type to list of supported extensions
        """
        return {
            'pdf': PDFProcessor.SUPPORTED_FORMATS,
        }

    def get_all_supported_extensions(self) -> List[str]:
        """
        Get flat list of all supported extensions.

        Returns:
            List of all supported file extensions
        """
        all_formats = []
        for formats in self.file_type_map.values():
            all_formats.extend(formats)
        return sorted(set(all_formats))


# Utility function for quick testing
def test_multimodal_processor(file_path: str, config_path: Optional[str] = None):
    """
    Quick test function for multimodal processor.

    Args:
        file_path: Path to file
        config_path: Optional path to config file
    """
    processor = MultiModalProcessor(config_path)

    print(f"\n{'='*60}")
    print(f"Multimodal Processor Test")
    print(f"{'='*60}")
    print(f"File: {file_path}")

    # Check if supported
    is_supported = processor.is_supported(file_path)
    print(f"Supported: {is_supported}")

    if not is_supported:
        print(f"\nFile type not supported!")
        print(f"Supported formats:")
        for file_type, formats in processor.get_supported_formats().items():
            print(f"  {file_type}: {', '.join(formats)}")
        return None

    # Detect file type
    file_type = processor.detect_file_type(file_path)
    print(f"Detected type: {file_type}")

    # Process file
    print(f"\nProcessing...")
    result = processor.process_file(file_path)

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Success: {result['success']}")
    print(f"File type: {result['file_type']}")
    print(f"Text length: {len(result['text'])} chars")
    print(f"Chunks: {len(result['chunks'])}")

    if result['metadata']:
        print(f"\nMetadata:")
        for key, value in list(result['metadata'].items())[:10]:
            print(f"  {key}: {value}")

    if result['text']:
        print(f"\nText preview (first 500 chars):\n{result['text'][:500]}")

    if result['chunks']:
        print(f"\nFirst chunk:\n{result['chunks'][0][:200]}")

    if result['error']:
        print(f"\nError: {result['error']}")

    return result


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) > 1:
        file = sys.argv[1]
        config = sys.argv[2] if len(sys.argv) > 2 else None
        test_multimodal_processor(file, config)
    else:
        print("Usage: python multimodal_processor.py <file_path> [config_path]")
        print("Example: python multimodal_processor.py sample.mp3")
        print("Example: python multimodal_processor.py sample.pdf config/rag_config.yaml")
