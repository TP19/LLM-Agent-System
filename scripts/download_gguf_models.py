#!/usr/bin/env python3
"""
GGUF Model Downloader

Downloads recommended GGUF models for embeddings and reranking.

Usage:
    python scripts/download_gguf_models.py [--model MODEL_NAME] [--output DIR]

Examples:
    # Download default embedding model
    python scripts/download_gguf_models.py

    # Download specific model
    python scripts/download_gguf_models.py --model nomic-embed-q8

    # Download to specific directory
    python scripts/download_gguf_models.py --output /path/to/models
"""

import argparse
from pathlib import Path
from typing import Dict, List
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ Error: huggingface-hub not installed")
    print("Install with: pip install huggingface-hub>=0.19.0")
    sys.exit(1)


# Recommended GGUF models for RAG
RECOMMENDED_MODELS = {
    "nomic-embed-q8": {
        "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "file": "nomic-embed-text-v1.5.Q8_0.gguf",
        "size": "275MB",
        "dim": 768,
        "type": "embedding",
        "description": "Best quality embedding model, excellent for code and text"
    },
    "nomic-embed-q6": {
        "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "file": "nomic-embed-text-v1.5.Q6_K.gguf",
        "size": "210MB",
        "dim": 768,
        "type": "embedding",
        "description": "Good balance of speed and quality"
    },
    "nomic-embed-q4": {
        "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "file": "nomic-embed-text-v1.5.Q4_K_M.gguf",
        "size": "140MB",
        "dim": 768,
        "type": "embedding",
        "description": "Fastest option, good quality"
    },
    "bge-large-q8": {
        "repo": "BAAI/bge-large-en-v1.5-GGUF",
        "file": "bge-large-en-v1.5-q8_0.gguf",
        "size": "450MB",
        "dim": 1024,
        "type": "embedding",
        "description": "High-quality embeddings, slower but very accurate"
    },
}


def list_models():
    """Display available models"""
    print("\n📚 Available GGUF Models:\n")
    print(f"{'Name':<20} {'Type':<12} {'Size':<10} {'Dims':<6} {'Description'}")
    print("─" * 90)

    for name, info in RECOMMENDED_MODELS.items():
        print(f"{name:<20} {info['type']:<12} {info['size']:<10} {info['dim']:<6} {info['description']}")

    print("\n💡 Recommendation: Start with 'nomic-embed-q8' for best quality")
    print("   Or use 'nomic-embed-q6' for good balance of speed/quality\n")


def download_model(model_name: str, output_dir: Path) -> bool:
    """
    Download a specific GGUF model

    Args:
        model_name: Model identifier (e.g., 'nomic-embed-q8')
        output_dir: Directory to save the model

    Returns:
        True if successful, False otherwise
    """
    if model_name not in RECOMMENDED_MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {', '.join(RECOMMENDED_MODELS.keys())}")
        return False

    model_info = RECOMMENDED_MODELS[model_name]

    print(f"\n📥 Downloading {model_name}...")
    print(f"   Repository: {model_info['repo']}")
    print(f"   File: {model_info['file']}")
    print(f"   Size: {model_info['size']}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download the file
        downloaded_path = hf_hub_download(
            repo_id=model_info['repo'],
            filename=model_info['file'],
            cache_dir=str(output_dir / ".cache"),
            local_dir=str(output_dir),
            local_dir_use_symlinks=False
        )

        final_path = output_dir / model_info['file']

        print(f"✅ Downloaded successfully!")
        print(f"   Location: {final_path}")
        print(f"\n💡 To use this model, update config/rag_config.yaml:")
        print(f"   embedding:")
        print(f"     model: \"{final_path}\"")
        print()

        return True

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def download_all(output_dir: Path, include_large: bool = False):
    """Download all recommended models"""
    models_to_download = ["nomic-embed-q8", "nomic-embed-q6", "nomic-embed-q4"]

    if include_large:
        models_to_download.append("bge-large-q8")

    print(f"\n📦 Downloading {len(models_to_download)} models to {output_dir}...")

    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name, output_dir):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"✅ Downloaded {success_count}/{len(models_to_download)} models successfully")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download GGUF models for LLM-Agent-System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model to download (e.g., nomic-embed-q8). Use --list to see available models."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".llm_engine" / "models" / "gguf",
        help="Output directory for models (default: ~/.llm_engine/models/gguf)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all recommended models"
    )

    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include large models when using --all"
    )

    args = parser.parse_args()

    # Show header
    print("\n" + "=" * 60)
    print("🔽 GGUF Model Downloader for LLM-Agent-System")
    print("=" * 60)

    # Handle --list
    if args.list:
        list_models()
        return

    # Handle --all
    if args.all:
        download_all(args.output, args.include_large)
        return

    # Handle specific model
    if args.model:
        if download_model(args.model, args.output):
            sys.exit(0)
        else:
            sys.exit(1)

    # Default: show help and list models
    parser.print_help()
    list_models()


if __name__ == "__main__":
    main()
