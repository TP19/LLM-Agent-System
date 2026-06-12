#!/usr/bin/env python3
"""
LLM-Agent-System - Installer (Linux only)

This script:
1. Detects hardware (GPU type + VRAM)
2. Creates virtual environment
3. Installs dependencies with correct GPU support
4. Installs llama-cpp-python with CUDA support
5. Installs LanceDB for vector storage
6. Optionally downloads recommended models
7. Auto-configures models.yaml based on detected hardware
"""

import sys
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import json

VERSION = "0.2.0"

# Pinned llama-server release for pre-built binary download
LLAMA_RELEASE = "b7757"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header():
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.CYAN}  LLM-Agent-System - Installer{Colors.END}")
    print(f"  Version: {VERSION}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}\n")


def print_ok(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.END} {msg}")


def print_info(msg: str):
    print(f"{Colors.BLUE}[...]{Colors.END} {msg}")


def print_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.END} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.END} {msg}")


def detect_os() -> Tuple[str, str, bool]:
    """Detect operating system and environment."""
    os_type = platform.system().lower()
    arch = platform.machine().lower()
    is_wsl = False
    if os_type == 'linux':
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    is_wsl = True
        except:
            pass
    return os_type, arch, is_wsl


def get_gpu_vram_mb() -> Optional[int]:
    """Get total GPU VRAM in MB (NVIDIA only)."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except (FileNotFoundError, ValueError):
        pass
    return None


def recommend_gpu_layers(vram_mb: Optional[int], model_size_b: int = 20) -> int:
    """Recommend n_gpu_layers based on available VRAM.

    For a 20B Q5_K_M model (~14GB), approximate layer usage:
    - Full offload (99 layers): ~14GB VRAM
    - 40 layers: ~8GB VRAM
    - 24 layers: ~5GB VRAM (leaves room for context)
    - 10 layers: ~3GB VRAM
    """
    if vram_mb is None:
        return 0  # CPU only

    if vram_mb >= 16000:  # 16GB+
        return 99  # Full offload
    elif vram_mb >= 12000:  # 12GB
        return 60
    elif vram_mb >= 8000:  # 8GB
        return 40
    elif vram_mb >= 6000:  # 6GB
        return 30
    elif vram_mb >= 4000:  # 4GB
        return 20
    else:  # < 4GB
        return 10


def detect_gpu() -> Tuple[Optional[str], str]:
    """Detect GPU type for acceleration (Linux only)."""
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return 'nvidia', '-DGGML_CUDA=on'
    except FileNotFoundError:
        pass

    # Check for AMD GPU (ROCm)
    rocm_path = Path('/opt/rocm')
    if rocm_path.exists():
        return 'amd', '-DGGML_HIPBLAS=on'

    return None, ''


def check_python_version():
    """Check if Python version is 3.10+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_error(f"Python 3.10+ required. Found: {version.major}.{version.minor}")
        sys.exit(1)
    print_ok(f"Python version: {version.major}.{version.minor}.{version.micro}")


def get_project_root() -> Path:
    return Path(__file__).parent.absolute()


def get_pip_path(venv_path: Path) -> Path:
    return venv_path / "bin" / "pip"


def get_python_path(venv_path: Path) -> Path:
    return venv_path / "bin" / "python"


def create_venv(project_root: Path) -> Path:
    """Create virtual environment."""
    venv_path = project_root / ".venv"
    if venv_path.exists():
        print_ok(f"Virtual environment exists: {venv_path}")
        return venv_path

    print_info(f"Creating virtual environment at {venv_path}")
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    print_ok("Virtual environment created")
    return venv_path


def run_pip(pip_path: Path, args: list, label: str, show_output: bool = True):
    """Run pip with visible progress output."""
    cmd = [str(pip_path)] + args
    if show_output:
        # Show pip output in real-time so user sees progress bars
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def install_requirements(venv_path: Path, project_root: Path):
    """Install base requirements."""
    pip_path = get_pip_path(venv_path)

    print_info("Upgrading pip...")
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], capture_output=True)
    print_ok("pip upgraded")

    requirements = project_root / "requirements.txt"
    if requirements.exists():
        print_info("Installing requirements (this may take a few minutes)...")
        print_info("You'll see pip's download and install progress below:")
        print("")
        result = run_pip(pip_path, ["install", "-r", str(requirements)], "requirements")
        print("")
        if result.returncode == 0:
            print_ok("Requirements installed")
        else:
            print_warn("Some requirements failed to install")
            if hasattr(result, 'stderr') and result.stderr:
                print_warn(f"Details: {result.stderr[-300:]}")



def install_llama_server(project_root: Path) -> bool:
    """Download pre-built llama-server binary from llama.cpp releases.

    Installs to project_root/bin/llama-server with shared libraries.
    Falls back to build-from-source instructions if platform is unsupported.
    """
    os_type, arch, _ = detect_os()

    # Determine platform-specific archive
    os_type, arch, _ = detect_os()
    gpu_type, _ = detect_gpu()

    platform_key = f"{os_type}-{arch}"

    # On Linux x86_64 with GPU, prefer Vulkan build (supports NVIDIA/AMD/Intel)
    if platform_key == "linux-x86_64" and gpu_type:
        archives = {
            "linux-x86_64": f"llama-{LLAMA_RELEASE}-bin-ubuntu-vulkan-x64.tar.gz",
        }
        print_info(f"GPU detected ({gpu_type}) - using Vulkan build for GPU offload")
    else:
        archives = {}

    # Linux platform archives
    archives.setdefault("linux-x86_64", f"llama-{LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz")
    archives.setdefault("linux-aarch64", f"llama-{LLAMA_RELEASE}-bin-ubuntu-arm64.tar.gz")

    archive_name = archives.get(platform_key)
    if not archive_name:
        print_warn(f"No pre-built binary for {platform_key}")
        print_info("Build from source:")
        print_info("  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        print_info("  cd ~/llama.cpp && cmake -B build && cmake --build build -j")
        return False

    bin_dir = project_root / "bin"
    bin_dir.mkdir(exist_ok=True)
    server_path = bin_dir / "llama-server"

    # Check if already installed
    if server_path.exists():
        print_ok(f"llama-server already installed: {server_path}")
        return True

    url = f"https://github.com/ggerganov/llama.cpp/releases/download/{LLAMA_RELEASE}/{archive_name}"
    print_info(f"Downloading llama-server {LLAMA_RELEASE} for {platform_key}...")
    print_info(f"URL: {url}")

    import tempfile
    import tarfile

    temp_dir = Path(tempfile.mkdtemp())
    archive_path = temp_dir / archive_name

    try:
        # Download
        if shutil.which("wget"):
            result = subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(archive_path), url]
            )
        elif shutil.which("curl"):
            result = subprocess.run(
                ["curl", "-L", "--progress-bar", "-o", str(archive_path), url]
            )
        else:
            print_info("Downloading with Python (no progress bar)...")
            urllib.request.urlretrieve(url, archive_path)
            result = type('R', (), {'returncode': 0})()

        if result.returncode != 0:
            print_error("Download failed")
            return False

        # Extract
        print_info("Extracting...")
        extract_dir = temp_dir / "extract"
        extract_dir.mkdir()

        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)

        # Find extracted directory (usually llama-bXXXX/)
        extracted = list(extract_dir.iterdir())
        if len(extracted) == 1 and extracted[0].is_dir():
            src_dir = extracted[0]
        else:
            src_dir = extract_dir

        # Copy llama-server binary
        src_server = src_dir / "llama-server"
        if not src_server.exists():
            # Try bin/ subdirectory
            src_server = src_dir / "bin" / "llama-server"
        if not src_server.exists():
            print_error("llama-server not found in archive")
            return False

        shutil.copy2(src_server, server_path)
        server_path.chmod(0o755)

        # Copy shared libraries
        lib_count = 0
        for lib in src_dir.glob("*.so*"):
            shutil.copy2(lib, bin_dir / lib.name)
            lib_count += 1

        print_ok(f"Installed: {server_path}")
        if lib_count > 0:
            print_ok(f"Copied {lib_count} shared libraries")

        # Verify binary works
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(bin_dir) + ':' + env.get('LD_LIBRARY_PATH', '')
        try:
            test = subprocess.run(
                [str(server_path), "--version"],
                env=env, capture_output=True, text=True, timeout=5
            )
            if test.returncode == 0:
                version_line = test.stdout.strip().split('\n')[0] if test.stdout else "unknown"
                print_ok(f"Verified: {version_line}")
        except Exception:
            print_warn("Binary installed but could not verify (may need GPU libraries)")

        return True

    except Exception as e:
        print_error(f"Installation failed: {e}")
        print_info("Build from source instead:")
        print_info("  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        print_info("  cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_llama_server_cuda(project_root: Path) -> bool:
    """Build llama-server from source with CUDA support.

    Requires: git, cmake, build-essential, CUDA toolkit.
    Produces: project_root/bin/llama-server with CUDA GPU acceleration.
    """
    bin_dir = project_root / "bin"
    bin_dir.mkdir(exist_ok=True)
    server_path = bin_dir / "llama-server"

    if server_path.exists():
        print_ok(f"llama-server already exists: {server_path}")
        if not ask_yes_no("Rebuild from source?", default=False):
            return True

    # Check prerequisites
    for tool in ['git', 'cmake', 'make']:
        if not shutil.which(tool):
            print_error(f"{tool} not found. Install it first.")
            return False

    # Check for CUDA
    nvcc = shutil.which('nvcc')
    if not nvcc:
        cuda_paths = ['/usr/local/cuda/bin/nvcc', '/usr/local/cuda-12.1/bin/nvcc']
        for p in cuda_paths:
            if os.path.isfile(p):
                nvcc = p
                break
    if not nvcc:
        print_error("CUDA toolkit not found (nvcc not in PATH)")
        print_info("Install CUDA first, or use option 1 (Vulkan pre-built)")
        return False

    print_ok(f"CUDA compiler: {nvcc}")

    build_dir = project_root / ".build-llama"
    llama_src = build_dir / "llama.cpp"

    try:
        build_dir.mkdir(exist_ok=True)

        # Clone or update
        if (llama_src / ".git").exists():
            print_info("Updating llama.cpp source...")
            subprocess.run(["git", "fetch", "--tags"], cwd=str(llama_src))
        else:
            print_info("Cloning llama.cpp...")
            subprocess.run(
                ["git", "clone", "--depth=1", "--branch", LLAMA_RELEASE,
                 "https://github.com/ggerganov/llama.cpp", str(llama_src)],
                check=True
            )

        # Build with CUDA
        cmake_build = llama_src / "build"
        print_info("Building with CUDA support (this takes ~5-10 minutes)...")

        env = os.environ.copy()
        cuda_home = os.path.dirname(os.path.dirname(nvcc))
        env['CUDA_HOME'] = cuda_home
        env['CUDACXX'] = nvcc

        subprocess.run(
            ["cmake", "-B", str(cmake_build), "-DGGML_CUDA=ON",
             "-DCMAKE_CUDA_ARCHITECTURES=all-major"],
            cwd=str(llama_src), env=env, check=True
        )

        subprocess.run(
            ["cmake", "--build", str(cmake_build), "-j", "--target", "llama-server"],
            cwd=str(llama_src), env=env, check=True
        )

        # Copy binary and libs to bin/
        built_server = cmake_build / "bin" / "llama-server"
        if not built_server.exists():
            built_server = cmake_build / "llama-server"
        if not built_server.exists():
            print_error("Build succeeded but llama-server binary not found")
            return False

        shutil.copy2(built_server, server_path)
        server_path.chmod(0o755)

        # Copy CUDA shared libraries
        lib_count = 0
        for lib in cmake_build.rglob("*.so*"):
            if 'ggml' in lib.name or 'llama' in lib.name:
                shutil.copy2(lib, bin_dir / lib.name)
                lib_count += 1

        print_ok(f"Built and installed: {server_path}")
        if lib_count > 0:
            print_ok(f"Copied {lib_count} shared libraries")

        return True

    except subprocess.CalledProcessError as e:
        print_error(f"Build failed: {e}")
        print_info("Try option 1 (Vulkan pre-built) instead")
        return False
    except Exception as e:
        print_error(f"Build failed: {e}")
        return False


def install_lancedb(venv_path: Path):
    """Install LanceDB for vector storage."""
    pip_path = get_pip_path(venv_path)

    print_info("Installing LanceDB and PyArrow...")
    result = run_pip(pip_path, ["install", "lancedb", "pyarrow"], "LanceDB")
    if result.returncode == 0:
        print_ok("LanceDB installed")
    else:
        print_warn("LanceDB installation failed")



def download_model(url: str, dest_path: Path, model_name: str):
    """Download a model file with progress."""
    if dest_path.exists():
        print_ok(f"Model already exists: {model_name}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print_info(f"Downloading {model_name}...")
    print_info(f"URL: {url}")
    print_info(f"Destination: {dest_path}")

    try:
        # Use wget or curl if available for better progress
        if shutil.which("wget"):
            result = subprocess.run(
                ["wget", "-O", str(dest_path), url],
                capture_output=False
            )
            if result.returncode == 0:
                print_ok(f"Downloaded: {model_name}")
                return True
        elif shutil.which("curl"):
            result = subprocess.run(
                ["curl", "-L", "-o", str(dest_path), url],
                capture_output=False
            )
            if result.returncode == 0:
                print_ok(f"Downloaded: {model_name}")
                return True

        # Fallback to Python urllib
        urllib.request.urlretrieve(url, dest_path)
        print_ok(f"Downloaded: {model_name}")
        return True
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False


def download_gguf_via_hf(repo_id: str, filename: str, dest_dir: Path) -> bool:
    """Download a GGUF model from HuggingFace using huggingface_hub.

    Resumable, progress bars, atomic — much better than wget/curl for GGUF files.
    Falls back to a clear error if huggingface_hub isn't installed yet.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print_warn("huggingface_hub not installed in the venv — falling back to wget/curl")
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        return download_model(url, dest_dir / filename, filename)

    dest_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"Downloading {filename}")
    print_info(f"  Repository: {repo_id}")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
        print_ok(f"Downloaded: {path}")
        return True
    except Exception as e:
        print_error(f"hf_hub_download failed: {e}")
        return False


def _wire_embedding_to_local_gguf(project_root: Path, embed_file: Path):
    """Point rag_config.yaml's embedding.model at the downloaded GGUF.

    Idempotent — only switches the active default if it's still pointing at
    the HuggingFace identifier. Preserves any custom path the user set.
    """
    cfg = project_root / "config" / "rag_config.yaml"
    if not cfg.exists():
        return
    text = cfg.read_text()
    hf_default = 'model: "Qwen/Qwen3-Embedding-0.6B"'
    gguf_line = f'model: "{embed_file}"'
    if hf_default in text:
        text = text.replace(hf_default, gguf_line, 1)
        cfg.write_text(text)
        print_ok(f"rag_config.yaml embedding now points at: {embed_file}")
    else:
        print_info(f"rag_config.yaml embedding already customized — left as-is. "
                   f"To use the new GGUF: set embedding.model: \"{embed_file}\"")


def configure_models_yaml(project_root: Path, gpu_type: Optional[str], model_path: Optional[str] = None):
    """Auto-configure models.yaml based on detected hardware."""
    config_file = project_root / "config" / "models.yaml"
    config_file.parent.mkdir(exist_ok=True)

    vram_mb = get_gpu_vram_mb() if gpu_type == 'nvidia' else None
    n_gpu_layers = recommend_gpu_layers(vram_mb)

    if vram_mb:
        print_ok(f"GPU VRAM: {vram_mb} MB")
    print_ok(f"Recommended GPU layers: {n_gpu_layers}")

    if not model_path:
        # Project-local default; install.py drops downloaded models here.
        model_path = str(project_root / "models" / "nvidia_Orchestrator-8B-Q5_K_M.gguf")

    # Smaller context for low-VRAM systems
    oracle_ctx = 16384 if (vram_mb and vram_mb >= 12000) else 8192
    agent_ctx = 8192 if (vram_mb and vram_mb >= 8000) else 4096

    config_content = f"""# LLM-Agent-System - Model Configuration
# Auto-configured by install.py
# GPU: {gpu_type or 'CPU'}{f' ({vram_mb} MB VRAM)' if vram_mb else ''}
# GPU layers: {n_gpu_layers}

models:
  oracle:
    model_path: "{model_path}"
    backend: "llama-server"
    n_ctx: {oracle_ctx}
    n_gpu_layers: {n_gpu_layers}
    verbose: false
    temperature: 0.1
    top_p: 0.9
    repeat_penalty: 1.1
    use_chat_api: true
    enable_thinking: true
  operator:
    model_path: "{model_path}"
    backend: "llama-server"
    n_ctx: {agent_ctx}
    n_gpu_layers: {n_gpu_layers}
    verbose: false
    temperature: 0.4
    top_p: 0.9
    top_k: 40
    repeat_penalty: 1.1
    use_chat_api: true
  security:
    model_path: "{model_path}"
    backend: "llama-server"
    n_ctx: {agent_ctx}
    n_gpu_layers: {n_gpu_layers}
    verbose: false
    temperature: 0.3
    top_p: 0.9
    use_chat_api: true
  coder:
    model_path: "{model_path}"
    backend: "llama-server"
    n_ctx: {oracle_ctx}
    n_gpu_layers: {n_gpu_layers}
    verbose: false
    temperature: 0.3
    top_p: 0.9
    use_chat_api: true
  summarization:
    model_path: "{model_path}"
    backend: "llama-server"
    n_ctx: {agent_ctx}
    n_gpu_layers: {n_gpu_layers}
    verbose: false
    enable_rag: true
    temperature: 0.7
    top_p: 0.9
    repeat_penalty: 1.05
    use_chat_api: true
  knowledge:
    model_path: "{model_path}"
    backend: "llama-server"
    n_ctx: {agent_ctx}
    n_gpu_layers: {n_gpu_layers}
    verbose: false
    temperature: 0.3
    top_p: 0.9
    repeat_penalty: 1.1
    use_chat_api: true
  triage:
    model_path: "~/models/gemma-3-4b-it-abliterated.q8_0.gguf"
    n_ctx: 2048
    n_gpu_layers: {min(n_gpu_layers, 24)}
    verbose: false
    temperature: 0.3
    top_p: 0.9
settings:
  command_timeout: 30
  enable_follow_up: true
  max_collaboration_cycles: 3
"""
    config_file.write_text(config_content)
    print_ok(f"Generated config/models.yaml (GPU layers: {n_gpu_layers})")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        response = input(prompt + suffix).strip().lower()
    except EOFError:
        return default
    if not response:
        return default
    return response in ('y', 'yes')


def create_launcher(project_root: Path, venv_path: Path):
    """Create launcher scripts."""
    launcher = project_root / "llm-agent-system"
    launcher.write_text(f"""#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate
python start_console.py "$@"
""")
    launcher.chmod(0o755)
    print_ok(f"Created launcher: {launcher}")

    # Create symlink
    local_bin = Path.home() / ".local" / "bin"
    if local_bin.exists():
        symlink = local_bin / "llm-agent-system"
        try:
            if symlink.exists():
                symlink.unlink()
            symlink.symlink_to(launcher)
            print_ok(f"Created symlink: {symlink}")
        except Exception as e:
            print_warn(f"Could not create symlink: {e}")


def main():
    print_header()

    # Check Python version
    check_python_version()

    # Detect environment
    os_type, arch, is_wsl = detect_os()
    if os_type != 'linux':
        print_error(f"Unsupported OS: {os_type}. LLM-Agent-System supports Linux only.")
        sys.exit(1)
    print_ok(f"OS: {os_type} ({arch})" + (" [WSL]" if is_wsl else ""))

    gpu_type, cmake_args = detect_gpu()
    vram_mb = None
    if gpu_type:
        print_ok(f"GPU: {gpu_type.upper()}")
        if gpu_type == 'nvidia':
            vram_mb = get_gpu_vram_mb()
            if vram_mb:
                layers = recommend_gpu_layers(vram_mb)
                print_ok(f"VRAM: {vram_mb} MB (recommended GPU layers: {layers})")
    else:
        print_info("GPU: None detected (will use CPU)")

    project_root = get_project_root()
    print_ok(f"Project root: {project_root}")

    # Create virtual environment
    print(f"\n{Colors.BOLD}[1/6] Virtual Environment{Colors.END}")
    venv_path = create_venv(project_root)

    # Install requirements
    print(f"\n{Colors.BOLD}[2/6] Python Dependencies{Colors.END}")
    install_requirements(venv_path, project_root)

    # Install llama-server binary
    print(f"\n{Colors.BOLD}[3/6] llama-server Binary{Colors.END}")
    if gpu_type == 'nvidia':
        print_info("NVIDIA GPU detected. Options:")
        print(f"  1. Download pre-built (Vulkan GPU support, quick ~30s)")
        print(f"  2. Build from source with CUDA (best performance, ~10 min)")
        print(f"  3. Skip (use existing llama-server)")
        try:
            choice = input("Choice [1/2/3] (default: 1): ").strip()
        except EOFError:
            choice = "1"
        if choice == "2":
            build_llama_server_cuda(project_root)
        elif choice == "3":
            print_info("Skipped. You'll need llama-server in PATH or ~/llama.cpp/build/bin/")
        else:
            install_llama_server(project_root)
    elif ask_yes_no("Download pre-built llama-server (recommended)?"):
        install_llama_server(project_root)
    else:
        print_info("Skipped llama-server download")
        print_info("You'll need llama-server in PATH or ~/llama.cpp/build/bin/")

    # Install LanceDB
    print(f"\n{Colors.BOLD}[4/6] LanceDB Vector Store{Colors.END}")
    install_lancedb(venv_path)

    # Download models
    print(f"\n{Colors.BOLD}[5/6] Model Downloads{Colors.END}")
    model_path = None
    # Drop downloads into <project>/models/ — convention for self-contained
    # installs. The directory is gitignored.
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # LLM — NVIDIA Orchestrator 8B Q5_K_M (~5.5 GB) is the recommended default.
    # Tuned for delegation / routing patterns, works well across all reference
    # agents (oracle, security, operator, coder, triage). Light enough for 8 GB+ VRAM.
    llm_file = models_dir / "nvidia_Orchestrator-8B-Q5_K_M.gguf"
    if llm_file.exists() and llm_file.stat().st_size > 1_000_000:
        print_ok(f"LLM already downloaded: {llm_file}")
        model_path = str(llm_file)
    elif ask_yes_no("Download recommended LLM (NVIDIA Orchestrator 8B Q5_K_M, ~5.5 GB)?", default=True):
        if download_gguf_via_hf("bartowski/nvidia_Orchestrator-8B-GGUF",
                                "nvidia_Orchestrator-8B-Q5_K_M.gguf",
                                models_dir):
            model_path = str(llm_file)
    else:
        print_info("Skipped LLM download — edit config/models.yaml later to point at your model")

    # Embedding model — Nomic v1.5 Q8_0 (~275 MB) for fully offline Knowledge / Summarizer use.
    # If skipped, the system falls back to the HuggingFace sentence-transformers
    # default (Qwen3-Embedding-0.6B, ~1.2 GB auto-fetched on first agent call).
    embed_file = models_dir / "nomic-embed-text-v1.5.Q8_0.gguf"
    if embed_file.exists() and embed_file.stat().st_size > 100_000:
        print_ok(f"Embedding model already downloaded: {embed_file}")
        _wire_embedding_to_local_gguf(project_root, embed_file)
    elif ask_yes_no("Download embedding model (Nomic v1.5 Q8_0, ~275 MB) for fully-local Knowledge/Summarizer?", default=True):
        if download_gguf_via_hf("nomic-ai/nomic-embed-text-v1.5-GGUF",
                                "nomic-embed-text-v1.5.Q8_0.gguf",
                                models_dir):
            _wire_embedding_to_local_gguf(project_root, embed_file)
    else:
        print_info("Skipped embedding GGUF — first Knowledge/Summarizer call will fetch the HF default (~1.2 GB)")

    # Auto-configure models.yaml with detected hardware
    print(f"\n{Colors.BOLD}[6/6] Configure Models{Colors.END}")
    configure_models_yaml(project_root, gpu_type, model_path)

    # Create launcher
    create_launcher(project_root, venv_path)

    # Done!
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.GREEN}  Installation Complete!{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"""
{Colors.CYAN}Quick Start:{Colors.END}
  ./llm-agent-system           - Start console

{Colors.CYAN}Configuration:{Colors.END}
  config/models.yaml     - Model paths
  config/rag_config.yaml - RAG settings

{Colors.GREEN}Installation complete. Run ./llm-agent-system to get started.{Colors.END}
""")


if __name__ == "__main__":
    main()
