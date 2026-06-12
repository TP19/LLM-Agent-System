#!/bin/bash
#
# LLM-Agent-System Installation Script
# Complete setup for new systems
#
# Usage: ./install.sh [options]
#
# Options:
#   --skip-python    Skip Python environment setup
#   --skip-user      Skip llm-agent user creation
#   --skip-deps      Skip system dependencies
#   --all            Install everything (default)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default options
INSTALL_PYTHON=true
INSTALL_USER=true
INSTALL_DEPS=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-python) INSTALL_PYTHON=false ;;
        --skip-user) INSTALL_USER=false ;;
        --skip-deps) INSTALL_DEPS=false ;;
        --all) INSTALL_PYTHON=true; INSTALL_USER=true; INSTALL_DEPS=true ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-python    Skip Python environment setup"
            echo "  --skip-user      Skip llm-agent user creation (requires sudo)"
            echo "  --skip-deps      Skip system dependencies (requires sudo)"
            echo "  --all            Install everything (default)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           LLM-Agent-System Installation Script                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Detect OS
if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
else
    OS="unknown"
fi
echo -e "${YELLOW}Detected OS: $OS $OS_VERSION${NC}\n"

#
# Step 1: System Dependencies
#
if [[ "$INSTALL_DEPS" == true ]]; then
    echo -e "${GREEN}[1/4] Installing System Dependencies${NC}"
    echo "────────────────────────────────────────"

    if [[ $EUID -ne 0 ]]; then
        echo -e "${YELLOW}Note: System dependencies require sudo${NC}"
        SUDO="sudo"
    else
        SUDO=""
    fi

    case $OS in
        ubuntu|debian)
            $SUDO apt-get update
            $SUDO apt-get install -y \
                python3 python3-pip python3-venv \
                git curl wget \
                build-essential \
                libpq-dev \
                jq
            ;;
        fedora|rhel|centos)
            $SUDO dnf install -y \
                python3 python3-pip \
                git curl wget \
                gcc gcc-c++ make \
                postgresql-devel \
                jq
            ;;
        arch)
            $SUDO pacman -Sy --noconfirm \
                python python-pip \
                git curl wget \
                base-devel \
                postgresql-libs \
                jq
            ;;
        *)
            echo -e "${YELLOW}Unknown OS. Please install dependencies manually:${NC}"
            echo "  - Python 3.10+"
            echo "  - pip, venv"
            echo "  - git, curl, wget, jq"
            ;;
    esac
    echo -e "${GREEN}✓ System dependencies installed${NC}\n"
else
    echo -e "${YELLOW}[1/4] Skipping system dependencies${NC}\n"
fi

#
# Step 2: Python Environment
#
if [[ "$INSTALL_PYTHON" == true ]]; then
    echo -e "${GREEN}[2/4] Setting Up Python Environment${NC}"
    echo "────────────────────────────────────────"

    VENV_DIR="${HOME}/envs/llm-agent-system"

    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating virtual environment at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    fi

    echo "Activating environment..."
    source "$VENV_DIR/bin/activate"

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing requirements..."
    if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
        pip install -r "$PROJECT_DIR/requirements.txt"
    else
        # Core dependencies if no requirements.txt
        pip install \
            openai anthropic \
            lancedb \
            rich typer \
            pyyaml \
            httpx aiohttp \
            pwntools
    fi

    echo -e "${GREEN}✓ Python environment ready at $VENV_DIR${NC}\n"
else
    echo -e "${YELLOW}[2/4] Skipping Python environment${NC}\n"
fi

#
# Step 3: LLM-Agent User Setup
#
if [[ "$INSTALL_USER" == true ]]; then
    echo -e "${GREEN}[3/4] Setting Up LLM-Agent User${NC}"
    echo "────────────────────────────────────────"

    if [[ -x "$SCRIPT_DIR/setup_llm_agent.sh" ]]; then
        if [[ $EUID -eq 0 ]]; then
            "$SCRIPT_DIR/setup_llm_agent.sh"
        else
            echo -e "${YELLOW}Running with sudo...${NC}"
            sudo "$SCRIPT_DIR/setup_llm_agent.sh"
        fi
    else
        echo -e "${RED}Error: setup_llm_agent.sh not found${NC}"
        echo "Please run manually: sudo $SCRIPT_DIR/setup_llm_agent.sh"
    fi
    echo ""
else
    echo -e "${YELLOW}[3/4] Skipping llm-agent user setup${NC}\n"
fi

#
# Step 4: Configuration
#
echo -e "${GREEN}[4/4] Configuration${NC}"
echo "────────────────────────────────────────"

# Create config directory if needed
mkdir -p "$PROJECT_DIR/config"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data"

# Check for models.yaml
if [[ ! -f "$PROJECT_DIR/config/models.yaml" ]]; then
    echo -e "${YELLOW}Creating default models.yaml...${NC}"
    cat > "$PROJECT_DIR/config/models.yaml" << 'EOF'
# LLM-Agent-System Model Configuration
# Customize paths based on your system

default:
  backend: llama-server
  endpoint: http://localhost:8090/v1

models:
  oracle:
    name: nvidia-orchestrator-8b
    # path: /path/to/model.gguf  # Uncomment and set for local models

  operator:
    name: nvidia-orchestrator-8b

  security:
    name: nvidia-orchestrator-8b

  coder:
    name: nvidia-orchestrator-8b

  knowledge:
    name: nvidia-orchestrator-8b

  triage:
    name: nvidia-orchestrator-8b

  summarizer:
    name: nvidia-orchestrator-8b

embedding:
  backend: auto  # Uses llama-cpp-python or falls back to llama-server
EOF
    echo -e "${GREEN}✓ Created default config/models.yaml${NC}"
    echo -e "${YELLOW}  Please edit this file to set your model paths${NC}"
else
    echo -e "${GREEN}✓ config/models.yaml exists${NC}"
fi

#
# Summary
#
echo -e "\n${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                 Installation Complete!                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "Next steps:"
echo -e "  1. ${YELLOW}Edit config/models.yaml${NC} with your model paths"
echo -e "  2. ${YELLOW}Start llama-server${NC} with your model:"
echo -e "     llama-server -m /path/to/model.gguf --port 8090"
echo -e "  3. ${YELLOW}Activate the environment${NC}:"
echo -e "     source ~/envs/llm-agent-system/bin/activate"
echo -e "  4. ${YELLOW}Run LLM-Agent-System${NC}:"
echo -e "     python start_interactive.py"
echo ""
echo -e "Documentation: ${BLUE}docs/LLM_AGENT_USER_SETUP.md${NC}"
