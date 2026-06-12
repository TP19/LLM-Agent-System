#!/bin/bash
# LLM-Agent-System - Auto Install Script (Linux only)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   LLM-Agent-System Release - Installer${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Detect script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-$HOME/llm-agent-system}"

echo -e "${YELLOW}Source:${NC} $SCRIPT_DIR"
echo -e "${YELLOW}Install to:${NC} $INSTALL_DIR"
echo ""

# Check Python version
echo -e "${BLUE}[1/5]${NC} Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Python $PYTHON_VERSION"

# Create install directory
echo -e "${BLUE}[2/5]${NC} Creating installation directory..."
if [ -d "$INSTALL_DIR" ]; then
    echo -e "  ${YELLOW}Directory exists, updating...${NC}"
else
    mkdir -p "$INSTALL_DIR"
    echo -e "  ${GREEN}✓${NC} Created $INSTALL_DIR"
fi

# Copy files
echo -e "${BLUE}[3/5]${NC} Copying files..."
if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
    cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
    echo -e "  ${GREEN}✓${NC} Files copied"
else
    echo -e "  ${YELLOW}Already in install directory${NC}"
fi

# Create virtual environment
echo -e "${BLUE}[4/5]${NC} Setting up virtual environment..."
cd "$INSTALL_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "  ${GREEN}✓${NC} Virtual environment created"
else
    echo -e "  ${YELLOW}Virtual environment exists${NC}"
fi

# Activate and install dependencies
source .venv/bin/activate

echo -e "${BLUE}[5/5]${NC} Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q 2>/dev/null || {
    echo -e "  ${YELLOW}Some optional dependencies may be missing${NC}"
}
echo -e "  ${GREEN}✓${NC} Dependencies installed"

# Create launcher script
echo -e "${BLUE}Creating launcher script...${NC}"
cat > "$INSTALL_DIR/llm-agent-system" << 'LAUNCHER'
#!/bin/bash
# LLM-Agent-System Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate
python start_console.py "$@"
LAUNCHER
chmod +x "$INSTALL_DIR/llm-agent-system"

# Create symlink in ~/.local/bin if it exists
if [ -d "$HOME/.local/bin" ]; then
    ln -sf "$INSTALL_DIR/llm-agent-system" "$HOME/.local/bin/llm-agent-system" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Symlink created in ~/.local/bin"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "To start LLM-Agent-System:"
echo -e "  ${BLUE}cd $INSTALL_DIR && ./llm-agent-system${NC}"
echo ""
echo -e "Or if ~/.local/bin is in PATH:"
echo -e "  ${BLUE}llm-agent-system${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Edit ${YELLOW}config/models.yaml${NC} with your model paths"
echo ""
