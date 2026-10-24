#!/usr/bin/env bash
# ==============================================================================
# OmniIntel Environment Setup Script
# Configures Python virtual environment, checks system requirements, and sets up dependencies.
# ==============================================================================

set -euo pipefail

# ANSI color formatting
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}        OmniIntel: System Environment Setup        ${NC}"
echo -e "${BLUE}====================================================${NC}"

# Step 1: Detect Operating System
OS_TYPE="$(uname -s)"
echo -e "${GREEN}[1/4]${NC} Detected OS: ${YELLOW}${OS_TYPE}${NC}"

# Step 2: Validate Python 3 Installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] python3 is not installed or not on PATH.${NC}"
    exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo -e "${GREEN}[2/4]${NC} Detected Python version: ${YELLOW}${PYTHON_VERSION}${NC}"

# Check minimum required Python version (>= 3.10)
python3 -c '
import sys
if sys.version_info < (3, 10):
    print("Python 3.10 or higher is required. Found: " + sys.version)
    sys.exit(1)
'

# Step 3: Configure Virtual Environment (.venv)
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${GREEN}[3/4]${NC} Creating virtual environment in ${YELLOW}${VENV_DIR}${NC}..."
    python3 -m venv "$VENV_DIR"
else
    echo -e "${GREEN}[3/4]${NC} Virtual environment already exists at ${YELLOW}${VENV_DIR}${NC}."
fi

# Step 4: Environment Variables Template
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo -e "${GREEN}[4/4]${NC} Generating ${YELLOW}.env${NC} from template..."
        cp .env.example .env
    fi
else
    echo -e "${GREEN}[4/4]${NC} ${YELLOW}.env${NC} configuration file already exists."
fi

echo -e "\n${GREEN}====================================================${NC}"
echo -e "${GREEN}✓ Environment setup complete!${NC}"
echo -e "To activate the virtual environment, run:"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo -e "To run all automated tests, run:"
echo -e "  ${YELLOW}make test${NC}"
echo -e "${GREEN}====================================================${NC}"
