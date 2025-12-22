#!/bin/bash
# Master build script - builds all components

set -e

echo "======================================"
echo "Building Multi-Language AI System"
echo "======================================"

# Build Python environment
echo -e "\n[1/3] Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
echo "Python setup completed!"

# Build C++
echo -e "\n[2/3] Building C++ components..."
bash scripts/build_cpp.sh

# Build Java
echo -e "\n[3/3] Building Java components..."
bash scripts/build_java.sh

echo -e "\n======================================"
echo "All builds completed successfully!"
echo "======================================"
