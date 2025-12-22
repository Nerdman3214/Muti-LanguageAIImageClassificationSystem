#!/bin/bash
# Build script for C++ components

set -e

echo "Building C++ Inference Engine..."

# Create build directory if it doesn't exist
mkdir -p cpp/build

# Navigate to build directory
cd cpp/build

# Run CMake and build
cmake ..
make

echo "C++ build completed successfully!"
echo "Output: cpp/build/InferenceEngine"
