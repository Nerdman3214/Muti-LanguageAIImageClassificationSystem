#!/bin/bash
# Build script for Java components

set -e

echo "Building Java components..."

# Check if Maven is installed
if ! command -v mvn &> /dev/null; then
    echo "Maven not found. Please install Maven first."
    exit 1
fi

# Build Java project
cd java
mvn clean package

echo "Java build completed successfully!"
echo "Output: java/target/MultiLanguageAIImageSystem-1.0.0.jar"
