cmake_minimum_required(VERSION 3.10)
project(InferenceEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/InferenceEngine.cpp
)

# Create shared library
add_library(inference_engine SHARED ${SOURCES})

# Create executable
add_executable(InferenceEngine src/main.cpp ${SOURCES})

# Optional: Add tests
enable_testing()
add_subdirectory(tests)
