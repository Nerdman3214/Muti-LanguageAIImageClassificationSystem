# ==============================================================================
# CMakeLists.txt - Build Configuration for the C++ Inference Engine
# ==============================================================================
#
# CMAKE OVERVIEW:
# ---------------
# CMake is a cross-platform build system generator. It doesn't compile code
# directly - instead, it generates build files for other systems:
# - Linux: Generates Makefiles (then you run 'make')
# - Windows: Generates Visual Studio projects
# - macOS: Generates Xcode projects or Makefiles
#
# WHY CMAKE?
# - Write one CMakeLists.txt, build on any platform
# - Automatic dependency detection (finds libraries, headers, etc.)
# - Modern C++ feature support (standards, sanitizers, etc.)
# - Integration with IDEs (VS Code, CLion, Visual Studio)
#
# BUILD WORKFLOW:
# 1. mkdir build && cd build     # Create build directory
# 2. cmake ..                     # Generate build files
# 3. make                         # Compile everything
#
# CMAKE SYNTAX BASICS:
# - Commands are FUNCTION_NAME(arguments)
# - Variables are ${VARIABLE_NAME}
# - Comments start with #
# - Commands are case-insensitive, but convention is lowercase
# ==============================================================================

# ==============================================================================
# CMake Version Requirement
# ==============================================================================
# cmake_minimum_required tells CMake what features we need.
# Version 3.10+ provides:
# - Modern target_* commands
# - Better find_package support
# - Improved C++ standard handling
cmake_minimum_required(VERSION 3.10)

# ==============================================================================
# Project Declaration
# ==============================================================================
# project() sets project name and languages used.
# LANGUAGES CXX tells CMake we're building C++ code.
# This affects compiler detection and default file extensions.
project(InferenceEngine LANGUAGES CXX)

# ==============================================================================
# C++ Standard Configuration
# ==============================================================================
# CMAKE_CXX_STANDARD specifies which C++ version to use.
# C++17 provides:
# - std::optional, std::variant
# - Structured bindings (auto [x, y] = pair)
# - if constexpr
# - Fold expressions
#
# CMAKE_CXX_STANDARD_REQUIRED=ON means fail if C++17 isn't available.
# Without it, CMake would silently fall back to an older standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# Exclude Problematic Directories
# ==============================================================================
# Python virtual environments can confuse the compiler when searching for headers.
# This line prevents Python venv directories from being accidentally included.
# (This was added to fix a specific issue with header conflicts)
set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES 
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})

# ==============================================================================
# Find JNI (Java Native Interface)
# ==============================================================================
# JNI allows our C++ code to be called from Java.
#
# find_package searches for installed libraries/frameworks:
# - Looks in standard system paths
# - Can use environment variables (JAVA_HOME)
# - Returns success/failure and sets variables
#
# REQUIRED means CMake will stop with an error if JNI isn't found.
# Without it, we'd have to check JNI_FOUND ourselves.
#
# JNI_INCLUDE_DIRS will contain paths like:
#   /usr/lib/jvm/java-21-openjdk/include
#   /usr/lib/jvm/java-21-openjdk/include/linux
find_package(JNI REQUIRED)

if(JNI_FOUND)
    # include_directories() adds paths to the compiler's header search path.
    # This allows #include <jni.h> to work.
    include_directories(${JNI_INCLUDE_DIRS})
    
    # message() prints during CMake configuration.
    # STATUS = informational, WARNING = warning, FATAL_ERROR = stop
    message(STATUS "JNI Include Dirs: ${JNI_INCLUDE_DIRS}")
else()
    # FATAL_ERROR stops CMake with an error message
    message(FATAL_ERROR "JNI not found - cannot build JNI bridge")
endif()

# ==============================================================================
# Find ONNX Runtime
# ==============================================================================
# ONNX Runtime is our neural network inference engine.
#
# It can be installed:
# 1. System-wide (find_package will find it)
# 2. Downloaded manually (set ONNXRUNTIME_ROOT environment variable)
#
# ENV{VAR} reads an environment variable.
# DEFINED checks if a variable exists.
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()

# ==============================================================================
# Find OpenCV
# ==============================================================================
# OpenCV is used for image loading and preprocessing.
#
# QUIET suppresses "not found" messages - we handle that ourselves.
# OpenCV is optional; we have a fallback loader if it's not available.
find_package(OpenCV QUIET)

# ==============================================================================
# Find CUDA (Optional - for GPU Acceleration)
# ==============================================================================
# CUDA enables GPU-accelerated inference.
# find_package(CUDAToolkit) is the modern CMake 3.17+ way.
#
# CUDAToolkit_FOUND will be true if CUDA is installed.
# CUDAToolkit_VERSION contains the version string (e.g., "11.8").
# CUDAToolkit_INCLUDE_DIRS contains header paths.
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# ==============================================================================
# Include Directories
# ==============================================================================
# Add our own include directory to the compiler's search path.
# CMAKE_CURRENT_SOURCE_DIR is the directory containing this CMakeLists.txt.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ==============================================================================
# Source Files
# ==============================================================================
# set() creates a variable. Here we list all our C++ source files.
# Using a variable makes it easier to reuse the list in multiple places.
set(SOURCES
    src/InferenceEngine.cpp   # Main inference engine implementation
    src/Softmax.cpp           # Softmax and top-K utilities
    src/ImageUtils.cpp        # Image loading and preprocessing
)

# JNI bridge source file (in sibling directory)
set(JNI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../jni/InferenceJNI.cpp
)

# ==============================================================================
# Shared Library Target
# ==============================================================================
# add_library creates a library target.
#
# SHARED means create a shared library (.so on Linux, .dll on Windows).
# Shared libraries are loaded at runtime, not linked into the executable.
# Java's System.loadLibrary() loads shared libraries.
#
# Target name: inference_engine
# Output: libinference_engine.so (on Linux)
add_library(inference_engine SHARED ${SOURCES} ${JNI_SOURCES})

# set_target_properties customizes the output file name.
# OUTPUT_NAME removes any "lib" prefix on some platforms.
set_target_properties(inference_engine PROPERTIES OUTPUT_NAME inference_engine)

# ==============================================================================
# CLI Executable Target
# ==============================================================================
# add_executable creates an executable target.
# This is the command-line tool for testing inference without Java.
add_executable(InferenceEngine src/main.cpp)

# target_link_libraries links libraries to targets.
# PRIVATE means the dependency is only for this target, not propagated.
# The executable links against our shared library.
target_link_libraries(InferenceEngine PRIVATE inference_engine)

# ==============================================================================
# JNI Linking
# ==============================================================================
# The shared library needs JNI headers and might need JNI libraries.
target_include_directories(inference_engine PRIVATE ${JNI_INCLUDE_DIRS})
target_link_libraries(inference_engine PRIVATE ${JNI_LIBRARIES})

# ==============================================================================
# ONNX Runtime Linking (Multiple Methods)
# ==============================================================================
# We try multiple ways to find and link ONNX Runtime:
#
# Method 1: find_package with CONFIG
#   Looks for ONNXRuntimeConfig.cmake (provided by some installations)
find_package(onnxruntime QUIET CONFIG)

if(onnxruntime_FOUND)
    # ONNX Runtime found via package manager
    # Check which target name is available (varies by installation method)
    if(TARGET onnxruntime::onnxruntime)
        target_link_libraries(inference_engine PRIVATE onnxruntime::onnxruntime)
    else()
        target_link_libraries(inference_engine PRIVATE onnxruntime)
    endif()
    
    # target_compile_definitions adds preprocessor definitions.
    # HAS_ONNXRUNTIME enables ONNX-specific code paths via #ifdef.
    target_compile_definitions(inference_engine PRIVATE HAS_ONNXRUNTIME)
    message(STATUS "ONNX Runtime linked via package")

# Method 2: Manual installation via ONNXRUNTIME_ROOT
elseif(ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    # Manually specify include and library directories
    target_include_directories(inference_engine PRIVATE ${ONNXRUNTIME_ROOT}/include)
    target_link_directories(inference_engine PRIVATE ${ONNXRUNTIME_ROOT}/lib)
    target_link_libraries(inference_engine PRIVATE onnxruntime)
    target_compile_definitions(inference_engine PRIVATE HAS_ONNXRUNTIME)
    message(STATUS "ONNX Runtime linked via ONNXRUNTIME_ROOT")

# Method 3: Not found - build without ONNX Runtime
else()
    # WARNING doesn't stop CMake, just alerts the user
    message(WARNING "ONNX Runtime not found; building without HAS_ONNXRUNTIME")
endif()

# ==============================================================================
# OpenCV Linking (Optional)
# ==============================================================================
if(OpenCV_FOUND)
    # Link OpenCV to both the shared library and executable
    # ${OpenCV_LIBS} contains all necessary OpenCV libraries
    target_link_libraries(inference_engine PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(inference_engine PRIVATE HAS_OPENCV)
    
    target_link_libraries(InferenceEngine PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(InferenceEngine PRIVATE HAS_OPENCV)
    
    message(STATUS "OpenCV linked")
endif()

# ==============================================================================
# CUDA Linking (Optional)
# ==============================================================================
if(CUDAToolkit_FOUND)
    # Link CUDA runtime library for GPU support
    # CUDA::cudart is a modern CMake imported target
    target_link_libraries(inference_engine PRIVATE CUDA::cudart)
    target_link_libraries(InferenceEngine PRIVATE CUDA::cudart)
    message(STATUS "CUDA libraries linked")
endif()

# ==============================================================================
# BUILD SUMMARY
# ==============================================================================
# After running cmake, you'll see messages indicating:
# - JNI found and where
# - ONNX Runtime linked (or not)
# - OpenCV linked (or not)
# - CUDA linked (or not)
#
# The resulting build products:
# 1. libinference_engine.so - Shared library for Java JNI
# 2. InferenceEngine - Command-line executable for testing
# ==============================================================================
