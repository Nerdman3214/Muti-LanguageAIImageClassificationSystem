cmake_minimum_required(VERSION 3.10)
project(InferenceEngine LANGUAGES CXX)

# ============================================
# COMPILER SETTINGS
# ============================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

# Enable debug symbols
set(CMAKE_BUILD_TYPE Debug)

# ============================================
# ONNX RUNTIME (Optional)
# ============================================
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()

if(ONNXRUNTIME_ROOT)
    include_directories(${ONNXRUNTIME_ROOT}/include)
    link_directories(${ONNXRUNTIME_ROOT}/lib)
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
endif()

# ============================================
# OPENCV (Optional)
# ============================================
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
endif()

# ============================================
# INCLUDE DIRECTORIES
# ============================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ============================================
# SOURCE FILES
# ============================================
set(SOURCES
    src/InferenceEngine.cpp
    src/Softmax.cpp
    src/ImageUtils.cpp
)

# ============================================
# BUILD TARGETS
# ============================================

# Shared library for JNI
add_library(inference_engine SHARED ${SOURCES})

# Executable for standalone testing
add_executable(InferenceEngine src/main.cpp ${SOURCES})

# ============================================
# LINKING
# ============================================

# Link ONNX Runtime if available
if(EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    target_link_libraries(inference_engine onnxruntime)
    target_link_libraries(InferenceEngine onnxruntime)
    target_compile_definitions(inference_engine PRIVATE HAS_ONNXRUNTIME)
    target_compile_definitions(InferenceEngine PRIVATE HAS_ONNXRUNTIME)
    message(STATUS "ONNX Runtime linked")
endif()

# Link OpenCV if available
if(OpenCV_FOUND)
    target_link_libraries(inference_engine ${OpenCV_LIBS})
    target_link_libraries(InferenceEngine ${OpenCV_LIBS})
    target_compile_definitions(inference_engine PRIVATE HAS_OPENCV)
    target_compile_definitions(InferenceEngine PRIVATE HAS_OPENCV)
    message(STATUS "OpenCV linked")
endif()

# ============================================
# TESTS (Optional)
# ============================================
enable_testing()

# Only add tests if CMakeLists.txt exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
    message(STATUS "Tests enabled")
else()
    message(STATUS "No tests found (tests/CMakeLists.txt missing)")
endif()

# ============================================
# INSTALLATION
# ============================================
install(TARGETS InferenceEngine DESTINATION bin)
install(TARGETS inference_engine DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

message(STATUS "")
message(STATUS "========================================")
message(STATUS "  Multi-Language AI Image Classifier")
message(STATUS "========================================")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "========================================")