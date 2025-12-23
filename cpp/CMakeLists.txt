cmake_minimum_required(VERSION 3.10)
project(InferenceEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime (if installed system-wide)
# For manual installation, set ONNXRUNTIME_ROOT
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()

if(ONNXRUNTIME_ROOT)
    include_directories(${ONNXRUNTIME_ROOT}/include)
    link_directories(${ONNXRUNTIME_ROOT}/lib)
endif()

# Find OpenCV for image loading
find_package(OpenCV QUIET)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/InferenceEngine.cpp
    src/Softmax.cpp
    src/ImageUtils.cpp
)

# Create shared library
add_library(inference_engine SHARED ${SOURCES})

# Create executable
add_executable(InferenceEngine src/main.cpp ${SOURCES})

# Link ONNX Runtime if available
if(EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    target_link_libraries(inference_engine onnxruntime)
    target_link_libraries(InferenceEngine onnxruntime)
    add_definitions(-DHAS_ONNXRUNTIME)
endif()

# Link OpenCV if available
if(OpenCV_FOUND)
    target_link_libraries(inference_engine ${OpenCV_LIBS})
    target_link_libraries(InferenceEngine ${OpenCV_LIBS})
    add_definitions(-DHAS_OPENCV)
endif()

# Optional: Add tests
enable_testing()
add_subdirectory(tests)
