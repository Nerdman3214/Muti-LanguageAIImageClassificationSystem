cmake_minimum_required(VERSION 3.10)
project(InferenceEngine LANGUAGES CXX)

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
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
endif()

# Find OpenCV for image loading
find_package(OpenCV QUIET)

# Find CUDA (optional for GPU acceleration)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

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

# Link CUDA libraries if available
if(CUDA_FOUND)
    target_link_libraries(inference_engine ${CUDA_LIBRARIES})
    target_link_libraries(InferenceEngine ${CUDA_LIBRARIES})
    message(STATUS "CUDA libraries linked")
endif()
