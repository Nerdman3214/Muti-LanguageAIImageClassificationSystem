cmake_minimum_required(VERSION 3.10)
project(InferenceEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly exclude Python venv from any include search
set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES 
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})

# ============================================
# CRITICAL FIX: Find JNI (Java Native Interface)
# ============================================
find_package(JNI REQUIRED)
if(JNI_FOUND)
    include_directories(${JNI_INCLUDE_DIRS})
    message(STATUS "JNI Include Dirs: ${JNI_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "JNI not found - cannot build JNI bridge")
endif()

# Find ONNX Runtime (if installed system-wide)
# For manual installation, set ONNXRUNTIME_ROOT
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()

# Find OpenCV for image loading
find_package(OpenCV QUIET)

# Find CUDA (optional for GPU acceleration)
find_package(CUDAToolkit QUIET)  # Modern CMake 3.17+ way
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/InferenceEngine.cpp
    src/Softmax.cpp
    src/ImageUtils.cpp
)

set(JNI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../jni/InferenceJNI.cpp
)

# Create shared library (used by Java JNI loader)
add_library(inference_engine SHARED ${SOURCES} ${JNI_SOURCES})
set_target_properties(inference_engine PROPERTIES OUTPUT_NAME inference_engine)

# Create CLI executable linked against the shared library
add_executable(InferenceEngine src/main.cpp)
target_link_libraries(InferenceEngine PRIVATE inference_engine)

target_include_directories(inference_engine PRIVATE ${JNI_INCLUDE_DIRS})
target_link_libraries(inference_engine PRIVATE ${JNI_LIBRARIES})

# Link ONNX Runtime if available
find_package(onnxruntime QUIET CONFIG)
if(onnxruntime_FOUND)
    if(TARGET onnxruntime::onnxruntime)
        target_link_libraries(inference_engine PRIVATE onnxruntime::onnxruntime)
    else()
        target_link_libraries(inference_engine PRIVATE onnxruntime)
    endif()
    target_compile_definitions(inference_engine PRIVATE HAS_ONNXRUNTIME)
    message(STATUS "ONNX Runtime linked via package")
elseif(ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    target_include_directories(inference_engine PRIVATE ${ONNXRUNTIME_ROOT}/include)
    target_link_directories(inference_engine PRIVATE ${ONNXRUNTIME_ROOT}/lib)
    target_link_libraries(inference_engine PRIVATE onnxruntime)
    target_compile_definitions(inference_engine PRIVATE HAS_ONNXRUNTIME)
    message(STATUS "ONNX Runtime linked via ONNXRUNTIME_ROOT")
else()
    message(WARNING "ONNX Runtime not found; building without HAS_ONNXRUNTIME")
endif()

# Link OpenCV if available
if(OpenCV_FOUND)
    target_link_libraries(inference_engine PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(inference_engine PRIVATE HAS_OPENCV)
    target_link_libraries(InferenceEngine PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(InferenceEngine PRIVATE HAS_OPENCV)
    message(STATUS "OpenCV linked")
endif()

# Link CUDA libraries if available
if(CUDAToolkit_FOUND)
    target_link_libraries(inference_engine PRIVATE CUDA::cudart)
    target_link_libraries(InferenceEngine PRIVATE CUDA::cudart)
    message(STATUS "CUDA libraries linked")
endif()
