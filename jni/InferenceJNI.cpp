#include <jni.h>
#include "InferenceEngine.hpp"
#include <memory>
#include <string>
#include <vector>

/**
 * JNI Bridge - Connects Java AIController to C++ InferenceEngine
 * 
 * These native methods are called from Java using the JNI mechanism.
 * The function names follow the convention: Java_<package>_<class>_<method>
 */

// Global engine instance (singleton pattern)
static std::unique_ptr<InferenceEngine> g_engine;

extern "C" {

/**
 * Initialize the inference engine
 * Returns: true if successful
 */
JNIEXPORT jboolean JNICALL
Java_ai_controller_NativeInference_initialize(
    JNIEnv* env,
    jobject /* this */,
    jstring modelPath
) {
    try {
        const char* pathChars = env->GetStringUTFChars(modelPath, nullptr);
        std::string path(pathChars);
        env->ReleaseStringUTFChars(modelPath, pathChars);

        g_engine = std::make_unique<InferenceEngine>();
        bool success = g_engine->initialize(path);

        return success ? JNI_TRUE : JNI_FALSE;

    } catch (const std::exception& e) {
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, e.what());
        return JNI_FALSE;
    }
}

/**
 * Run inference on an image
 * Returns: JSON string with results
 */
JNIEXPORT jstring JNICALL
Java_ai_controller_NativeInference_classifyImage(
    JNIEnv* env,
    jobject /* this */,
    jstring imagePath
) {
    try {
        if (!g_engine) {
            jclass exClass = env->FindClass("java/lang/IllegalStateException");
            env->ThrowNew(exClass, "Engine not initialized");
            return nullptr;
        }

        const char* pathChars = env->GetStringUTFChars(imagePath, nullptr);
        std::string path(pathChars);
        env->ReleaseStringUTFChars(imagePath, pathChars);

        // Run inference
        auto probabilities = g_engine->classifyImage(path);

        // Find top prediction
        auto maxIt = std::max_element(probabilities.begin(), probabilities.end());
        size_t classIdx = std::distance(probabilities.begin(), maxIt);
        float confidence = *maxIt;

        // Build JSON response
        std::string json = "{"
            "\"class\":" + std::to_string(classIdx) + ","
            "\"confidence\":" + std::to_string(confidence) + ","
            "\"engine\":\"ONNX\","
            "\"device\":\"GPU\""
            "}";

        return env->NewStringUTF(json.c_str());

    } catch (const std::exception& e) {
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, e.what());
        return nullptr;
    }
}

/**
 * Cleanup resources
 */
JNIEXPORT void JNICALL
Java_ai_controller_NativeInference_cleanup(
    JNIEnv* /* env */,
    jobject /* this */
) {
    g_engine.reset();
}

/**
 * Get engine version
 */
JNIEXPORT jstring JNICALL
Java_ai_controller_NativeInference_getVersion(
    JNIEnv* env,
    jobject /* this */
) {
    if (g_engine) {
        return env->NewStringUTF(g_engine->getVersion().c_str());
    }
    return env->NewStringUTF("not initialized");
}

} // extern "C"
