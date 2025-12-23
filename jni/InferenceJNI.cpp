#include <jni.h>
#include "../cpp/include/InferenceEngine.hpp"
#include <iostream>

/**
 * JNI Bridge - Connects Java AIController to C++ InferenceEngine
 * 
 * These native methods are called from Java using the JNI mechanism.
 * The function names follow the convention: Java_<package>_<class>_<method>
 */

// Global inference engine instance
static InferenceEngine* g_engine = nullptr;
static bool g_initialized = false;

/**
 * Initialize the inference engine with a model
 */
bool ensureInitialized(const char* modelPath) {
    if (!g_initialized) {
        g_engine = new InferenceEngine();
        g_initialized = g_engine->initialize(modelPath);
        if (!g_initialized) {
            delete g_engine;
            g_engine = nullptr;
        }
    }
    return g_initialized;
}

/**
 * JNI: Predict class from logits array
 * Signature: (F[])I
 */
extern "C" JNIEXPORT jint JNICALL 
Java_ai_controller_AIController_predict(JNIEnv* env, jobject /*obj*/, jfloatArray arr) {
    if (arr == nullptr) return -1;

    jsize len = env->GetArrayLength(arr);
    if (len == 0) return -1;

    std::vector<float> logits(static_cast<size_t>(len));
    env->GetFloatArrayRegion(arr, 0, len, reinterpret_cast<jfloat*>(logits.data()));

    // Ensure engine is initialized
    if (!g_engine) {
        g_engine = new InferenceEngine();
    }
    
    int prediction = g_engine->predict(logits);
    return static_cast<jint>(prediction);
}

/**
 * JNI: Classify an image file
 * Signature: (Ljava/lang/String;Ljava/lang/String;)[F
 * 
 * @param imagePath Path to the image to classify
 * @param modelPath Path to the ONNX model file
 * @return Array of class probabilities (length 1000 for ImageNet)
 */
extern "C" JNIEXPORT jfloatArray JNICALL 
Java_ai_controller_AIController_classifyImage(JNIEnv* env, jobject /*obj*/, 
                                               jstring jImagePath, jstring jModelPath) {
    // Get C strings from Java strings
    const char* imagePath = env->GetStringUTFChars(jImagePath, nullptr);
    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    
    if (imagePath == nullptr || modelPath == nullptr) {
        if (imagePath) env->ReleaseStringUTFChars(jImagePath, imagePath);
        if (modelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
        return nullptr;
    }
    
    jfloatArray result = nullptr;
    
    try {
        // Initialize engine if needed
        if (!ensureInitialized(modelPath)) {
            std::cerr << "Failed to initialize inference engine" << std::endl;
            env->ReleaseStringUTFChars(jImagePath, imagePath);
            env->ReleaseStringUTFChars(jModelPath, modelPath);
            return nullptr;
        }
        
        // Run classification
        std::vector<float> probs = g_engine->classifyImage(imagePath);
        
        // Create Java float array for result
        result = env->NewFloatArray(probs.size());
        if (result != nullptr) {
            env->SetFloatArrayRegion(result, 0, probs.size(), probs.data());
        }
        
    } catch (const std::exception& e) {
        std::cerr << "JNI Error: " << e.what() << std::endl;
    }
    
    // Release strings
    env->ReleaseStringUTFChars(jImagePath, imagePath);
    env->ReleaseStringUTFChars(jModelPath, modelPath);
    
    return result;
}

/**
 * JNI: Get number of classes supported by the model
 * Signature: ()I
 */
extern "C" JNIEXPORT jint JNICALL 
Java_ai_controller_AIController_getNumClasses(JNIEnv* /*env*/, jobject /*obj*/) {
    if (g_engine && g_initialized) {
        return static_cast<jint>(g_engine->getNumClasses());
    }
    return 1000;  // ImageNet default
}

/**
 * JNI: Get engine version string
 * Signature: ()Ljava/lang/String;
 */
extern "C" JNIEXPORT jstring JNICALL 
Java_ai_controller_AIController_getVersion(JNIEnv* env, jobject /*obj*/) {
    if (g_engine && g_initialized) {
        return env->NewStringUTF(g_engine->getVersion().c_str());
    }
    return env->NewStringUTF("1.0.0");
}

/**
 * JNI: Cleanup native resources
 * Signature: ()V
 */
extern "C" JNIEXPORT void JNICALL 
Java_ai_controller_AIController_cleanup(JNIEnv* /*env*/, jobject /*obj*/) {
    if (g_engine) {
        delete g_engine;
        g_engine = nullptr;
        g_initialized = false;
    }
}
