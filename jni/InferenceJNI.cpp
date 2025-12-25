#include <jni.h>

// Include standard C++ headers BEFORE anything that might include system headers
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Undefine any problematic macros
#ifdef initialize
#undef initialize
#endif
#ifdef classifyImage  
#undef classifyImage
#endif
#ifdef cleanup
#undef cleanup
#endif

// Now include our header
#include "InferenceEngine.hpp"

// Singleton engine reused across JNI calls to avoid reload overhead
static std::unique_ptr<InferenceEngine> g_engine;

// Helper function to throw Java exceptions from C++
static void throwRuntime(JNIEnv* env, const std::string& message) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass != nullptr) {
        env->ThrowNew(exClass, message.c_str());
    }
}

namespace {

InferenceEngine& ensureEngine(JNIEnv* env) {
    if (!g_engine) {
        g_engine = std::make_unique<InferenceEngine>();
        if (!g_engine->initialize("models/resnet50_imagenet.onnx")) {
            throwRuntime(env, "Failed to initialize inference engine");
            throw std::runtime_error("Engine init failed");
        }
    }
    return *g_engine;
}

} // namespace

extern "C" {

JNIEXPORT jfloatArray JNICALL
Java_ai_controller_AIController_nativeInfer(
    JNIEnv* env,
    jobject /* this */,
    jstring imagePath
) {
    try {
        if (!imagePath) {
            throwRuntime(env, "imagePath is null");
            return nullptr;
        }

        const char* pathChars = env->GetStringUTFChars(imagePath, nullptr);
        std::string path(pathChars ? pathChars : "");
        env->ReleaseStringUTFChars(imagePath, pathChars);

        if (path.empty()) {
            throwRuntime(env, "imagePath is empty");
            return nullptr;
        }

        InferenceEngine& engine = ensureEngine(env);
        std::vector<float> probabilities = engine.classifyImage(path);

        jfloatArray output = env->NewFloatArray(static_cast<jsize>(probabilities.size()));
        if (!output) {
            throwRuntime(env, "Failed to allocate float array for output");
            return nullptr;
        }

        env->SetFloatArrayRegion(output, 0, static_cast<jsize>(probabilities.size()), probabilities.data());
        return output;

    } catch (const std::exception& e) {
        throwRuntime(env, e.what());
        return nullptr;
    }
}

// ============================================
// RL Policy Inference (CartPole)
// ============================================

// Static RL engine (separate from image classification engine)
static std::unique_ptr<InferenceEngine> g_rlEngine;

static InferenceEngine& ensureRLEngine(JNIEnv* env) {
    if (!g_rlEngine) {
        g_rlEngine = std::make_unique<InferenceEngine>();
        if (!g_rlEngine->initializeRL("models/cartpole_policy.onnx")) {
            throwRuntime(env, "Failed to initialize RL policy engine");
            throw std::runtime_error("RL engine init failed");
        }
    }
    return *g_rlEngine;
}

JNIEXPORT jfloatArray JNICALL
Java_ai_controller_AIController_nativeInferState(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray stateArray
) {
    try {
        if (!stateArray) {
            throwRuntime(env, "state array is null");
            return nullptr;
        }

        jsize stateLen = env->GetArrayLength(stateArray);
        if (stateLen <= 0) {
            throwRuntime(env, "state array is empty");
            return nullptr;
        }

        // Copy Java array to C++ vector
        std::vector<float> state(stateLen);
        env->GetFloatArrayRegion(stateArray, 0, stateLen, state.data());

        // Run RL inference
        InferenceEngine& engine = ensureRLEngine(env);
        std::vector<float> actionLogits = engine.inferState(state);

        // Convert back to Java array
        jfloatArray output = env->NewFloatArray(static_cast<jsize>(actionLogits.size()));
        if (!output) {
            throwRuntime(env, "Failed to allocate float array for action logits");
            return nullptr;
        }

        env->SetFloatArrayRegion(output, 0, static_cast<jsize>(actionLogits.size()), actionLogits.data());
        return output;

    } catch (const std::exception& e) {
        throwRuntime(env, e.what());
        return nullptr;
    }
}

} // extern "C"
