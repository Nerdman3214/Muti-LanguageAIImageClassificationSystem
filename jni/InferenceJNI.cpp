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

} // extern "C"
