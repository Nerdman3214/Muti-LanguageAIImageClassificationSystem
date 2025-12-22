#include <jni.h>
#include "InferenceEngine.h"

extern "C" JNIEXPORT jint JNICALL Java_AIController_predict(JNIEnv* env, jobject /*obj*/, jfloatArray arr) {
    if (arr == nullptr) return -1;

    jsize len = env->GetArrayLength(arr);
    if (len == 0) return -1;

    std::vector<float> logits(static_cast<size_t>(len));
    env->GetFloatArrayRegion(arr, 0, len, reinterpret_cast<jfloat*>(logits.data()));

    InferenceEngine engine;
    int prediction = engine.predict(logits);
    return static_cast<jint>(prediction);
}
