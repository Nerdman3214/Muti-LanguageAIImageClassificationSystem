#include "InferenceEngine.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

/**
 * Test numerical stability of softmax
 * 
 * Critical: softmax(large numbers) should not overflow
 */
TEST(SoftmaxTest, NumericalStability) {
    InferenceEngine engine;
    
    // Extreme values that would overflow naive implementation
    std::vector<float> logits{1000.0f, 1001.0f, 999.0f};
    
    auto probs = engine.softmax(logits);
    
    ASSERT_EQ(probs.size(), 3);
    
    // Check sum = 1
    float sum = probs[0] + probs[1] + probs[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    
    // Check argmax is correct
    auto maxIt = std::max_element(probs.begin(), probs.end());
    size_t maxIdx = std::distance(probs.begin(), maxIt);
    EXPECT_EQ(maxIdx, 1);  // 1001 is largest
    
    // Check no NaN or Inf
    for (float p : probs) {
        EXPECT_FALSE(std::isnan(p));
        EXPECT_FALSE(std::isinf(p));
    }
}

/**
 * Test softmax properties
 */
TEST(SoftmaxTest, MathematicalProperties) {
    InferenceEngine engine;
    std::vector<float> logits{1.0f, 2.0f, 3.0f};
    
    auto probs = engine.softmax(logits);
    
    // All probabilities in [0, 1]
    for (float p : probs) {
        EXPECT_GE(p, 0.0f);
        EXPECT_LE(p, 1.0f);
    }
    
    // Monotonic: higher logit → higher prob
    EXPECT_GT(probs[2], probs[1]);
    EXPECT_GT(probs[1], probs[0]);
}

/**
 * Test ONNX engine initialization
 */
TEST(InferenceEngineTest, Initialization) {
    InferenceEngine engine;
    
    // Should fail with invalid path
    bool result = engine.initialize("nonexistent.onnx");
    EXPECT_FALSE(result);
    
    // Version should still be accessible
    std::string version = engine.getVersion();
    EXPECT_FALSE(version.empty());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}