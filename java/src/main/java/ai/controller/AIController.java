public class AIController {

    static {
        System.loadLibrary("inference_engine");
    }

    public native int predict(float[] logits);

    public static void main(String[] args) {
        AIController ai = new AIController();
        float[] modelOutput = {0.1f, 0.2f, 3.4f, 0.05f};
        int prediction = ai.predict(modelOutput);
        System.out.println("Predicted class: " + prediction);
    }
}
