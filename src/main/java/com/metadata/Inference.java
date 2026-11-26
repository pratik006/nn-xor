package com.metadata;

public record Inference(
        double temperature,
        double threshold,
        OutputMode outputMode,
        int topK
) {
    public Inference() {
        this(1.0, 0.5, OutputMode.RAW, 1);
    }
}
