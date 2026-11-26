package com.metadata;

public record Training(
        Optimizer optimizer,
        LossFunction lossFunction,
        double learningRate,
        int epochs,
        int batchSize,
        double momentum,
        double beta1,
        double beta2,
        double epsilon,
        double weightDecay,
        double dropout,
        boolean earlyStopping,
        int patience,
        double gradientClipping,
        long randomSeed
) {
    public Training() {
        this(
                Optimizer.SGD,
                LossFunction.MSE,
                0.001,
                1000,
                32,
                0.0,
                0.9,
                0.999,
                1e-8,
                0.0,
                0.0,
                false,
                10,
                0.0,
                42
        );
    }
}
