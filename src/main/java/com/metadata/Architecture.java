package com.metadata;

public record Architecture(
        NetworkType type,
        int inputSize,
        int outputSize,
        int[] hiddenLayers,
        ActivationFunction activation,
        ActivationFunction outputActivation,
        Initialization initialization,
        boolean useBias
) {
    public Architecture() {
        this(
                NetworkType.MLP,
                0,
                0,
                new int[]{8},
                ActivationFunction.TANH,
                ActivationFunction.SIGMOID,
                Initialization.XAVIER,
                true
        );
    }
}
