package com.nn.dnn;

import com.metadata.ModelSpec;

public class NeuralNetworkFactory {
    public NeuralNetwork build(ModelSpec spec) {
        return new NeuralNetwork(spec);
    }
}
