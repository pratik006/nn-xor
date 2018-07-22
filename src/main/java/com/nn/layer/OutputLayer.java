package com.nn.layer;

import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Activation;

public class OutputLayer extends AbstractLayer {

	public OutputLayer(int inputCount, int nodes, int batchSize) {
		super(inputCount, nodes, batchSize);
	}
	
	public OutputLayer(int inputCount, int nodes, int batchSize, Activation activation) {
		this(inputCount, nodes, batchSize);
		this.activations.add(activation);
	}
	
	@Override
	public RealMatrix backward(RealMatrix prevResult, RealMatrix loss) {
		RealMatrix derivative = applyDerivative(result);
		RealMatrix errorGradient = scalarMultiply(derivative, loss);
		return backwardUpdate(errorGradient, prevResult);
	}

}
