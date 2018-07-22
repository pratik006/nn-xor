package com.nn.layer;

import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Activation;

public class OutputLayer extends AbstractLayer {

	public OutputLayer(int inputCount, int nodes) {
		super(inputCount, nodes);
	}
	
	public OutputLayer(int inputCount, int nodes, Activation activation) {
		this(inputCount, nodes);
		this.activations.add(activation);
	}
	
	@Override
	public RealMatrix backward(RealMatrix prevResult, RealMatrix loss) {
		RealMatrix derivative = applyDerivative(result);
		RealMatrix errorGradient = scalarMultiply(derivative, loss);
		return backwardUpdate(errorGradient, prevResult);
	}

}
