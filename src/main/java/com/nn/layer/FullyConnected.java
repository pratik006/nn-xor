package com.nn.layer;

import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Activation;

public class FullyConnected extends AbstractLayer {
	
	public FullyConnected(int inputCount, int nodes) {
		super(inputCount, nodes);
	}
	
	public FullyConnected(int inputCount, int nodes, Activation activation) {
		this(inputCount, nodes);
		this.activations.add(activation);
	}	
	
	public RealMatrix backward(RealMatrix prevResult, RealMatrix loss) {
		RealMatrix derivative = applyDerivative(result);
		RealMatrix prevWeightErrors = next.backward(this.result, loss);
		RealMatrix errorGradient = scalarMultiply(derivative, prevWeightErrors);
		return backwardUpdate(errorGradient, prevResult);
	}
}
