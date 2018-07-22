package com.nn.layer;

import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Activation;

public class FullyConnected extends AbstractLayer {
	
	public FullyConnected(int inputCount, int nodes, int batchSize) {
		super(inputCount, nodes, batchSize);
	}
	
	public FullyConnected(int inputCount, int nodes, int batchSize, Activation activation) {
		this(inputCount, nodes, batchSize);
		this.activations.add(activation);
	}	
	
	public void add(Layer layer) {
		this.next = layer;
	}
	
	public RealMatrix backward(RealMatrix prevResult, RealMatrix loss) {
		RealMatrix derivative = applyDerivative(result);
		RealMatrix prevWeightErrors = next.backward(this.result, loss);
		RealMatrix errorGradient = scalarMultiply(derivative, prevWeightErrors);
		return backwardUpdate(errorGradient, prevResult);
	}
}
