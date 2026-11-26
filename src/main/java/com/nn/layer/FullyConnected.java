package com.nn.layer;

import com.nn.activation.Activation;
import org.apache.commons.math3.linear.RealVector;

public class FullyConnected extends AbstractLayer {
	
	public FullyConnected(int inputCount, int nodes) {
		super(inputCount, nodes);
	}
	
	public FullyConnected(int inputCount, int nodes, Activation activation) {
		this(inputCount, nodes);
		this.activations.add(activation);
	}	
	
	public RealVector backward(RealVector prevResult, RealVector loss) {
		RealVector derivative = applyDerivative(result);
		RealVector prevWeightErrors = next.backward(this.result, loss);
		RealVector errorGradient = derivative.ebeMultiply(prevWeightErrors);
		return backwardUpdate(errorGradient, prevResult);
	}
}
