package com.nn.layer;

import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Activation;
import com.nn.loss.AbsoluteLoss;
import com.nn.loss.Loss;

public class OutputLayer extends AbstractLayer {

	private Loss loss = AbsoluteLoss.INSTANCE;
	
	public OutputLayer(int inputCount, int nodes, int batchSize) {
		super(inputCount, nodes, batchSize);
	}
	
	public OutputLayer(int inputCount, int nodes, int batchSize, Activation activation) {
		this(inputCount, nodes, batchSize);
		this.activations.add(activation);
	}
	
	public OutputLayer(int inputCount, int nodes, int batchSize, Activation activation, Loss loss) {
		this(inputCount, nodes, batchSize, activation);
		this.loss = loss;
	}
	
	@Override
	public RealMatrix forward(RealMatrix in, RealMatrix target) {
		iterationNo++;
		result = applyActivation(weights.multiply(in).add(bias));
		return loss.calcCost(result, target);
	}
	
	@Override
	public RealMatrix backward(RealMatrix prevResult, RealMatrix loss) {
		RealMatrix derivative = applyDerivative(result);
		RealMatrix errorGradient = scalarMultiply(derivative, loss);
		return backwardUpdate(errorGradient, prevResult);
	}

}
