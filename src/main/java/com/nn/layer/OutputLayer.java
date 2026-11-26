package com.nn.layer;

import com.nn.activation.Activation;
import com.nn.loss.AbsoluteLoss;
import com.nn.loss.Loss;
import org.apache.commons.math3.linear.RealVector;


public class OutputLayer extends AbstractLayer {

	private Loss loss = AbsoluteLoss.INSTANCE;

	public OutputLayer(int inputCount, int nodes) {
		super(inputCount, nodes);
	}

	public OutputLayer(int inputCount, int nodes, Activation activation) {
		this(inputCount, nodes);
		this.activations.add(activation);
	}

	public OutputLayer(int inputCount, int nodes, Activation activation, Loss loss) {
		this(inputCount, nodes, activation);
		this.loss = loss;
	}

	@Override
	public RealVector forward(RealVector in, RealVector target) {

		iterationNo++;

		result = applyActivation(
				weights.operate(in).add(bias)
		);

		return loss.calcCost(result, target);
	}

	@Override
	public RealVector backward(RealVector prevResult, RealVector lossGradient) {

		RealVector derivative = applyDerivative(result);

		RealVector errorGradient = derivative.ebeMultiply(lossGradient);

		return backwardUpdate(errorGradient, prevResult);
	}
}

