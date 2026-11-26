package com.nn.activation;

import org.apache.commons.math3.linear.RealVector;


public enum Softmax implements Activation {
	INSTANCE;

	@Override
	public RealVector apply(RealVector in) {
		double max = in.getMaxValue();
		RealVector exp = in.map(x -> Math.exp(x - max));
		double sum = exp.getL1Norm();
		return exp.mapDivide(sum);
	}

	@Override
	public RealVector derivative(RealVector in, RealVector out) {
		return out.ebeMultiply(
				out.map(x -> 1.0 - x)
		);
	}
}

