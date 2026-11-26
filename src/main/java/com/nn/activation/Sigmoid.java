package com.nn.activation;

import org.apache.commons.math3.linear.RealVector;


public enum Sigmoid implements Activation {
	INSTANCE;

	@Override
	public RealVector apply(RealVector in) {
		return in.map(Sigmoid::sigmoid);
	}

	@Override
	public RealVector derivative(RealVector in, RealVector out) {
		return out.map(Sigmoid::dsigmoid);
	}

	static double sigmoid(double val) {
		return 1.0 / (1.0 + Math.exp(-val));
	}

	static double dsigmoid(double val) {
		return val * (1.0 - val);
	}
}
