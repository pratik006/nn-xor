package com.nn.activation;

import org.apache.commons.math3.linear.RealVector;


public enum Tanh implements Activation {
	INSTANCE;

	@Override
	public RealVector apply(RealVector in) {
		return in.map(Tanh::tanh);
	}

	@Override
	public RealVector derivative(RealVector in, RealVector out) {
		return out.map(Tanh::dtanh);
	}

	static double tanh(double x) {
		return Math.tanh(x);
	}

	static double dtanh(double x) {
		return 1.0 - x * x;
	}
}

