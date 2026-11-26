package com.nn.activation;

import org.apache.commons.math3.linear.RealVector;


public enum Relu implements Activation {
	INSTANCE;

	@Override
	public RealVector apply(RealVector in) {
		return in.map(x -> Math.max(0.0, x));
	}

	@Override
	public RealVector derivative(RealVector in, RealVector out) {
		return out.map(x -> x > 0.0 ? 1.0 : 0.0);
	}
}

