package com.nn.activation;

import org.apache.commons.math3.linear.RealVector;


public interface Activation {
	RealVector apply(RealVector in);
	RealVector derivative(RealVector in, RealVector out);
}

