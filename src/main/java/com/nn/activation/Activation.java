package com.nn.activation;

import org.apache.commons.math3.linear.RealMatrix;

public interface Activation {
	RealMatrix apply(RealMatrix in);
	RealMatrix derivative(RealMatrix in);
}
