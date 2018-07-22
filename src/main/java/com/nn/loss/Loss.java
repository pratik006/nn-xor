package com.nn.loss;

import org.apache.commons.math3.linear.RealMatrix;

public interface Loss {
	RealMatrix calcCost(RealMatrix y, RealMatrix target);
}
