package com.nn.loss;

import org.apache.commons.math3.linear.RealVector;

public interface Loss {
	RealVector calcCost(RealVector y, RealVector target);
}
