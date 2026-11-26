package com.nn.loss;

import org.apache.commons.math3.linear.RealVector;

public enum AbsoluteLoss implements Loss {
	INSTANCE;
	
	@Override
	public RealVector calcCost(RealVector y, RealVector target) {
		return y.subtract(target);
	}

}
