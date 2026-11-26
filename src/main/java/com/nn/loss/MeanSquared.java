package com.nn.loss;

import org.apache.commons.math3.linear.RealVector;

public enum MeanSquared implements Loss {
	INSTANCE;

	@Override
	public RealVector calcCost(RealVector y, RealVector target) {
		RealVector loss = y.subtract(target);
		return loss.ebeMultiply(loss);
	}

}
