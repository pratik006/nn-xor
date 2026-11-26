package com.nn.loss;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class CrossEntropy implements Loss {

	@Override
	public RealVector calcCost(RealVector y, RealVector target) {
		double[] loss = new double[target.getDimension()];
		for (int i = 0; i < target.getDimension(); i++) {
			double prediction = Math.max(1e-15,
					Math.min(1.0 - 1e-15, y.getEntry(i)));

			loss[i] = -target.getEntry(i) * Math.log(prediction);
		}
		return MatrixUtils.createRealVector(loss);
	}
}
