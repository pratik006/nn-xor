package com.nn.loss;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class CrossEntropy implements Loss {

	@Override
	public RealMatrix calcCost(RealMatrix y, RealMatrix target) {
		double sum = 0;
		double[] loss = new double[target.getRowDimension()];
		for (int i=0;i<target.getRowDimension();i++) {
			sum += target.getEntry(i, 0)*Math.log(y.getEntry(i, 0));
		}
		for (int i=0;i<target.getRowDimension();i++) {
			loss[i] = -sum;
		}
		return MatrixUtils.createColumnRealMatrix(loss);
	}

}
