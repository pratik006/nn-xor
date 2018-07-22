package com.nn.loss;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public enum MeanSquared implements Loss {
	INSTANCE;
	
	@Override
	public RealMatrix calcCost(RealMatrix y, RealMatrix target) {
		RealMatrix loss = y.subtract(target);
		double res[] = new double[loss.getRowDimension()];
		for (int i=0;i<loss.getRowDimension();i++) {
			res[i] = loss.getEntry(i, 0);
			res[i] = res[i]*res[i];
		}
		return MatrixUtils.createColumnRealMatrix(res);
	}

}
