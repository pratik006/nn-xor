package com.nn.activation;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public enum Sigmoid implements Activation {
	INSTANCE;
	
	@Override
	public RealMatrix apply(RealMatrix in) {
		double[][] data = new double[in.getRowDimension()][in.getColumnDimension()];
		for (int i=0;i<in.getRowDimension();i++) {
			for(int j=0;j<in.getColumnDimension();j++) {
				data[i][j] = sigmoid(in.getEntry(i, j));
			}
		}
		return MatrixUtils.createRealMatrix(data);
	}

	@Override
	public RealMatrix gradient(RealMatrix in) {
		double[][] data = new double[in.getRowDimension()][in.getColumnDimension()];
		for (int i=0;i<in.getRowDimension();i++) {
			for(int j=0;j<in.getColumnDimension();j++) {
				data[i][j] = dsigmoid(in.getEntry(i, j));
			}
		}
		return MatrixUtils.createRealMatrix(data);
	}
	
	static double sigmoid(double val) {
		return 1/(1+Math.exp(-val));
	}

	static double dsigmoid(double val) {
		return val*(1-val);
	}
}
