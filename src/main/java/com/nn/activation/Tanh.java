package com.nn.activation;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public enum Tanh implements Activation {
	INSTANCE;
	
	@Override
	public RealMatrix apply(RealMatrix in) {
		double[][] data = new double[in.getRowDimension()][in.getColumnDimension()];
		for (int i=0;i<in.getRowDimension();i++) {
			for(int j=0;j<in.getColumnDimension();j++) {
				data[i][j] = tanh(in.getEntry(i, j));
			}
		}
		return MatrixUtils.createRealMatrix(data);
	}

	@Override
	public RealMatrix derivative(RealMatrix in, RealMatrix out) {
		double[][] data = new double[out.getRowDimension()][out.getColumnDimension()];
		for (int i=0;i<out.getRowDimension();i++) {
			for(int j=0;j<out.getColumnDimension();j++) {
				data[i][j] = dtanh(out.getEntry(i, j));
			}
		}
		return MatrixUtils.createRealMatrix(data);
	}
	
	static double tanh(double x) {
		return Math.tanh(x);
	}

	static double dtanh(double x) {
		double y = tanh(x);
		return 1 - y*y;
	}
}
