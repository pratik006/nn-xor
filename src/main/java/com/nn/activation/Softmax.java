package com.nn.activation;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public enum Softmax implements Activation {
	INSTANCE;
	
	@Override
	public RealMatrix apply(RealMatrix in) {
		double sum = 0;
		double data[][] = new double[in.getRowDimension()][in.getColumnDimension()]; 
		for (int i=0;i<in.getRowDimension();i++) {
			for (int j=0;j<in.getColumnDimension();j++) {
				double e = Math.exp(in.getEntry(i, j));
				sum += e;
			}
		}
		
		for (int i=0;i<data.length;i++) {
			for (int j=0;j<data[0].length;j++) {
				data[i][j] = data[i][j]/sum;
			}
		}
		
		
		return MatrixUtils.createRealMatrix(data);
	}

	@Override
	public RealMatrix derivative(RealMatrix in) {
		return in;
	}

}
