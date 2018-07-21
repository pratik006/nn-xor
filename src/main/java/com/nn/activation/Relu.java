package com.nn.activation;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public enum Relu implements Activation {
	INSTANCE;
	
	@Override
	public RealMatrix apply(RealMatrix in) {
		double data[][] = new double[in.getRowDimension()][in.getColumnDimension()]; 
		for (int i=0;i<in.getRowDimension();i++) {
			for (int j=0;j<in.getColumnDimension();j++) {
				data[i][j] = Math.max(0, in.getEntry(i, j));
			}
		}
		return MatrixUtils.createRealMatrix(data);
	}

	@Override
	public RealMatrix derivative(RealMatrix in) {
		double data[][] = new double[in.getRowDimension()][in.getColumnDimension()]; 
		for (int i=0;i<in.getRowDimension();i++) {
			for (int j=0;j<in.getColumnDimension();j++) {
				data[i][j] = in.getEntry(i, j)>0?1 : 0;
			}
		}
		return MatrixUtils.createRealMatrix(data);
	}

}
