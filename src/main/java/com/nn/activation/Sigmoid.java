package com.nn.activation;

import java.util.Arrays;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public enum Sigmoid implements Activation {
	INSTANCE;
	
	@Override
	public RealMatrix apply(RealMatrix in) {
		return MatrixUtils.createColumnRealMatrix(Arrays.stream(in.getColumn(0)).map(Sigmoid::sigmoid).toArray());
	}

	@Override
	public RealMatrix gradient(RealMatrix in) {
		// TODO Auto-generated method stub
		return null;
	}
	
	static double sigmoid(double val) {
		return 1/(1+Math.exp(-val));
	}

}
