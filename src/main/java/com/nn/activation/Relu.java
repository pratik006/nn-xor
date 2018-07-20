package com.nn.activation;

import org.apache.commons.math3.linear.RealMatrix;

public enum Relu implements Activation {
	INSTANCE;
	
	@Override
	public RealMatrix apply(RealMatrix in) {
		//TODO
		return in;
	}

	@Override
	public RealMatrix gradient(RealMatrix in) {
		// TODO Auto-generated method stub
		return in;
	}

}
