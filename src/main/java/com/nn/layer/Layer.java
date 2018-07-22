package com.nn.layer;

import org.apache.commons.math3.linear.RealMatrix;

public interface Layer {
	RealMatrix predict(RealMatrix in);
	RealMatrix forward(RealMatrix in, RealMatrix target);
	RealMatrix backward(RealMatrix prevResult, RealMatrix loss);
	void add(Layer layer);
	void updateWeightsBiases();
}
