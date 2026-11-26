package com.nn.layer;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;


public interface Layer {

	RealVector predict(RealVector in);

	RealVector forward(RealVector in, RealVector target);

	RealVector backward(RealVector prevResult, RealVector loss);

	void add(Layer layer);

	void updateWeightsBiases();

	RealMatrix getWeights();

	RealVector getBiases();

    void setBiases(RealVector realVector);

    void setWeights(RealMatrix realMatrix);

    Layer getNext();
}

