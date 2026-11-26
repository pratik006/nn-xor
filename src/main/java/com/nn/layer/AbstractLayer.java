package com.nn.layer;

import com.nn.activation.Activation;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;


public abstract class AbstractLayer implements Layer {

	private final int nodes;
	private final int inputCount;

	protected double learningRate = 0.1d;

	protected RealMatrix weights;
	protected RealMatrix weightDeltas;

	protected RealVector bias;
	protected RealVector biasDeltas;

	protected RealVector result;

	protected Layer next;

	protected List<Activation> activations = new ArrayList<>();
	protected int iterationNo;

	public AbstractLayer(int inputCount, int nodes) {
		this.inputCount = inputCount;
		this.nodes = nodes;
		initWeights();
		resetDeltas();
	}

	public void initWeights() {
		double limit = Math.sqrt(6.0 / (inputCount + nodes));

		double[][] weightValues = new double[nodes][inputCount];
		double[] biasValues = new double[nodes];

		for (int row = 0; row < nodes; row++) {
			for (int col = 0; col < inputCount; col++) {
				weightValues[row][col] =
						(Math.random() * 2.0 - 1.0) * limit;
			}
		}

		this.weights = MatrixUtils.createRealMatrix(weightValues);
		this.bias = MatrixUtils.createRealVector(biasValues);
	}

	public void resetDeltas() {

		double[][] dW = new double[nodes][inputCount];
		double[] dB = new double[nodes];

		this.weightDeltas = MatrixUtils.createRealMatrix(dW);
		this.biasDeltas = MatrixUtils.createRealVector(dB);
	}

	public void add(Layer layer) {
		this.next = layer;
	}

	public RealVector predict(RealVector in) {

		RealVector result = applyActivation(
				weights.operate(in).add(bias)
		);

		return (next == null) ? result : next.predict(result);
	}

	public RealVector forward(RealVector in, RealVector target) {

		iterationNo++;

		result = applyActivation(
				weights.operate(in).add(bias)
		);

		return (next == null) ? result : next.forward(result, target);
	}

	protected RealVector applyActivation(RealVector in) {

		RealVector result = in.copy();

		for (Activation activation : activations) {
			result = activation.apply(result);
		}

		return result;
	}

	protected RealVector applyDerivative(RealVector in) {

		RealVector result = in.copy();

		for (Activation activation : activations) {
			result = activation.derivative(in, result);
		}

		return result;
	}

	protected RealVector backwardUpdate(
			RealVector errorGradient,
			RealVector prevResult) {

		RealMatrix weightErrors = errorGradient.outerProduct(prevResult)
				.scalarMultiply(-learningRate);

		RealVector biasErrors = errorGradient.mapMultiply(-learningRate);

		weightDeltas = weightDeltas.add(weightErrors);
		biasDeltas = biasDeltas.add(biasErrors);

		return weights.transpose().operate(errorGradient);
	}

	public void updateWeightsBiases() {

		this.weights = this.weights.add(weightDeltas);
		this.bias = this.bias.add(biasDeltas);

		iterationNo = 0;
		resetDeltas();

		if (next != null)
			next.updateWeightsBiases();
	}

	public RealMatrix getWeights() {
		return weights;
	}

	public RealVector getBiases() {
		return bias;
	}

	@Override
	public void setBiases(RealVector biases) {
		bias = biases;
	}

	@Override
	public void setWeights(RealMatrix weights) {
		this.weights = weights;
	}

	@Override
	public Layer getNext() {
		return next;
	}
}

