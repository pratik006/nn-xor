package com.nn.dnn;

import com.metadata.ActivationFunction;
import com.metadata.Architecture;
import com.metadata.ModelSpec;
import com.nn.activation.Activation;
import com.nn.activation.Relu;
import com.nn.activation.Sigmoid;
import com.nn.activation.Tanh;
import com.nn.layer.FullyConnected;
import com.nn.layer.Layer;
import com.nn.layer.OutputLayer;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NeuralNetwork {
	public double loss = 1.0;
	private int batchSize = 1;
	
	public NeuralNetwork() {}
	
	public NeuralNetwork(int batchSize) {
		this.batchSize = batchSize;
	}

	public NeuralNetwork(ModelSpec spec) {
		Architecture arch = spec.architecture();
		int previousSize = arch.inputSize();

		Layer root = null;
		Layer previous = null;

		// Build hidden layers
		for (int nodes : arch.hiddenLayers()) {

			Layer current = new FullyConnected(
					previousSize,
					nodes,
					getActivation(arch.activation())
			);

			if (root == null) {
				root = current;
			} else {
				previous.add(current);
			}

			previous = current;
			previousSize = nodes;
		}

		// Build output layer
		Layer output = new OutputLayer(
				previousSize,
				arch.outputSize(),
				getActivation(arch.outputActivation())
		);

		if (root == null) {
			root = output;
		} else {
			previous.add(output);
		}

		setRoot(root);
	}

	private Activation getActivation(ActivationFunction activation) {
		return switch (activation) {
			case TANH -> Tanh.INSTANCE;
			case SIGMOID -> Sigmoid.INSTANCE;
			case RELU -> Relu.INSTANCE;
			//case LINEAR -> Linear.INSTANCE;
			default -> throw new IllegalArgumentException(
					"Unsupported activation: " + activation
			);
		};
	}
	
	private Layer root;
	
	public Layer getRoot() {
		return root;
	}

	public void setRoot(Layer root) {
		this.root = root;
	}

	public void train(double[][] inputs, double[][] outputs) {
		double avgLoss = 0;
		long epoch = inputs.length/batchSize;

		long sampleCount = 0;
		double totalLoss = 0;

		for (int i = 0; i < epoch; i++) {
			RealVector[] losses = new RealVector[batchSize];

			for (int j = 0; j < batchSize; j++) {
				int index = batchSize * i + j;
				RealVector expected =
						MatrixUtils.createRealVector(outputs[index]);
				RealVector input =
						MatrixUtils.createRealVector(inputs[index]);
				losses[j] = root.forward(input, expected);
				root.backward(input, losses[j]);
				totalLoss += losses[j].getL1Norm();
				sampleCount++;
			}

			root.updateWeightsBiases();
			loss = totalLoss / sampleCount;
		}
	}
	
	public void train(int[][] inputs, int[][] outputs) {
		double[][] dinputs = new double[inputs.length][inputs[0].length];
		double[][] doutputs = new double[inputs.length][outputs[0].length];
		
		for (int i=0;i<inputs.length;i++) {
			for(int j=0;j<inputs[i].length;j++) {
				dinputs[i][j] = (double)inputs[i][j];
			}
		}
		for (int i=0;i<outputs.length;i++) {
			for(int j=0;j<outputs[i].length;j++) {
				doutputs[i][j] = (double)outputs[i][j];
			}
		}
		train(dinputs, doutputs);
	}
	
	public double[] predict(double[] input) {
		return root.predict(MatrixUtils.createRealVector(input)).toArray();
	}
	
	public int[] predict(int[] inputs) {
		return toInt(predict(toDouble(inputs)));
	}
	
	public RealMatrix avgLoss(RealMatrix[] losses) {
		double[] res = new double[losses[0].getRowDimension()];
		for (int i=0;i<losses.length;i++) {
			for (int j=0;j<losses[i].getRowDimension();j++) {
				res[j] +=  losses[i].getEntry(j, 0);
			}
		}
		for (int i=0;i<losses[0].getRowDimension();i++) {
			res[i] = res[i]/losses.length;
		}
		return MatrixUtils.createColumnRealMatrix(res);
	}
	
	public RealMatrix meanSquaredLoss(RealMatrix[] losses) {
		double[] res = new double[losses[0].getRowDimension()];
		for (int i=0;i<losses.length;i++) {
			for (int j=0;j<losses[i].getRowDimension();j++) {
				double loss = losses[i].getEntry(j, 0); 
				res[j] +=  loss*loss;
			}
		}
		for (int i=0;i<losses[0].getRowDimension();i++) {
			res[i] = res[i]/losses.length;
		}
		return MatrixUtils.createColumnRealMatrix(res);
	}
	
	public double[] toDouble(int[] inputs) {
		double[] dinputs = new double[inputs.length];
		for (int i=0;i<inputs.length;i++) {
			dinputs[i] = (double)inputs[i];
		}
		return dinputs;
	}
	
	public int[] toInt(double[] inputs) {
		int[] dinputs = new int[inputs.length];
		for (int i=0;i<inputs.length;i++) {
			dinputs[i] = (int) Math.round(inputs[i]);
		}
		return dinputs;
	}

	public List<Layer> getLayers() {
		List<Layer> layers = new ArrayList<>();

		Layer current = root;

		while (current != null) {
			layers.add(current);
			current = current.getNext(); // Use your actual next-layer getter
		}

		return Collections.unmodifiableList(layers);
	}
}
