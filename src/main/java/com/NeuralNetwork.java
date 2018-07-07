package com;

import java.util.Arrays;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class NeuralNetwork {
	private int layers;
	private int outputCount;
	private int inputCount;
	double learningRate = 0.5d;
	RealMatrix[] hiddenMatrix;
	RealMatrix[] weightMatrices;
	RealMatrix biasMatrices;
	RealMatrix expectedMatrix;
	double avgErrorRate = 0;
	long iterations;
	
	public NeuralNetwork(int layers, int inputCount, int outputCount) {
		this.layers = layers;
		this.outputCount = outputCount;
		this.inputCount = inputCount;
		initWeights();
	}
	
	public void initWeights() {
		double[][][] weights = new double[layers+1][inputCount][];
		double[] biases = new double[layers+1];
		
		for (int i=0;i<=layers;i++) {
			int l = (i==layers) ? outputCount : inputCount;
			for (int j=0;j<inputCount;j++) {
				weights[i][j] = new double[l];
				for (int k=0; k<l; k++) {
					weights[i][j][k] = Math.random();
				}
			}
			biases[i] = Math.random();
		}
		
		/*weights = new double[][][] {
			{{0.3d, 0.45d}, {0.56d, 0.12d}},
			{{0.43d, 0.65d}, {0.78d, 0.32d}},
			{{0.12d, 0.85d}, {0.32d, 0.56d}}
		};
		biases = new double[][] {
			{ 0.35f, 0.35f },
			{ 0.45d, 0.45d },
			{ 0.65d, 0.13d }
		};*/
		weightMatrices = new RealMatrix[weights.length];
		biasMatrices = MatrixUtils.createColumnRealMatrix(biases);
		for (int i=0;i<weights.length;i++) {
			weightMatrices[i] = MatrixUtils.createRealMatrix(weights[i]);
		}
		hiddenMatrix = new RealMatrix[layers + 2];
	}
	
	public void train(double[] inputs, double[] expectedOutputs) {
		this.hiddenMatrix[0] = MatrixUtils.createColumnRealMatrix(inputs);
		this.expectedMatrix = MatrixUtils.createColumnRealMatrix(expectedOutputs);
		iterations++;
		double[] res = forwardPass(1);
		calculateCost(layers+1);
		//if (expectedOutputs[0]- res[0] > 0.5)
			System.out.println("avg error rate: "+avgErrorRate+"   "+Arrays.toString(inputs)+"  "+(expectedOutputs[0]- res[0]));
		/*System.out.println("error: "+err+"[ bias1: "+Arrays.toString(biasMatrices[0].getData()[0])+", "+Arrays.toString(biasMatrices[1].getData()[0])+"], [weight1: "
				+Arrays.toString(weightMatrices[0].getData()[0])+Arrays.toString(weightMatrices[0].getData()[1])+"]");*/
	}
	
	public double[] getResults(double[] inputs) {
		this.hiddenMatrix[0] = MatrixUtils.createColumnRealMatrix(inputs);
		return forwardPass(1);
	}
	
	public double[] forwardPass(int layer) {
		if (layer > layers + 1) {
			hiddenMatrix[layer-1] = sigmoid(hiddenMatrix[layer-1]);
			return hiddenMatrix[layer-1].getColumn(0);
		}
		RealMatrix inputMatrix = hiddenMatrix[layer-1];
		RealMatrix weightMatrix = weightMatrices[layer-1];
		hiddenMatrix[layer] = weightMatrix.transpose().multiply(inputMatrix);
		hiddenMatrix[layer] = hiddenMatrix[layer].scalarAdd(biasMatrices.getColumn(0)[0]);
		return forwardPass(layer+1);
	}

	public void calculateCost(int layer) {
		if (layer == 0)
			return;
		
		RealMatrix outputs = hiddenMatrix[layer];
		RealMatrix activations = hiddenMatrix[layer-1];
		RealMatrix weights = weightMatrices[layer-1];
		RealMatrix error = expectedMatrix.subtract(outputs);
		
		double[] temp = new double[error.getData()[0].length];
		for (int i=0;i<error.getData()[0].length;i++) {
			temp[i] = error.getData()[0][i]*error.getData()[0][i]/2;
		}
		error = MatrixUtils.createColumnRealMatrix(temp);
		avgErrorRate = (avgErrorRate*(iterations-1) + error.getData()[0][0])/(iterations);
		
		RealMatrix errorGradient = dsigmoid(outputs);
		
		double[] res = new double[error.getColumn(0).length];
		for (int i=0;i<error.getColumn(0).length;i++) {
			res[i] = error.getColumn(0)[i] * errorGradient.getColumn(0)[i];
		}
		RealMatrix finalErrorMatrix = MatrixUtils.createColumnRealMatrix(res);
		
		RealMatrix hiddenWeightErrors = finalErrorMatrix.multiply(activations.transpose()).scalarMultiply(learningRate);
		weightMatrices[layer-1] = weights.add(hiddenWeightErrors.transpose());
		
		//calculate bias errors
		RealMatrix biasErrors = finalErrorMatrix.scalarMultiply(learningRate);
		biasMatrices = biasMatrices.add(MatrixUtils.createColumnRealMatrix(new double[]{biasErrors.getColumn(0)[0], biasErrors.getColumn(0)[0]}));
		//print(weights);
		
		calculateCost(layer - 1, error);
	}
	
	private void calculateCost(int layer, RealMatrix error) {
		if (layer == 0)
			return;
		
		RealMatrix output = hiddenMatrix[layer];
		RealMatrix doutput = dsigmoid(sigmoid(output));
		RealMatrix weights = weightMatrices[layer];
		RealMatrix hiddenErrors = weights.multiply(error);
		
		double[][] res = new double[output.getRowDimension()][output.getColumnDimension()];
		for (int i=0;i<hiddenErrors.getRowDimension();i++) {
			for (int j=0;j<hiddenErrors.getColumnDimension();j++) {
				res[i][j] = hiddenErrors.getData()[i][j] * doutput.getData()[i][j];
			}
		}
		hiddenErrors = MatrixUtils.createRealMatrix(res);
		
		RealMatrix weightErrors = hiddenErrors.multiply(hiddenMatrix[layer-1].transpose());
		weightMatrices[layer-1] = weightMatrices[layer-1].add(weightErrors);
		
		RealMatrix biasErrors = hiddenErrors.scalarMultiply(learningRate);
		biasMatrices = biasMatrices.add(biasErrors);
		calculateCost(layer - 1, hiddenErrors);
	}
	
	private void print(RealMatrix matrix) {
		System.out.println("[");
		for (int i=0;i<matrix.getData().length;i++) {
			for (int j=0;j<matrix.getData()[i].length;j++) {
				System.out.print(matrix.getData()[i][j]+"    ");
			}
			System.out.println();
		}
		System.out.println("]");
	}
	
	static double sigmoid(double val) {
		return 1/(1+Math.exp(-val));
	}
	
	static RealMatrix sigmoid(RealMatrix matrix) {
		return MatrixUtils.createColumnRealMatrix(Arrays.stream(matrix.getColumn(0)).map(NeuralNetwork::sigmoid).toArray());
	}
	
	public static RealMatrix dsigmoid(RealMatrix matrix) {
		return MatrixUtils.createColumnRealMatrix(Arrays.stream(matrix.getColumn(0)).map(val->val*(1 - val)).toArray());
	}
}
