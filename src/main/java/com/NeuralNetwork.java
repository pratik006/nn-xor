package com;

import java.util.Arrays;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class NeuralNetwork {
	private int layers;
	private int inputLen;
	private int outputCount;
	private int neuronCount;
	double learningRate = 0.01d;
	RealMatrix[] hiddenMatrix;
	RealMatrix[] weightMatrices;
	RealMatrix[] biasMatrices;
	RealMatrix expectedMatrix;
	double avgErrorRate = 0;
	long iterations;
	
	public NeuralNetwork(int layers, int neuronCount, int outputCount) {
		this.layers = layers;
		this.outputCount = outputCount;
		this.neuronCount = neuronCount;
		initWeights2();
	}
	
	public void initWeights2() {
		double[][][] weights = new double[layers+1][neuronCount][];
		double[][][] biases = new double[layers+1][][];
		weights[0] = new double[][] {
			{0.8476120393755635,-0.6731715020114386},
	        {0.5716613629768057,0.10632328563113491},
	          {0.5842254251316357,0.1262484323075692},
	          {-0.9161971210157449,-0.24694179431984375}
		};
		
		weights[1] = new double[][] {
			{-0.9840260537481855,0.7193805539640015,-0.4294791687687227,-0.3465218493432385}
		};
		
		biases[0] = new double[][] {
			{0.5872951592253921},
			{0.5606661418911605},
			{0.4115604997168312},
			{-0.6242288964207718}
		};
		biases[1] = new double[][] {{-0.6735499685229893}};
		
		this.weightMatrices = new RealMatrix[2];
		this.weightMatrices[0] = MatrixUtils.createRealMatrix(weights[0]);
		this.weightMatrices[1] = MatrixUtils.createRealMatrix(weights[1]);
		this.biasMatrices = new RealMatrix[2];
		this.biasMatrices[0] = MatrixUtils.createRealMatrix(biases[0]);
		this.biasMatrices[1] = MatrixUtils.createRealMatrix(biases[1]);
		
		hiddenMatrix = new RealMatrix[layers + 2];
	}
	
	public void initWeights() {
		double[][][] weights = new double[layers+1][neuronCount][];
		double[][] biases = new double[layers+1][];
		
		for (int i=0;i<=layers;i++) {
			int l = (i==layers) ? outputCount : neuronCount;
			biases[i] = new double[l];
			for (int j=0;j<neuronCount;j++) {
				weights[i][j] = new double[l];
				for (int k=0; k<l; k++) {
					weights[i][j][k] = Math.random();
				}
			}
			
			for (int j=0;j<l;j++) {
				biases[i][j] = Math.random();
			}
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
		biasMatrices = new RealMatrix[layers+1];
		for (int i=0;i<weights.length;i++) {
			weightMatrices[i] = MatrixUtils.createRealMatrix(weights[i]);
			biasMatrices[i] = MatrixUtils.createColumnRealMatrix(biases[i]);
		}
		
		hiddenMatrix = new RealMatrix[layers + 2];
	}
	int iteration = 0;
	public void train(double[] inputs, double[] expectedOutputs) {iteration++;
		inputLen = inputs.length;
		outputCount = expectedOutputs.length;
		this.hiddenMatrix[0] = MatrixUtils.createColumnRealMatrix(inputs);
		this.expectedMatrix = MatrixUtils.createColumnRealMatrix(expectedOutputs);
		iterations++;
		double[] res = forwardPass(1);
		//System.out.println(Arrays.toString(inputs)+"  "+iteration+" "+Arrays.toString(res));
		calculateCost(layers+1);
		//if (expectedOutputs[0]- res[0] > 0.5)
			//System.out.println("avg error rate: "+avgErrorRate+"   "+Arrays.toString(inputs)+"  "+(expectedOutputs[0]- res[0]));
		//System.out.println("weight 2: ["+getString(weightMatrices[1])+"]");
		//System.out.println("weight 1: ["+getString(weightMatrices[0])+"]");
		//System.out.println("bias0: ["+getString(biasMatrices[0])+"]");
		//System.out.println("bias1: ["+getString(biasMatrices[1])+"]");
	}
	
	public double[] predict(double[] inputs) {
		inputLen = inputs.length;
		this.hiddenMatrix[0] = MatrixUtils.createColumnRealMatrix(inputs);
		iterations++;
		return forwardPass(1);
	}
	
	public double[] forwardPass(int layer) {
		if (layer > layers + 1) {			
			return hiddenMatrix[layer-1].getColumn(0);
		}
		RealMatrix inputMatrix = hiddenMatrix[layer-1];
		
		RealMatrix weightMatrix = null;
		/*if (inputMatrix.getColumn(0).length != neuronCount) {
			weightMatrix = weightMatrices[layer-1].getSubMatrix(0, 3, 0, 1);
		} else {*/
			weightMatrix = weightMatrices[layer-1];	
		//}
		
		
		hiddenMatrix[layer] = weightMatrix.multiply(inputMatrix);
		hiddenMatrix[layer] = hiddenMatrix[layer].add(biasMatrices[layer-1].getColumnMatrix(0));
		hiddenMatrix[layer] = sigmoid(hiddenMatrix[layer]);
		return forwardPass(layer+1);
	}

	public void calculateCost(int layer) {
		if (layer == 0)
			return;
		
		RealMatrix outputs = hiddenMatrix[layer];
		RealMatrix activations = hiddenMatrix[layer-1];
		RealMatrix weights = weightMatrices[layer-1];
		RealMatrix error = expectedMatrix.subtract(outputs);
		
		double[] outputErrors = error.getData()[0];
		double[] gradients = Arrays.stream(outputErrors).map(NeuralNetwork::dsigmoid).toArray();
		RealMatrix gradientMatrix = MatrixUtils.createColumnRealMatrix(scalarMultiply(gradients, outputErrors)).scalarMultiply(learningRate);
		
		//RealMatrix errorGradient = dsigmoid(outputs);
		
		// Calculate deltas
		RealMatrix hiddenWeightErrors = gradientMatrix.multiply(activations.transpose());
		weightMatrices[layer-1] = weights.add(hiddenWeightErrors);
		//todo 
		biasMatrices[layer-1] = biasMatrices[layer-1].add(gradientMatrix);
		calculateCost(layer - 1, error);
	}
	
	private void calculateCost(int layer, RealMatrix error) {
		if (layer == 0)
			return;
		
		RealMatrix output = hiddenMatrix[layer];
		RealMatrix activations = hiddenMatrix[layer-1];
		RealMatrix doutput = dsigmoid(output);
		RealMatrix weights = weightMatrices[layer];
		
		RealMatrix hiddenErrors = weights.transpose().multiply(error);
		//calc hidden gradient
		//scalar multiply
		RealMatrix hiddenGradient = scalarMultiply(doutput, hiddenErrors).scalarMultiply(learningRate);
		
		// calc deltas
		RealMatrix weightErrors = hiddenGradient.multiply(activations.transpose());
		//weightMatrices[layer-1] = scalarAdd(weightMatrices[layer-1], weightErrors);
		weightMatrices[layer-1] = weightMatrices[layer-1].add(weightErrors);
		biasMatrices[layer-1] = biasMatrices[layer-1].add(hiddenGradient);
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
	
	public static double dsigmoid(double y) {
		return y*(1 - y);
	}
	
	public static double[] scalarMultiply(double[] x, double[] y) {
		double[] res = new double[x.length];
		for (int i=0;i<x.length;i++) {
			res[i] = x[i] * y[i];
		}
		
		return res;
	}
	
	public static RealMatrix scalarMultiply(RealMatrix xm, RealMatrix ym) {
		double[][] x = xm.getData();
		double[][] y = ym.getData();
		double[][] res = new double[x.length][x[0].length];
		for (int i=0;i<x.length;i++) {
			for (int j=0;j<x[i].length;j++) {
				res[i][j] = x[i][j] * y[i][j];	
			}
		}
		
		return MatrixUtils.createRealMatrix(res);
	}
	
	public static RealMatrix scalarAdd(RealMatrix xm, RealMatrix ym) {
		double[][] x = xm.getData();
		double[][] y = ym.getData();
		
		if (x.length == y.length && x[0].length == y[0].length) {
			return xm.add(ym);
		}
		
		double[][] res = new double[x.length][x[0].length];
		for (int i=0;i<min(x.length, y.length);i++) {
			for (int j=0;j<min(x[i].length, y[i].length);j++) {
				res[i][j] = x[i][j] * y[i][j];
			}
		}
		
		return MatrixUtils.createRealMatrix(res);
	}
	
	public static int min(int...vals) {
		return Arrays.stream(vals).min().getAsInt();
	}
	
	public static String getString(RealMatrix matrix) {
		StringBuilder sb = new StringBuilder();
		for (int i=0;i<matrix.getRowDimension();i++) {
			for (int j=0;j<matrix.getColumnDimension();j++) {
				sb.append(matrix.getData()[i][j]).append(",");
			}
		}
		
		return sb.toString();
	}
}
