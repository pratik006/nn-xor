package com;

import java.math.BigDecimal;
import java.util.Arrays;

import com.matrix.BigDecimalMatrix;


public class NeuralNetwork {
	public static final int SCALE = 17;
	public static final BigDecimal MINUS_ONE = new BigDecimal("-1");
	
	private int layers;
	private int outputCount;
	private int neuronCount;
	BigDecimal learningRate = new BigDecimal("0.01");
	BigDecimalMatrix[] hiddenMatrix;
	BigDecimalMatrix[] weightMatrices;
	BigDecimalMatrix[] biasMatrices;
	BigDecimalMatrix expectedMatrix;
	double avgErrorRate = 0;
	long iterations;
	
	public NeuralNetwork(int layers, int neuronCount, int outputCount) {
		this.layers = layers;
		this.outputCount = outputCount;
		this.neuronCount = neuronCount;
		initWeights();
	}
	
	public void initWeights2() {
		String[][][] weights = new String[layers+1][neuronCount][];
		String[][][] biases = new String[layers+1][][];
		weights[0] = new String[][] {
			{"0.8476120393755635","-0.6731715020114386"},
	        {"0.5716613629768057","0.10632328563113491"},
	          {"0.5842254251316357","0.1262484323075692"},
	          {"-0.9161971210157449","-0.24694179431984375"}
		};
		
		weights[1] = new String[][] {
			{"-0.9840260537481855","0.7193805539640015","-0.4294791687687227","-0.3465218493432385"}
		};
		
		biases[0] = new String[][] {
			{"0.5872951592253921"},
			{"0.5606661418911605"},
			{"0.4115604997168312"},
			{"-0.6242288964207718"}
		};
		biases[1] = new String[][] {{"-0.6735499685229893"}};
		
		this.weightMatrices = new BigDecimalMatrix[2];
		this.weightMatrices[0] = createRealMatrix(weights[0]);
		this.weightMatrices[1] = createRealMatrix(weights[1]);
		this.biasMatrices = new BigDecimalMatrix[2];
		this.biasMatrices[0] = createRealMatrix(biases[0]);
		this.biasMatrices[1] = createRealMatrix(biases[1]);
		
		hiddenMatrix = new BigDecimalMatrix[layers + 2];
	}
	
	public void initWeights() {
		String[][][] weights = new String[layers+1][neuronCount][];
		String[][] biases = new String[layers+1][];
		
		for (int i=0;i<=layers;i++) {
			int l = (i==layers) ? outputCount : (i==0)?2 : neuronCount;
			for (int j=0;j<neuronCount;j++) {
				weights[i][j] = new String[l];
				for (int k=0; k<l; k++) {
					weights[i][j][k] = Math.random()+"";
				}
			}
			int m = i==layers ? outputCount : neuronCount;
			biases[i] = new String[m];
			for (int j=0;j<m;j++)
				biases[i][j] = Math.random()+"";
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
		weightMatrices = new BigDecimalMatrix[weights.length];
		biasMatrices = new BigDecimalMatrix[layers+1];
		for (int i=0;i<weights.length;i++) {
			if (i==layers)
				weightMatrices[i] = createRealMatrix(weights[i]).transpose();
			else 
				weightMatrices[i] = createRealMatrix(weights[i]);
		}
		
		for (int i=0;i<biases.length;i++) {
			biasMatrices[i] = createColumnRealMatrix(biases[i]);
		}
		
		hiddenMatrix = new BigDecimalMatrix[layers + 2];
	}
	int iteration = 0;
	public void train(double[] inputs, double[] expectedOutputs) {iteration++;
		outputCount = expectedOutputs.length;
		this.hiddenMatrix[0] = createColumnRealMatrix(inputs);
		this.expectedMatrix = createColumnRealMatrix(expectedOutputs);
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
		this.hiddenMatrix[0] = createColumnRealMatrix(inputs);
		iterations++;
		return forwardPass(1);
	}
	
	public double[] forwardPass(int layer) {
		if (layer > layers + 1) {			
			return new double[] {hiddenMatrix[layer-1].getElement(1, 1).doubleValue()};
		}
		BigDecimalMatrix inputMatrix = hiddenMatrix[layer-1];
		
		BigDecimalMatrix weightMatrix = null;
		/*if (inputMatrix.getColumn(0).length != neuronCount) {
			weightMatrix = weightMatrices[layer-1].getSubMatrix(0, 3, 0, 1);
		} else {*/
			weightMatrix = weightMatrices[layer-1];	
		//}
		
		
		hiddenMatrix[layer] = weightMatrix.multiply(inputMatrix);
		hiddenMatrix[layer] = hiddenMatrix[layer].add(biasMatrices[layer-1]);
		hiddenMatrix[layer] = sigmoid(hiddenMatrix[layer]);
		return forwardPass(layer+1);
	}

	public void calculateCost(int layer) {
		if (layer == 0)
			return;
		
		BigDecimalMatrix outputs = hiddenMatrix[layer];
		BigDecimalMatrix activations = hiddenMatrix[layer-1];
		BigDecimalMatrix weights = weightMatrices[layer-1];
		BigDecimalMatrix error = expectedMatrix.subtract(outputs);
		
		BigDecimalMatrix gradients = dsigmoid(outputs);
		BigDecimalMatrix gradientMatrix = scalarMultiply(gradients, error).multiply(learningRate);
		
		// Calculate deltas
		BigDecimalMatrix hiddenWeightErrors = gradientMatrix.multiply(activations.transpose());
		weightMatrices[layer-1] = weights.add(hiddenWeightErrors);
		biasMatrices[layer-1] = biasMatrices[layer-1].add(gradientMatrix);
		calculateCost(layer - 1, error);
	}
	
	private void calculateCost(int layer, BigDecimalMatrix error) {
		if (layer == 0)
			return;
		
		BigDecimalMatrix output = hiddenMatrix[layer];
		BigDecimalMatrix activations = hiddenMatrix[layer-1];
		BigDecimalMatrix doutput = dsigmoid(output);
		BigDecimalMatrix weights = weightMatrices[layer];
		
		BigDecimalMatrix hiddenErrors = (weights.transpose()).multiply(error);
		//calc hidden gradient
		//scalar multiply
		BigDecimalMatrix hiddenGradient = scalarMultiply(doutput, hiddenErrors).multiply(learningRate);
		
		// calc deltas
		BigDecimalMatrix weightErrors = hiddenGradient.multiply(activations.transpose());
		//weightMatrices[layer-1] = scalarAdd(weightMatrices[layer-1], weightErrors);
		weightMatrices[layer-1] = weightMatrices[layer-1].add(weightErrors);
		biasMatrices[layer-1] = biasMatrices[layer-1].add(hiddenGradient);
		calculateCost(layer - 1, hiddenErrors);
	}
	
	private void print(BigDecimalMatrix matrix) {
		System.out.println("[");
		for (int i=0;i<matrix.getHeight();i++) {
			for (int j=0;j<matrix.getWidth();j++) {
				System.out.print(matrix.getElement(i, j)+"    ");
			}
			System.out.println();
		}
		System.out.println("]");
	}
	
	static BigDecimal sigmoid(BigDecimal val) {
		return new BigDecimal(1/(1+Math.exp(-val.doubleValue())));
	}
	
	static BigDecimalMatrix sigmoid(BigDecimalMatrix matrix) {
		BigDecimalMatrix result = new BigDecimalMatrix(matrix.getHeight(), matrix.getWidth(), 20);
		for (int i=1;i<=matrix.getHeight();i++) {
			for (int j=1;j<=matrix.getWidth();j++) {
				result.setElement(i, j, sigmoid(matrix.getElement(i, j)));
			}
		}
		return result;
	}
	
	public static BigDecimalMatrix dsigmoid(BigDecimalMatrix matrix) {
		BigDecimalMatrix result = new BigDecimalMatrix(matrix.getHeight(), matrix.getWidth(), SCALE);
		for (int i=1;i<=matrix.getHeight();i++) {
			for (int j=1;j<=matrix.getWidth();j++) {
				result.setElement(i, j, dsigmoid(matrix.getElement(i, j)));
			}
		}
		return result;
	}
	
	public static BigDecimal dsigmoid(BigDecimal y) {
		return y.multiply(BigDecimal.ONE.subtract(y));
	}
	
	public static BigDecimalMatrix scalarMultiply(BigDecimalMatrix x, BigDecimalMatrix y) {
		BigDecimalMatrix res = new BigDecimalMatrix(x.getHeight(), x.getWidth(), 20);
		for (int i=1;i<=x.getHeight();i++) {
			for (int j=1;j<=x.getWidth();j++) {
				res.setElement(i, j, x.getElement(i, j).multiply(y.getElement(i, j)));				
			}
		}
		
		return res;
	}
	
	public static BigDecimalMatrix scalarAdd(BigDecimalMatrix xm, BigDecimalMatrix ym) {
		BigDecimalMatrix r = new BigDecimalMatrix(xm.getRowDimension(), xm.getColumnDimension(), 20);
		
		if (xm.getRowDimension() == ym.getRowDimension() && xm.getColumnDimension() == ym.getColumnDimension()) {
			return xm.add(ym);
		}
		
		/*double[][] res = new double[x.length][x[0].length];
		for (int i=0;i<min(x.length, y.length);i++) {
			for (int j=0;j<min(x[i].length, y[i].length);j++) {
				res[i][j] = x[i][j] * y[i][j];
			}
		}*/
		
		return r;
	}
	
	public static int min(int...vals) {
		return Arrays.stream(vals).min().getAsInt();
	}
	
	public static String getString(BigDecimalMatrix matrix) {
		StringBuilder sb = new StringBuilder();
		for (int i=0;i<matrix.getRowDimension();i++) {
			for (int j=0;j<matrix.getColumnDimension();j++) {
				sb.append(matrix.getElement(i+1, j+1)).append(",");
			}
		}
		
		return sb.toString();
	}
	
	public static BigDecimalMatrix createRealMatrix(String[][] data) {
		BigDecimalMatrix matrix = new BigDecimalMatrix(data.length, data[0].length, SCALE);
		for (int i=0;i<data.length;i++) {
			for (int j=0;j<data[i].length;j++) {
				matrix.setElement(i+1, j+1, new BigDecimal(data[i][j]));
			}
		}
		return matrix;
	}
	
	public static BigDecimalMatrix createColumnRealMatrix(String[] data) {
		BigDecimalMatrix matrix = new BigDecimalMatrix(data.length, 1, SCALE);
		for (int i=0;i<data.length;i++) {
			matrix.setElement(i+1, 1, new BigDecimal(data[i]));
		}
		return matrix;
	}
	
	public static BigDecimalMatrix createColumnRealMatrix(double[] data) {
		BigDecimalMatrix matrix = new BigDecimalMatrix(data.length, 1, SCALE);
		for (int i=0;i<data.length;i++) {
			matrix.setElement(i+1, 1, new BigDecimal(data[i]));
		}
		return matrix;
	}
}
