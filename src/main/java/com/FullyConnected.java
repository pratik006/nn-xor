package com;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Activation;

public class FullyConnected {
	private int layers;
	private int nodes;
	private int inputCount;
	double learningRate = 0.5d;
	RealMatrix[] hiddenMatrix;
	RealMatrix[] weightMatrices;
	RealMatrix weights;
	RealMatrix bias;
	RealMatrix biasMatrices;
	RealMatrix expectedMatrix;
	double avgErrorRate = 0;
	long iterations;
	
	private FullyConnected next;
	private List<Activation> activations = new ArrayList<>();
	private RealMatrix result;
	
	public FullyConnected(int layers, int inputCount, int outputCount) {
		this.layers = layers;
		this.nodes = outputCount;
		this.inputCount = inputCount;
		initWeights();
	}
	
	public FullyConnected(int inputCount, int nodes, Activation activation) {
		this.inputCount = inputCount;
		this.nodes = nodes;
		this.activations.add(activation);
		initWeights3();
	}

	public void initWeights3() {
		double[][] W = new double[inputCount][nodes];
		double[] B = new double[nodes];
		
		for (int i=0;i<inputCount;i++) {
			W[i] = new double[nodes]; 
			for (int j=0;j<nodes;j++) {
				W[i][j] = Math.random();
			}
		}
		this.weights = MatrixUtils.createRealMatrix(W);
		
		for (int i=0;i<nodes;i++) {
			B[i] = Math.random();
		}
		this.bias = MatrixUtils.createColumnRealMatrix(B);
		//System.out.println(this.hashCode()+" -- "+this.weights.getRowDimension()+", "+this.weights.getColumnDimension());
		//System.out.println(B.length);
	}
	
	public void initWeights() {
		double[][][] weights = new double[layers+1][inputCount][];
		
		double[] biases = new double[layers+1];
		
		for (int i=0;i<=layers;i++) {
			int l = (i==layers) ? nodes : inputCount;
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
		RealMatrix loss = forwardPass(1);
		System.out.println("avg error rate: "+avgErrorRate+"   "+Arrays.toString(inputs)+"  "+loss.getColumnMatrix(0));
		calculateCost(layers+1);
		//if (expectedOutputs[0]- res[0] > 0.5)
			
		/*System.out.println("error: "+err+"[ bias1: "+Arrays.toString(biasMatrices[0].getData()[0])+", "+Arrays.toString(biasMatrices[1].getData()[0])+"], [weight1: "
				+Arrays.toString(weightMatrices[0].getData()[0])+Arrays.toString(weightMatrices[0].getData()[1])+"]");*/
	}
	
	public RealMatrix getResults(double[] inputs) {
		this.hiddenMatrix[0] = MatrixUtils.createColumnRealMatrix(inputs);
		return forwardPass(1);
	}
	
	public RealMatrix forwardPass(int layer) {
		if (layer > layers + 1) {
			hiddenMatrix[layer-1] = sigmoid(hiddenMatrix[layer-1]);
			return hiddenMatrix[layer-1].subtract(expectedMatrix);
		}
		RealMatrix inputMatrix = hiddenMatrix[layer-1];
		RealMatrix weightMatrix = weightMatrices[layer-1];
		hiddenMatrix[layer] = weightMatrix.transpose().multiply(inputMatrix);
		hiddenMatrix[layer] = hiddenMatrix[layer].scalarAdd(biasMatrices.getColumn(0)[0]);
		return forwardPass(layer+1);
	}
	
	public void add(FullyConnected layer) {
		this.next = layer;
	}
	
	public RealMatrix forward(RealMatrix in, RealMatrix out) {
		for (Activation activation : activations) {
			this.result = activation.apply(in.multiply(weights).add(bias.transpose()));	
		}
		
		if (next != null) {
			return next.forward(result, out);
		}
		
		/*gradient = dsigmoid(out.subtract(in));
		gradient = out.subtract(in).multiply(this.weights.transpose());
		this.weights.subtract(gradient.scalarMultiply(0.01));
		this.bias.subtract(gradient.scalarMultiply(0.01));*/
		//System.out.println(this.hashCode()+" -- "+this.weights.getRowDimension()+", "+this.weights.getColumnDimension());
		return out.subtract(result);
	}
	
	public RealMatrix backward(RealMatrix prevResult, RealMatrix loss) {
		RealMatrix localGradient = null;
		if (next != null) {
			localGradient = next.backward(this.result, loss);
		}
		
		localGradient = activations.get(0).gradient(localGradient==null ? result : localGradient);
		
		localGradient = scalarMultiply(localGradient, loss).scalarMultiply(learningRate);
		RealMatrix weightErrors = localGradient.multiply(prevResult).transpose();
		
		this.weights.subtract(weightErrors.scalarMultiply(0.01));
		this.bias.subtract(localGradient.scalarMultiply(0.01));
		return localGradient;
	}
	
	public RealMatrix predict(RealMatrix in) {
		//System.out.println(this.hashCode()+" -- "+this.weights.getRowDimension()+", "+this.weights.getColumnDimension());
		RealMatrix result = null;
		for (Activation activation : activations) {
			result = activation.apply(in.multiply(weights).add(bias));	
		}
		
		if (next == null) {
			return result;
		}
		
		return next.predict(result);	
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
		return MatrixUtils.createColumnRealMatrix(Arrays.stream(matrix.getColumn(0)).map(FullyConnected::sigmoid).toArray());
	}
	
	public static RealMatrix dsigmoid(RealMatrix matrix) {
		double[][] data = new double[matrix.getRowDimension()][matrix.getColumnDimension()];
		for (int i=0;i<matrix.getRowDimension();i++) {
			for (int j=0;j<matrix.getColumnDimension();j++) {
				data[i][j] = matrix.getData()[i][j]*(1 - matrix.getData()[i][j]);
			}
		}
		return MatrixUtils.createRealMatrix(data);
		//return MatrixUtils.createColumnRealMatrix(Arrays.stream(matrix.getColumn(0)).map(val->val*(1 - val)).toArray());
	}
	
	public static RealMatrix scalarMultiply(RealMatrix x, RealMatrix y) {
		RealMatrix res = MatrixUtils.createRealMatrix(x.getRowDimension(), x.getColumnDimension());
		for (int i=0;i<x.getRowDimension();i++) {
			for (int j=0;j<x.getColumnDimension();j++) {
				res.setEntry(i, j, x.getData()[i][j]*(y.getData()[i][j]));				
			}
		}
		
		return res;
	}
}
