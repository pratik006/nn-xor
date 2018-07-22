package com.nn.layer;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Activation;

public abstract class AbstractLayer implements Layer {
	private int nodes;
	private int inputCount;
	protected double learningRate = 0.1d;
	
	protected RealMatrix weights;
	protected RealMatrix weightDeltas;
	protected RealMatrix bias;
	protected RealMatrix biasDeltas;
	protected RealMatrix result;
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
		double[][] W = new double[nodes][inputCount];
		double[] B = new double[nodes];
		
		for (int i=0;i<nodes;i++) {
			W[i] = new double[inputCount]; 
			for (int j=0;j<inputCount;j++) {
				W[i][j] = Math.random();
				W[i][j] = W[i][j]*W[i][j]*W[i][j];
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
	
	public void resetDeltas() {
		double[][] dW = new double[nodes][inputCount];
		double[] dB = new double[nodes];
		
		for (int i=0;i<nodes;i++) {
			dW[i] = new double[inputCount]; 
			for (int j=0;j<inputCount;j++) {
				dW[i][j] = 0.0d;
			}
		}
		this.weightDeltas = MatrixUtils.createRealMatrix(dW);
		
		for (int i=0;i<nodes;i++) {
			dB[i] = 0.0d;
		}
		this.biasDeltas = MatrixUtils.createColumnRealMatrix(dB);
	}
	
	public void add(Layer layer) {
		this.next = layer;
	}
	
	public RealMatrix predict(RealMatrix in) {
		RealMatrix result = applyActivation(weights.multiply(in).add(bias));
		return (next == null) ? result : next.predict(result);	
	}
	
	public RealMatrix forward(RealMatrix in, RealMatrix target) {
		iterationNo++;
		//System.out.println(this.hashCode()+" -- "+this.weights.getRowDimension()+", "+this.weights.getColumnDimension());
		result = applyActivation(weights.multiply(in).add(bias));
		return next.forward(result, target);
	}
	
	protected RealMatrix applyActivation(RealMatrix in) {
		RealMatrix result = in;
		for (Activation activation : activations) {
			result = activation.apply(result);	
		}
		return result;
	}
	
	protected RealMatrix applyDerivative(RealMatrix in) {
		RealMatrix result = in;
		for (Activation activation : activations) {
			result = activation.derivative(in, result);	
		}
		return result;
	}
	
	protected RealMatrix backwardUpdate(RealMatrix errorGradient, RealMatrix prevResult) {
		RealMatrix weightErrors = errorGradient.multiply(prevResult.transpose()).scalarMultiply((-1)*learningRate);
		RealMatrix biasErrors = errorGradient.scalarMultiply((-1)*learningRate);
		weightDeltas = weightDeltas.add(weightErrors);		
		biasDeltas = biasDeltas.add(biasErrors);
		return weights.transpose().multiply(errorGradient);
	}
	
	public void updateWeightsBiases() {
		this.weights = this.weights.add(weightDeltas);
		this.bias = this.bias.add(biasDeltas);
		iterationNo = 0;
		resetDeltas();
		if (next != null)
			next.updateWeightsBiases();
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
	
	protected void print(RealMatrix matrix) {
		System.out.println("[");
		for (int i=0;i<matrix.getData().length;i++) {
			for (int j=0;j<matrix.getData()[i].length;j++) {
				System.out.print(matrix.getData()[i][j]+"    ");
			}
			System.out.println();
		}
		System.out.println("]");
	}
}
