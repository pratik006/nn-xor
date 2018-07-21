package com.nn.layer;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Activation;

public class FullyConnected {
	private int nodes;
	private int inputCount;
	double learningRate = 0.1d;
	
	double avgErrorRate = 0;
	long iterations;
	
	private RealMatrix weights;
	private RealMatrix bias;
	private FullyConnected next;
	private List<Activation> activations = new ArrayList<>();
	private RealMatrix result;
	
	public FullyConnected(int inputCount, int nodes) {
		this.inputCount = inputCount;
		this.nodes = nodes;
		initWeights();
	}
	
	public FullyConnected(int inputCount, int nodes, Activation activation) {
		this(inputCount, nodes);
		this.activations.add(activation);
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
	
	public void add(FullyConnected layer) {
		this.next = layer;
	}
	
	public RealMatrix forward(RealMatrix in, RealMatrix target) {
		//System.out.println(this.hashCode()+" -- "+this.weights.getRowDimension()+", "+this.weights.getColumnDimension());
		this.result = weights.multiply(in).add(bias);
		for (Activation activation : activations) {
			this.result = activation.apply(this.result);	
		}
		
		if (next != null) {
			return next.forward(result, target);
		}
		
		return result.subtract(target);
	}
	
	public RealMatrix backward(RealMatrix prevResult, RealMatrix loss) {
		RealMatrix derivative = activations.get(0).derivative(result);
		RealMatrix errorGradient = null;
		
		if (next != null) {
			RealMatrix prevErrorGradient = next.backward(this.result, loss);
			errorGradient = scalarMultiply(derivative, next.weights.transpose().multiply(prevErrorGradient));
		} else {
			errorGradient = scalarMultiply(derivative, loss);
		}
		
		RealMatrix weightErrors = errorGradient.multiply(prevResult.transpose()).scalarMultiply((-1)*learningRate);
		RealMatrix biasErrors = errorGradient.scalarMultiply((-1)*learningRate);
		this.weights = this.weights.add(weightErrors);
		this.bias = this.bias.add(biasErrors);
		return errorGradient;
	}
	
	public RealMatrix predict(RealMatrix in) {
		RealMatrix result = null;
		for (Activation activation : activations) {
			result = activation.apply(weights.multiply(in).add(bias));	
		}
		
		if (next == null) {
			return result;
		}
		
		return next.predict(result);	
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
