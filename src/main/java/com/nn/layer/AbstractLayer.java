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
	protected RealMatrix bias;
	protected RealMatrix result;
	protected Layer next;	
	protected List<Activation> activations = new ArrayList<>();
	
	public AbstractLayer(int inputCount, int nodes) {
		this.inputCount = inputCount;
		this.nodes = nodes;
		initWeights();
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
	
	protected RealMatrix backwardUpdate(RealMatrix errorGradient, RealMatrix prevResult) {
		RealMatrix weightErrors = errorGradient.multiply(prevResult.transpose()).scalarMultiply((-1)*learningRate);
		RealMatrix biasErrors = errorGradient.scalarMultiply((-1)*learningRate);
		this.weights = this.weights.add(weightErrors);
		this.bias = this.bias.add(biasErrors);		
		return weights.transpose().multiply(errorGradient);
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
