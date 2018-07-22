package com.nn.dnn;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Tanh;
import com.nn.layer.FullyConnected;
import com.nn.layer.Layer;
import com.nn.layer.OutputLayer;

public class NeuralNetwork {
	FullyConnected root = null;
	
	public void train(double[][] inputs, double[][] outputs) {
		int nodes = 8;
		root = new FullyConnected(2, nodes, Tanh.INSTANCE);
		Layer layer2 = new OutputLayer(nodes, 1, Tanh.INSTANCE);
		root.add(layer2);
		
		double avgLoss = 0;
		for (int i=0;i<inputs.length;i++) {
			RealMatrix expected = MatrixUtils.createColumnRealMatrix(outputs[i]);
			RealMatrix input = MatrixUtils.createColumnRealMatrix(inputs[i]);
			
			RealMatrix loss = root.forward(input, expected);
			avgLoss = (i*avgLoss + loss.getEntry(0, 0))/(i+1);
			//System.out.println(input+"   "+loss+"    "+avgLoss);
			root.backward(input, loss);			
		}
	}
	
	public double[] predict(double[] input) {
		return root.predict(MatrixUtils.createColumnRealMatrix(input)).getColumn(0);
	}
}
