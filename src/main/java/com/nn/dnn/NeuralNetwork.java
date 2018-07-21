package com.nn.dnn;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Sigmoid;
import com.nn.activation.Tanh;
import com.nn.layer.FullyConnected;

public class NeuralNetwork {
	
	public void train(double[][] inputs, double[][] outputs) {
		int nodes = 8;
		FullyConnected root = new FullyConnected(2, nodes, Tanh.INSTANCE);
		//FullyConnected layer1 = new FullyConnected(nodes, nodes, Sigmoid.INSTANCE);
		FullyConnected layer2 = new FullyConnected(nodes, 1, Tanh.INSTANCE);
		root.add(layer2);
		//layer1.add(layer2);
		
		double avgLoss = 0;
		for (int k=0;k<10000;k++) {
			int index = (int) ((Math.random()*10) % 4);
			RealMatrix expected = MatrixUtils.createColumnRealMatrix(outputs[index]);
			RealMatrix input = MatrixUtils.createColumnRealMatrix(inputs[index]);
			
			RealMatrix loss = root.forward(input, expected);
			avgLoss = (k*avgLoss + loss.getEntry(0, 0))/(k+1);
			//System.out.println(input+"   "+loss+"    "+avgLoss);
			root.backward(input, loss);			
		}
		
		System.out.println("0 Result "+root.predict(MatrixUtils.createColumnRealMatrix(new double[] {1,1})));
		System.out.println("1 Result "+root.predict(MatrixUtils.createColumnRealMatrix(new double[] {1,0})));
		System.out.println("1 Result "+root.predict(MatrixUtils.createColumnRealMatrix(new double[] {0,1})));
		System.out.println("0 Result "+root.predict(MatrixUtils.createColumnRealMatrix(new double[] {0,0})));
	}
}
