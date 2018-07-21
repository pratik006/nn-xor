package com.nn.dnn;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Relu;
import com.nn.activation.Sigmoid;
import com.nn.layer.FullyConnected;

public class NeuralNetwork {
	
	public void train(double[][] inputs, double[][] outputs) {
		FullyConnected root = new FullyConnected(2, 3, Relu.INSTANCE);
		FullyConnected layer1 = new FullyConnected(3, 3, Relu.INSTANCE);
		FullyConnected layer2 = new FullyConnected(3, 1, Sigmoid.INSTANCE);
		root.add(layer1);
		layer1.add(layer2);
		
		
		
		for (int i=0;i<inputs.length;i++) {
			RealMatrix expected = MatrixUtils.createColumnRealMatrix(outputs[i]);
			RealMatrix input = MatrixUtils.createRowRealMatrix(inputs[i]);
			
			RealMatrix loss = root.forward(input, expected);
			root.backward(input, loss);
			System.out.println(loss);
		}
		
	}
}
