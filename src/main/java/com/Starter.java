package com;

import java.util.Arrays;

import com.nn.dnn.NeuralNetwork;

public class Starter {


	public static void main(String[] args) {
		double[][] inTemplate = new double[][]{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		};
		double[][] outTemplate = new double[][]{
			{0},
			{1},
			{1},
			{0}
		};
		
		int inputSize = 8000;
		double[][] inputs = new double[inputSize][2];
		double[][] outputs = new double[inputSize][2];
		for (int k=0;k<inputSize;k++) {
			int index = (int) ((Math.random()*10) % 4);
			inputs[k] = inTemplate[index];
			outputs[k] = outTemplate[index];
		}
		
		NeuralNetwork nn = new NeuralNetwork(20);
		nn.train(inputs, outputs);
		
		System.out.println("0 Result "+Arrays.toString(nn.predict(new double[] {1,1})));
		System.out.println("1 Result "+Arrays.toString(nn.predict(new double[] {1,0})));
		System.out.println("1 Result "+Arrays.toString(nn.predict(new double[] {0,1})));
		System.out.println("0 Result "+Arrays.toString(nn.predict(new double[] {0, 0})));
	}

}
