package com;

public class Starter {


	public static void main(String[] args) {
		double[][] inputs = new double[][]{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		};
		double[][] outputs = new double[][]{
			{0},
			{1},
			{1},
			{0}
		};
		NeuralNetwork nn = new NeuralNetwork();
		nn.train(inputs, outputs);
	}

}
