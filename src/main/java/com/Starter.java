package com;

import java.util.Arrays;
import java.util.Random;

public class Starter {


	public static void main(String[] args) {
		double[][] inputs = new double[][]{
			{1, 1},
			{0, 0},
			{1, 0},
			{0, 1}
		};
		double[][] outputs = new double[][]{
			{0},
			{0},
			{1},
			{1}
		};
		NeuralNetwork starter = new NeuralNetwork(1);
		
		Random random = new Random();
		for (int i=0;i<100000;i++) {
			//int index = 1;
			int index = random.nextInt(4);
			starter.train(inputs[index], outputs[index]);	
		}
		
		
		System.out.println(Arrays.toString(starter.getResults(new double[]{1, 1})));
		System.out.println(Arrays.toString(starter.getResults(new double[]{0, 0})));
		System.out.println(Arrays.toString(starter.getResults(new double[]{1, 0})));
		System.out.println(Arrays.toString(starter.getResults(new double[]{0, 1})));
	}

}
