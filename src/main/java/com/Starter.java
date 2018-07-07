package com;

import java.util.Arrays;
import java.util.Random;

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
		NeuralNetwork nn = new NeuralNetwork(1,4,1);
		
		Random random = new Random();
		StringBuilder sb = new StringBuilder();
		int[] arr = new int[] {3,3,2,3,2,1,1,2,3,1,2,1,3,1,1,2,1,3,1,3,0,3,1,3,3,3,2,3,2,0,2,3,0,1,3,1,1,1,3,2,3,2,3,1,3,2,1,1,3,1,0,3,1,1,3,3,0,2,1,3,3,1,3,3,2,3,0,1,3,3,1,1,3,1,0,1,3,0,0,1,1,1,3,2,3,3,0,2,1,3,1,2,3,3,1,1,1,0,0,3,0,3,3,1,1,0,3,3,1,3,2,3,2,2,3,2,3,0,0,3,1,0,2,3,0,3,0,2,0,2,2,2,0,3,2,2,0,1,3,1,1,0,0,1,2,3,0,3,3,3,3,2,2,0,0,3,3,0,0,0,1,0,2,1,1,3,2,0,2,2,3,3,0,2,2,3,2,0,3,0,3,0,1,0,1,1,1,2,1,3,3,1,3,0,3,2,1,0,1,3,2,0,0,3,1,0,3,0,0,1,1,2,3,2,1,3,2,0,2,2,0,1,0,2,0,0,2,1,2,1,0,0,1,2,3,3,0,2,1,3,0,2,1,1,3,3,1,0,1,2,3,1,1,2,0,2,1,0,0,1,0,2,1,2,1,1,2,3,2,0,2,3,2,2,3,1,1,0,1,1,2,3,3,0,2,2,1,1,2,2,1,1,1,2,0,1,3,1,1,3,0,1,0,2,2,1,2,2,0,2,0,2,2,0,2,0,1,0,2,1,2,1,3,1,1,0,2,1,2,1,3,1,0,2,2,2,0,3,2,2,3,1,3,1,0,3,3,3,0,0,2,2,2,0,0,0,1,3,0,1,1,3,2,3,3,0,2,3,1,0,0,2,1,0,1,1,2,2,3,2,3,2,1,1,3,3,1,1,2,1,1,0,3,1,1,3,0,3,0,1,2,1,3,3,3,3,0,3,2,1,0,2,3,3,2,1,0,0,0,0,0,3,3,2,2,0,1,2,2,3,0,0,2,0,3,0,0,3,3,2,3,1,0,0,1,2,0,3,2,2,0,1,2,2,3,1,3,2,0,1,2,3,3,1,0,2,3,2,0,1,2,1,0,2,1,2,3,2,1,2,1,0,1,0,1,2,1,2,2,0,3,1,1,0,3,0,2,2,3,2};
		for (int i=0;i<10000000;/*arr.length*/i++) {
			//int index = 2;
			//int index = arr[i];
			int index = random.nextInt(4);
			//sb.append(index+",");
			nn.train(inputs[index], outputs[index]);	
		}
		//System.out.println(sb);
		
		System.out.println("0,0 => "+Arrays.toString(nn.predict(new double[]{0, 0})));
		System.out.println("1,1 => "+Arrays.toString(nn.predict(new double[]{1, 1})));
		System.out.println("1,0 => " + Arrays.toString(nn.predict(new double[]{1, 0})));
		System.out.println("0,1 => "+Arrays.toString(nn.predict(new double[]{0, 1})));
	}

}
