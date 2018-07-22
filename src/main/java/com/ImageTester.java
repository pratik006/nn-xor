package com;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

import com.mnist.MnistReader;
import com.nn.activation.Relu;
import com.nn.activation.Tanh;
import com.nn.dnn.NeuralNetwork;
import com.nn.layer.FullyConnected;
import com.nn.layer.Layer;
import com.nn.layer.OutputLayer;

public class ImageTester {
	static String baseurl = "/home/pratik/git/nn-demos/mnist/digitrecog/mnist/";
	static String imagesFile = "train-images-idx3-ubyte";
	static String labelsFile = "train-labels-idx1-ubyte";
	
	public static void main(String[] args) throws IOException {
		MnistReader reader = new MnistReader(
			new FileInputStream(baseurl+imagesFile),
			new FileInputStream(baseurl+labelsFile)
		);
		reader.read();
		int[][] images = reader.getImages();
		int[][] labels = new int[images.length][10];
		for (int i=0;i<reader.getLabels().length;i++) {
			labels[i] = new int[10];
			for (int j=0;j<10;j++) {
				labels[i][j] = reader.getLabels()[i] == j ? 1 : 0;
			}
		}
		
		NeuralNetwork nn = new NeuralNetwork(20);
		int nodes = 32;
		Layer root = new FullyConnected(images[0].length, nodes, Relu.INSTANCE);
		Layer layer1 = new FullyConnected(nodes, nodes, Tanh.INSTANCE);
		Layer outputLayer = new OutputLayer(nodes, labels[0].length, Tanh.INSTANCE);
		layer1.add(outputLayer);
		root.add(layer1);
		nn.setRoot(root);
		nn.train(images, labels);
		
		System.out.println("0 Result "+Arrays.toString(nn.predict(new double[] {1,1})));
		System.out.println("1 Result "+Arrays.toString(nn.predict(new double[] {1,0})));
		System.out.println("1 Result "+Arrays.toString(nn.predict(new double[] {0,1})));
		System.out.println("0 Result "+Arrays.toString(nn.predict(new double[] {0, 0})));
	}
}
