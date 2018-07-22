package com.nn.dnn;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import com.nn.activation.Tanh;
import com.nn.layer.FullyConnected;
import com.nn.layer.Layer;
import com.nn.layer.OutputLayer;

public class NeuralNetwork {
	private int batchSize = 1;
	
	public NeuralNetwork() {}
	
	public NeuralNetwork(int batchSize) {
		this.batchSize = batchSize;
	}
	
	FullyConnected root = null;
	
	public void train(double[][] inputs, double[][] outputs) {
		int nodes = 8;
		root = new FullyConnected(2, nodes, batchSize, Tanh.INSTANCE);
		Layer layer2 = new OutputLayer(nodes, 1, batchSize, Tanh.INSTANCE);
		root.add(layer2);
		
		double avgLoss = 0;
		long epoch = inputs.length/batchSize;
		
		RealMatrix expected = null;
		RealMatrix input = null;
		for (int i=0;i<epoch;i++) {
			RealMatrix[] losses = new RealMatrix[batchSize];
			for (int j=0;j<batchSize;j++) {
				int index = (int) (batchSize*i+j);
				expected = MatrixUtils.createColumnRealMatrix(outputs[index]);
				input = MatrixUtils.createColumnRealMatrix(inputs[index]);
				losses[j] = root.forward(input, expected);
				root.backward(input, losses[j]);
				avgLoss = (i*avgLoss + losses[j].getEntry(0, 0))/(i+1);
				//System.out.println(loss.getColumn(0)[0]);
				System.out.println(avgLoss);
			}
			/*RealMatrix avgBatchLoss = meanSquaredLoss(losses);
			for (int j=0;j<batchSize;j++) {
				int index = (int) (batchSize*i+j);
				input = MatrixUtils.createColumnRealMatrix(inputs[index]);
				//root.backward(input, avgBatchLoss);
			}*/
		}
	}
	
	public double[] predict(double[] input) {
		return root.predict(MatrixUtils.createColumnRealMatrix(input)).getColumn(0);
	}
	
	public RealMatrix avgLoss(RealMatrix[] losses) {
		double[] res = new double[losses[0].getRowDimension()];
		for (int i=0;i<losses.length;i++) {
			for (int j=0;j<losses[i].getRowDimension();j++) {
				res[j] +=  losses[i].getEntry(j, 0);
			}
		}
		for (int i=0;i<losses[0].getRowDimension();i++) {
			res[i] = res[i]/losses.length;
		}
		return MatrixUtils.createColumnRealMatrix(res);
	}
	
	public RealMatrix meanSquaredLoss(RealMatrix[] losses) {
		double[] res = new double[losses[0].getRowDimension()];
		for (int i=0;i<losses.length;i++) {
			for (int j=0;j<losses[i].getRowDimension();j++) {
				double loss = losses[i].getEntry(j, 0); 
				res[j] +=  loss*loss;
			}
		}
		for (int i=0;i<losses[0].getRowDimension();i++) {
			res[i] = res[i]/losses.length;
		}
		return MatrixUtils.createColumnRealMatrix(res);
	}
}
