package com;

import com.metadata.ModelSpec;
import com.metadata.ModelSpecReader;
import com.nn.dnn.NeuralNetwork;
import com.nn.dnn.NeuralNetworkFactory;
import com.nn.dnn.WeightsReaderWriter;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

public class Starter {
	private static final Logger LOG = Logger.getLogger(Starter.class.getName());

	public static void main(String[] args) throws IOException {
		String modelName = "xor";
		ModelSpecReader modelSpecReader = new ModelSpecReader();
		ModelSpec spec = modelSpecReader.read();

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
		
		int inputSize = 10000000;
		double[][] inputs = new double[inputSize][2];
		double[][] outputs = new double[inputSize][2];
		for (int k=0;k<inputSize;k++) {
			int index = (int) ((Math.random()*10) % 4);
			inputs[k] = inTemplate[index];
			outputs[k] = outTemplate[index];
		}
		
		NeuralNetwork nn = new NeuralNetworkFactory().build(spec);

		final AtomicBoolean end = new AtomicBoolean(false);
		Thread t = new Thread(() -> {
			while (!end.get()) {
				LOG.info("Loss: " + new DecimalFormat("0.00000").format(nn.loss));
				try {
					Thread.sleep(5000);
				} catch (InterruptedException e) {
					throw new RuntimeException(e);
				}
			}
		});
		t.start();

		if (WeightsReaderWriter.exists(modelName)) {
			WeightsReaderWriter.load(nn, modelName);
		} else {
			nn.train(inputs, outputs);
			WeightsReaderWriter.save(nn, modelName);
		}

    	double[] input = new double[] {1,1};
		double probability = nn.predict(input)[0];
		LOG.info("expected 0 Result "+(probability < 0.5 ? 0 : 1));

		input = new double[] {1,0};
		probability = nn.predict(input)[0];
		LOG.info("expected 1 Result "+(probability < 0.5 ? 0 : 1));

		input = new double[] {0,1};
		probability = nn.predict(input)[0];
		LOG.info("expected 1 Result "+(probability < 0.5 ? 0 : 1));

		input = new double[] {0,0};
		probability = nn.predict(input)[0];
		LOG.info("expected 0 Result "+(probability < 0.5 ? 0 : 1));

		end.getAndSet(true);
	}

}
