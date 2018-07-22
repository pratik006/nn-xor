package com.mnist;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class MnistReader {
	
	InputStream imageIs;
	InputStream labelIs;
	
	int magicNumber;
	int nImages;
	int nRows;
	int nCols;
	int[][] images;
	int[] labels;
	
	public MnistReader(InputStream imageIs, InputStream labelIs) {
		this.imageIs = imageIs;
		this.labelIs = labelIs;
	}
	
	public void read() throws IOException {
		byte[] b = new byte[16];
		imageIs.read(b);
		this.magicNumber = ByteBuffer.wrap(b, 0, 4).getInt();
		this.nImages = ByteBuffer.wrap(b, 4, 4).getInt();
		this.nRows = ByteBuffer.wrap(b, 8, 4).getInt();
		this.nCols = ByteBuffer.wrap(b, 12, 4).getInt();
		
		images = new int[this.nImages][this.nRows*this.nCols];
		b = new byte[this.nRows*this.nCols*this.nImages];
		int rLen = imageIs.read(b);
		assert(rLen == 60000);
		int ctr = 0;
		for (int i=0;i<nImages;i++) {
			for (int j=0;j<nRows*nCols;j++) {
				images[i][j] = (int)b[ctr++];
			}
		}
		imageIs.close();
		
		
		b = new byte[8];
		rLen = labelIs.read(b);
		b = new byte[nImages];
		rLen = labelIs.read(b);
		labels = new int[nImages];
		for (int i=0;i<nImages;i++) {
			labels[i] = (int)b[i];
		}
		
		labelIs.close();
	}
	
	public int[][] getImages() {
		return images;
	}
	
	public int[] getLabels() {
		return labels;
	}
	
	public static void main(String[] args) throws IOException {
		String baseurl = "/home/pratik/git/nn-demos/mnist/digitrecog/mnist/";
		MnistReader reader = new MnistReader(new FileInputStream(baseurl+"train-images-idx3-ubyte"), new FileInputStream(baseurl+"train-labels-idx1-ubyte"));
		reader.read();
		System.out.println(reader.getImages().length);
	}
}
