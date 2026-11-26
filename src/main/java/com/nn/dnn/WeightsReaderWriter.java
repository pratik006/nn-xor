
package com.nn.dnn;

import com.nn.layer.Layer;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.*;
import java.nio.file.*;
import java.util.List;

public class WeightsReaderWriter {
    private static final Path WEIGHTS_FILE = Paths.get("data", "models");
    private static final int MAGIC = 0x4E4E3031; // NN01
    private static final int VERSION = 1;

    public static void save(NeuralNetwork network, String modelName) throws IOException {
        Path filePath = WEIGHTS_FILE.resolve(modelName + ".bin");
        Path parent = filePath.getParent();

        if (parent != null) {
            Files.createDirectories(parent);
        }

        Path tempFile = filePath.resolveSibling(
                filePath.getFileName() + ".tmp"
        );

        try (DataOutputStream output = new DataOutputStream(
                new BufferedOutputStream(
                        Files.newOutputStream(tempFile)))) {

            List<Layer> layers = network.getLayers();

            // File header
            output.writeInt(MAGIC);
            output.writeInt(VERSION);
            output.writeInt(layers.size());

            // Save each layer
            for (Layer layer : layers) {

                RealMatrix weights = layer.getWeights();
                RealVector biases = layer.getBiases();

                int rows = weights.getRowDimension();
                int columns = weights.getColumnDimension();

                // Matrix dimensions
                output.writeInt(rows);
                output.writeInt(columns);

                // Matrix values
                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < columns; column++) {
                        output.writeDouble(
                                weights.getEntry(row, column)
                        );
                    }
                }

                // Bias vector
                output.writeInt(biases.getDimension());

                for (int i = 0; i < biases.getDimension(); i++) {
                    output.writeDouble(biases.getEntry(i));
                }
            }
        }

        // Replace existing file only after successful serialization.
        try {
            Files.move(
                    tempFile,
                    filePath,
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE
            );
        } catch (AtomicMoveNotSupportedException exception) {
            Files.move(
                    tempFile,
                    filePath,
                    StandardCopyOption.REPLACE_EXISTING
            );
        }
    }

    public static void load(NeuralNetwork network, String modelName) throws IOException {
        Path filePath = WEIGHTS_FILE.resolve(modelName + ".bin");
        if (!Files.exists(filePath)) {
            throw new FileNotFoundException(
                    "Model file not found: " + filePath
            );
        }

        List<Layer> layers = network.getLayers();

        try (DataInputStream input = new DataInputStream(
                new BufferedInputStream(
                        Files.newInputStream(filePath)))) {

            int magic = input.readInt();
            int version = input.readInt();
            int layerCount = input.readInt();

            if (magic != MAGIC) {
                throw new IOException("Invalid model file");
            }

            if (version != VERSION) {
                throw new IOException(
                        "Unsupported model version: " + version
                );
            }

            if (layerCount != layers.size()) {
                throw new IOException(
                        "Network architecture mismatch"
                );
            }

            for (int i = 0; i < layerCount; i++) {

                Layer layer = layers.get(i);

                int rows = input.readInt();
                int columns = input.readInt();

                RealMatrix weights = layer.getWeights();

                if (weights.getRowDimension() != rows ||
                        weights.getColumnDimension() != columns) {

                    throw new IOException(
                            "Weight dimensions mismatch at layer " + i
                    );
                }

                double[][] weightValues = new double[rows][columns];

                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < columns; column++) {
                        weightValues[row][column] = input.readDouble();
                    }
                }

                int biasCount = input.readInt();

                RealVector biases = layer.getBiases();

                if (biases.getDimension() != biasCount) {
                    throw new IOException(
                            "Bias dimensions mismatch at layer " + i
                    );
                }

                double[] biasValues = new double[biasCount];

                for (int j = 0; j < biasCount; j++) {
                    biasValues[j] = input.readDouble();
                }

                // Restore parameters
                layer.setWeights(
                        org.apache.commons.math3.linear.MatrixUtils
                                .createRealMatrix(weightValues)
                );

                layer.setBiases(
                        org.apache.commons.math3.linear.MatrixUtils
                                .createRealVector(biasValues)
                );
            }

            if (input.read() != -1) {
                throw new IOException("Unexpected data at end of model file");
            }
        }

        System.out.println("loaded model at: " + filePath.getFileName());
    }

    public static boolean exists(String modelName) {
        return Files.exists(WEIGHTS_FILE.resolve(modelName + ".bin"));
    }
}
