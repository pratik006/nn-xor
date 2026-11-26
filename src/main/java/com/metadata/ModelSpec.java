package com.metadata;


import com.fasterxml.jackson.annotation.JsonInclude;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class ModelSpec {

    private int specVersion = 1;

    private String name = "unnamed-model";
    private String description = "";

    private Architecture architecture = new Architecture();
    private Training training = new Training();
    private Inference inference = new Inference();

    // ---------------- ARCHITECTURE ----------------

    public static class Architecture {

        private NetworkType type = NetworkType.MLP;

        private int inputSize;
        private int outputSize;

        private int[] hiddenLayers = {8};

        private Activation activation;
        private Activation outputActivation;

        private Initialization initialization = Initialization.XAVIER;
        private boolean useBias = true;

        // Getters and setters
        public NetworkType getType() { return type; }
        public void setType(NetworkType type) { this.type = type; }

        public int getInputSize() { return inputSize; }
        public void setInputSize(int inputSize) { this.inputSize = inputSize; }

        public int getOutputSize() { return outputSize; }
        public void setOutputSize(int outputSize) { this.outputSize = outputSize; }

        public int[] getHiddenLayers() { return hiddenLayers; }
        public void setHiddenLayers(int[] hiddenLayers) { this.hiddenLayers = hiddenLayers; }

        public Activation getActivation() { return activation; }
        public void setActivation(Activation activation) { this.activation = activation; }

        public Activation getOutputActivation() { return outputActivation; }
        public void setOutputActivation(Activation outputActivation) {
            this.outputActivation = outputActivation;
        }

        public Initialization getInitialization() { return initialization; }
        public void setInitialization(Initialization initialization) {
            this.initialization = initialization;
        }

        public boolean isUseBias() { return useBias; }
        public void setUseBias(boolean useBias) { this.useBias = useBias; }
    }

    // ---------------- TRAINING ----------------

    public static class Training {

        private Optimizer optimizer = Optimizer.SGD;
        private LossFunction lossFunction = LossFunction.MSE;

        private double learningRate = 0.001;
        private int epochs = 1000;
        private int batchSize = 32;

        private double momentum = 0.0;
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double epsilon = 1e-8;

        private double weightDecay = 0.0;
        private double dropout = 0.0;

        private boolean earlyStopping = false;
        private int patience = 10;

        // Zero means disabled
        private double gradientClipping = 0.0;

        private long randomSeed = 42;

        public Optimizer getOptimizer() { return optimizer; }
        public void setOptimizer(Optimizer optimizer) { this.optimizer = optimizer; }

        public LossFunction getLossFunction() { return lossFunction; }
        public void setLossFunction(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
        }

        public double getLearningRate() { return learningRate; }
        public void setLearningRate(double learningRate) { this.learningRate = learningRate; }

        public int getEpochs() { return epochs; }
        public void setEpochs(int epochs) { this.epochs = epochs; }

        public int getBatchSize() { return batchSize; }
        public void setBatchSize(int batchSize) { this.batchSize = batchSize; }

        public double getMomentum() { return momentum; }
        public void setMomentum(double momentum) { this.momentum = momentum; }

        public double getBeta1() { return beta1; }
        public void setBeta1(double beta1) { this.beta1 = beta1; }

        public double getBeta2() { return beta2; }
        public void setBeta2(double beta2) { this.beta2 = beta2; }

        public double getEpsilon() { return epsilon; }
        public void setEpsilon(double epsilon) { this.epsilon = epsilon; }

        public double getWeightDecay() { return weightDecay; }
        public void setWeightDecay(double weightDecay) { this.weightDecay = weightDecay; }

        public double getDropout() { return dropout; }
        public void setDropout(double dropout) { this.dropout = dropout; }

        public boolean isEarlyStopping() { return earlyStopping; }
        public void setEarlyStopping(boolean earlyStopping) {
            this.earlyStopping = earlyStopping;
        }

        public int getPatience() { return patience; }
        public void setPatience(int patience) { this.patience = patience; }

        public double getGradientClipping() { return gradientClipping; }
        public void setGradientClipping(double gradientClipping) {
            this.gradientClipping = gradientClipping;
        }

        public long getRandomSeed() { return randomSeed; }
        public void setRandomSeed(long randomSeed) { this.randomSeed = randomSeed; }
    }

    // ---------------- INFERENCE ----------------

    public static class Inference {

        private double temperature = 1.0;
        private double threshold = 0.5;

        private OutputMode outputMode = OutputMode.RAW;
        private int topK = 1;

        public double getTemperature() { return temperature; }
        public void setTemperature(double temperature) { this.temperature = temperature; }

        public double getThreshold() { return threshold; }
        public void setThreshold(double threshold) { this.threshold = threshold; }

        public OutputMode getOutputMode() { return outputMode; }
        public void setOutputMode(OutputMode outputMode) { this.outputMode = outputMode; }

        public int getTopK() { return topK; }
        public void setTopK(int topK) { this.topK = topK; }
    }

    // ---------------- ENUMS ----------------

    public enum NetworkType {
        MLP
    }

    public enum Activation {
        TANH, SIGMOID, RELU, LEAKY_RELU, LINEAR, SOFTMAX
    }

    public enum Initialization {
        XAVIER, HE, RANDOM, ZERO
    }

    public enum Optimizer {
        SGD, SGD_MOMENTUM, ADAM, RMSPROP
    }

    public enum LossFunction {
        MSE, BINARY_CROSS_ENTROPY, CATEGORICAL_CROSS_ENTROPY
    }

    public enum OutputMode {
        RAW, PROBABILITY, CLASSIFICATION
    }

    // ---------------- ROOT GETTERS / SETTERS ----------------

    public int getSpecVersion() { return specVersion; }
    public void setSpecVersion(int specVersion) { this.specVersion = specVersion; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public Architecture getArchitecture() { return architecture; }
    public void setArchitecture(Architecture architecture) {
        this.architecture = architecture;
    }

    public Training getTraining() { return training; }
    public void setTraining(Training training) { this.training = training; }

    public Inference getInference() { return inference; }
    public void setInference(Inference inference) { this.inference = inference; }
}
