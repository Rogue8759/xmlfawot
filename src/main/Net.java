package main;

import java.util.Random;

import math.Linear;

public class Net {
	private static Linear vec;
	private Random random;
	private final int inputSize;
	private final int numLayers;
	private float[][] neuronStates;
	private float[][][] synStates;
	private float[][] biasStates;
	private int[] layersizes;
	private int penultLayer;
	private int numSynapses;
	private int numBiases;
	private int lastLayer;
	private float softmaxDenom; // The denominator used to calculate softmax in output layer
	private	float momentum;
	private boolean softmaxEnabled=false;
	private float dropoutChance;
	private static final float LOG_OFFSET = (float) Math.exp(-100.0); // Prevent log problems, arbitrary
	private static int epochs;
	private static int epoch;
	private float sumWeightedStates; // The sum of the weighted states of prev. neurons by synapses
	public float[] outputUnactivated;

	// ADAM initial variables
	private float[] mirror;
	private float[] velocity;
	private float[] mhat;
	private float[] vhat;
	public static float b1=0.9f;
	public static float b2=0.999f;

	//RMSPROP
	private float[] gradSquared;
	private int gradientLength;
	private float alpha;
	public static float e = (float) Math.E;	
	
	public Net copy() {
		Net copy = new Net(layersizes);
		copy.setLearningRate(alpha);
		copy.setLayerStates(neuronStates);
		copy.setSynapses(synStates);
		copy.setBiases(biasStates);
		
		return copy;
	}
	
	public static float activation(float x) { return vec.sigmoid(x); }
	public static float derivative(float x) { return vec.sigmoidDv(x); }
	public void setDropoutChance(float dropoutChance) { this.dropoutChance = dropoutChance; }
	public void setLayerStates(float[][] layerstates) { neuronStates = layerstates; }
	public void setSynapses(float[][][] synapses) { synStates = synapses; }
	public void setBiases(float[][] biases) { biasStates = biases; }
	public float getLearningRate() { return alpha; }

	public float synapseGrad(float ex, float b, float prvState) {
		/*
		 * ex: expected value
		 * b: bias of current neuron
		 * prvState: state of previous layer's neuron
		 */
		return  (activation(sumWeightedStates + b) - ex) 
				* (derivative(sumWeightedStates + b) * prvState);
	}
	
	public float biasGrad(float ex, float b) {
		return (activation(sumWeightedStates + b) - ex) 
				* derivative(sumWeightedStates + b);
	}
	
	public float neuronGrad(float ex, float b, float corrWeight) {
		return  (activation(sumWeightedStates + b) - ex) 
				* (derivative(sumWeightedStates + b) * corrWeight);
	}
	
	public double BCEderivMultilabel(float z, float ex) {
		if (Math.round(ex) == 1) {
			return activation(z) * Math.exp(z);
		} else {
			return Math.pow(Math.exp(z),2.0f) * activation(z);
		}
	}
	
	public double BCEdvMLneuron(float z, float ex, float corrWeight) {
		return BCEderivMultilabel(z, ex) * corrWeight;
	}
	
	public double BCEdvMLSyn(float z, float ex, float prevState) {
		return BCEderivMultilabel(z, ex) * prevState;
	}
	
	public double BCEdvMLbias(float z, float ex) {
		return BCEderivMultilabel(z, ex) * 1.0f;
	}
	
	public double softmax(float x) {
		return Math.exp(x) / softmaxDenom;
	}
	
	public double softmaxDv(float x) {
		return softmax(x) - Math.pow(softmax(x), 2.0f);
	}
	
	public void insertInput(float[] inputData) { neuronStates[0] = inputData;}

	public void setSoftmax() {
		softmaxEnabled = true;
	}
	public void setSoftmax(boolean enableSoftmax) {
		softmaxEnabled = enableSoftmax;
	}

	public boolean getSoftmax() {
		return softmaxEnabled;
	}	
	
	public void setEpochs(int eps) {
		epochs = eps;
	}

	public float[] getOutputLayer() {
		return neuronStates[lastLayer];
	}

	public float[] backprop(float[] output) {
		float[] gradient = vec.zeroVector(gradientLength);

		int backpropStart;
		int biasInsPointer = numSynapses;
		int synapseInsPointer = 0;
		int layersize = 0;
		int previousLayersize = 0;
		float[] outputBuffer;
		
		// If we are doing CCE, start doing residuals squared backprop at the penultimate layer
		backpropStart = (softmaxEnabled ? -2 : -1);
		
		// If softmax is enabled, do cross-entropy
		if (softmaxEnabled) {
			layersize = layersizes[lastLayer];
			previousLayersize = layersizes[penultLayer];
			outputBuffer = vec.zeroVector(previousLayersize);
			
			for (int n = 0; n<layersize;n++) {
				for (int prv_n = 0; prv_n<layersizes[penultLayer];prv_n++) {
					gradient[biasInsPointer] += BCEdvMLbias(outputUnactivated[n], output[n]);
					gradient[synapseInsPointer] += BCEdvMLSyn(outputUnactivated[n], output[n], neuronStates[penultLayer][prv_n]);
					outputBuffer[prv_n] += BCEdvMLneuron(outputUnactivated[n], output[n], synStates[penultLayer][prv_n][n]);
					synapseInsPointer++;
				}
				biasInsPointer++;
			}	
			output = outputBuffer;	
		}
		
		// Begin hidden backprop
		for (int l = numLayers+backpropStart;l>0;l--) {
			layersize = layersizes[l];
			previousLayersize = layersizes[l-1];
			outputBuffer = vec.zeroVector(previousLayersize);
			
			for (int n = 0; n<layersize;n++) {
				sumWeightedStates = 0;
				
				for (int prn = 0; prn<layersizes[l-1];prn++) {
					sumWeightedStates += neuronStates[l-1][prn] * synStates[l-1][prn][n];
				}

				
				for (int prv_n = 0; prv_n<layersizes[l-1];prv_n++) {	
					gradient[synapseInsPointer] += synapseGrad(output[n], biasStates[l][n], neuronStates[l-1][prv_n]);
					gradient[biasInsPointer] += biasGrad(output[n], biasStates[l][n]);
					outputBuffer[prv_n] += neuronGrad(output[n], biasStates[l][n], synStates[l-1][prv_n][n]);
					
					synapseInsPointer++;
				}
				biasInsPointer++;
			}	
			output = outputBuffer;	
		}	
		return vec.multiply(gradient, -alpha);
	}

	public void changeNetParams(float[] gradient) {
		int biasInsPointer = this.numSynapses;
		int synapseInsPointer = 0;
		int layerSize = 0;

		for (int layer = numLayers - 1; layer > 0; layer--) {
			layerSize = layersizes[layer];
			
			for (int n = 0; n < layerSize; n++) {				
				for (int prv_n = 0; prv_n<layersizes[layer - 1]; prv_n++) {	
					synStates[layer-1][prv_n][n] += gradient[synapseInsPointer];
					biasStates[layer][n] += gradient[biasInsPointer];
					
					synapseInsPointer++;
				}
				biasInsPointer++;
			}	
		}	
	}
	
	public void trainMomentum(float[][] inputData, float[][] outputData) {
		int samples=inputData.length;

		float[][] gradients = new float[samples][gradientLength]; // Gradients buffer
		float[] epochGrad = new float[gradientLength];
		float[] prevGrad;
		float[] acceleration;
		
		// Momentum training scheme
		velocity = vec.zeroVector(gradientLength);

		for (int e=0;e<epochs;e++)
			for (int datum=0;datum<samples;datum++) {
				insertInput(inputData[datum]);
				propagate();
				gradients[datum] = backprop(outputData[datum]);
				
			}
		
			prevGrad = epochGrad;
			epochGrad = vec.averageVecs(gradients);
			acceleration = vec.subtract(epochGrad, prevGrad);
			velocity = vec.multiply(velocity, momentum);
			velocity = vec.add(velocity, vec.multiply(acceleration,1-momentum));
			
			changeNetParams(velocity);
			epoch++;
	}
	
	public void trainBasic(float[][] inputData, float[][] outputData) {
		int samples=inputData.length;

		float[][] gradients = new float[samples][gradientLength]; // Gradients buffer
		float[] epochGrad = new float[gradientLength];
		
		// Normal/regular gradient descent
		for (int e=0;e<epochs;e++)
			for (int datum=0;datum<samples;datum++) {
				insertInput(inputData[datum]);
				propagate();
				gradients[datum] = backprop(outputData[datum]);
				
			}
			epochGrad = vec.averageVecs(gradients);
			changeNetParams(epochGrad);
			epoch++;
	}
	
	
	public void trainRmsprop(float[][] inputData, float[][] outputData) {
		int samples=inputData.length;
		float[][] gradients = new float[samples][gradientLength]; // Gradients buffer
		float[] epochGrad = new float[gradientLength];
		
		// RMSProp training scheme
		for (int e=0;e<epochs;e++)
			for (int datum=0;datum<samples;datum++) {
				insertInput(inputData[datum]);
				propagate();
				gradients[datum] = backprop(outputData[datum]);
			}

			epochGrad = vec.averageVecs(gradients);
			gradSquared = vec.add(vec.multiply(gradSquared, momentum), vec.multiply(vec.square(epochGrad), 1-momentum));
			changeNetParams(vec.multiply(epochGrad, 1/(vec.scalar_sqrt(gradSquared)+0.0000000000001f)));
	}
	
	public void trainAdam(float[][] inputData, float[][] outputData) {
		int samples=inputData.length;

		float[][] gradients = new float[samples][gradientLength]; // Gradients buffer
		float[] epochGrad = new float[gradientLength];

		for (int e=0;e<epochs;e++)
			for (int datum=0;datum<samples;datum++) {
				insertInput(inputData[datum]);
				propagate();
				gradients[datum] = backprop(outputData[datum]);				
			}

			epochGrad = vec.averageVecs(gradients);
			mirror = vec.add(vec.multiply(mirror, b1), vec.multiply(epochGrad, 1-b1));
			velocity = vec.add(vec.multiply(velocity, b2), vec.multiply(vec.square(epochGrad), 1-b2));
			mhat = vec.multiply(mirror, (float) (1.0f/(1.0f-Math.pow(b1, epoch+1))));
			vhat = vec.multiply(velocity, (float) (1.0f/(1.0f-Math.pow(b2, epoch+1))));
			
			changeNetParams(vec.multiply(mhat, 1/(vec.scalar_sqrt(vhat)+0.00000000000001f)));
			epoch++;
	}
	
	public void train(float[][] inputData, float[][] outputData, String optimizer) {
		if (optimizer.equalsIgnoreCase("momentum")) {
			trainMomentum(inputData, outputData);
		} else if (optimizer.equalsIgnoreCase("rmsprop")) {
			trainRmsprop(inputData, outputData);
		} else if (optimizer.equalsIgnoreCase("adam")) {
			trainAdam(inputData, outputData);
		} else if (optimizer.equalsIgnoreCase("normal")) {
			trainBasic(inputData, outputData);
		}
		
			
	}
	
	public void setLearningRate(float rate) {
		alpha = rate;
	}

	public double get_loss(float[][] testInput, float[][] testOutput) {
		int testAmount = testInput.length;
		int outputLayerSize = testOutput[0].length;
		float lossTotal = 0.0f;
		float[] obsOutput;

		if (softmaxEnabled) {
			for (int i=0;i<testAmount;i++) { 
				insertInput(testInput[i]);
				propagate();
				obsOutput = getOutputLayer();

				for (int j=0;j<outputLayerSize;j++) {
					// Binary cross-entropy loss function
					lossTotal -= (testOutput[i][j]) * Math.log(obsOutput[j] + LOG_OFFSET) + (1.0f - testOutput[i][j]) * Math.log(1.0f - obsOutput[j]+LOG_OFFSET);
				}
				
			}
		} else {
			for (int i=0;i<testAmount;i++) { 
				insertInput(testInput[i]);
				propagate();
				obsOutput = getOutputLayer();
				
				for (int j=0;j<outputLayerSize;j++) {
					lossTotal += Math.pow((testOutput[i][j] - obsOutput[j]), 2);
				}
			}
		}
		
		return Math.pow(lossTotal, 0.5f);
		
		
		
	}
	
	public float[] getLayer(int layer) {
		return neuronStates[layer];
	}
	
	public void propagate() {
		int prevLayersize;
		int layersize;

		for (int i =1;i<numLayers;i++) {
			prevLayersize = layersizes[i-1];
			layersize = layersizes[i];

			for (int n=0;n<layersize;n++) {
				neuronStates[i][n] = 0;
				softmaxDenom = 0.0f;
				
				// Calculate weighted sum of states and add bias, run through activation
				if (Math.random() > 1.0f - dropoutChance) {
					neuronStates[i][n] = 0.0f;
				} else {
					for (int prv_n=0;prv_n<prevLayersize;prv_n++) {
						neuronStates[i][n] += neuronStates[i-1][prv_n] * synStates[i-1][prv_n][n];
					}
					if (i==(numLayers-1)) {
						outputUnactivated[n] = neuronStates[i][n] + biasStates[i][n];
					}

					softmaxDenom += Math.exp(neuronStates[i][n]);
					neuronStates[i][n] = activation(neuronStates[i][n] + biasStates[i][n]);
				}
				
			}
		}
	}
	
	
	public Net(int[] theLayersizes) {
		numLayers = theLayersizes.length;	// The amount of layers
		lastLayer = numLayers-1; 		// The index of the last layer
		penultLayer = numLayers-2;
		inputSize = theLayersizes[0]; 	// Size of input layer
		layersizes = theLayersizes;
		gradientLength = 0;

		vec = new Linear();
		random = new Random();
		
		// Declare all of the data layers of the network
		neuronStates = new float[numLayers][];
		biasStates = new float[numLayers][];
		synStates = new float[numLayers][][];
		outputUnactivated = new float[layersizes[numLayers-1]];
		
		// Initialize all the neuron state data layers
		for (int i = 0; i<numLayers; i++) {
			neuronStates[i] = new float[layersizes[i]];
			biasStates[i] = new float[layersizes[i]];
			numBiases += layersizes[i];

			// Set the network to random gaussian noise
			for (int j=0; j<layersizes[i];j++) {
				neuronStates[i][j] = (float) (random.nextGaussian() * 2.0f);
			}
		}
		
		// Initialize all the synapse weights to zero
		for (int i = 0; i<lastLayer; i++) {
			synStates[i] = new float[layersizes[i]][layersizes[i+1]];
			numSynapses += layersizes[i] * layersizes[i+1];
		}

		
		gradientLength = numSynapses + numBiases;
		gradientLength -= inputSize; // We aren't updating the first input layer's biases
		
		// Initialize all of the gradient buffers for various optimizers
		mirror = new float[gradientLength];
		velocity  = new float[gradientLength];
		mhat = new float[gradientLength];
		vhat = new float[gradientLength];
		gradSquared = vec.zeroVector(gradientLength);
		
	}
	
}
