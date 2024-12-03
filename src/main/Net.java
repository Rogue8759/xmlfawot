package main;

public class Net {
	private final int inputSize;
	private final int numLayers;
	private float[][] neuronStates;
	private float[][][] synStates;
	private float[][] biasStates;
	private int[] layersizes;
	private int numSynapses;
	private int numBiases;
	private int lastLayer;
	private float softmaxDenominator; // The sum of exponentials of the outputs used to calculate a softmax'ed output
	private final float momentum = 0.90f;
	private boolean softmaxEnabled=false;
	private float dropoutChance = 0.10f;
	private static final float LOG_OFFSET = 0.000000000000000001f; // Prevent log problems
	private static int epochs;
	private static int epoch;
	
	// ADAM SHIT
	private float[] mirror;
	private float[] velocity;
	private float[] mhat;
	private float[] vhat;

	//RMSPROP
	private float[] gradSquared;
	
	
	private int gradientLength;
	private float alpha;
	public static float e = 2.71828182845905f;
	
	
	public Net copy() {
		Net copy = new Net(layersizes);
		copy.setLearningRate(alpha);
		copy.setLayerStates(neuronStates);
		copy.setSynapses(synStates);
		copy.setBiases(biasStates);
		
		return copy;
	}
	
	
	public static float activation(float x) {
		if (x < 0.0f) {
			return 0;
		}
		return x;
	}
	
	
	public static float derivative(float x) {
		if (x < 0.0f) {
			return 0;
		}
		return 1;
	}
	
	
	public void setDropoutChance(float dropoutChance) {
		this.dropoutChance = dropoutChance;
	}
	
	public void setLayerStates(float[][] layerstates) {
		neuronStates = layerstates;
	}
	
	public void setSynapses(float[][][] synapses) {
		synStates = synapses;
	}
	
	public void setBiases(float[][] biases) {
		biasStates = biases;
	}
	
	public float getLearningRate() {
		return alpha;
	}
	
	public static float synapseGrad(float ex, float wstates, float b, float pstate) {
		return 2.0f
				* (activation(wstates + b) - ex) 
				* (derivative(wstates + b) * pstate);
		
	}
	
	public static float biasGrad(float ex, float wstates, float b) {
		return 2.0f
				* (activation(wstates + b) - ex) 
				* derivative(wstates + b);
	}
	
	public static float neuronGrad(float ex, float wstates, float b, float corrWeight) {
		return 2.0f
				* (activation(wstates + b) - ex) 
				* (derivative(wstates + b) * corrWeight);
	}
	
	public static double synapseGradEntropy(float ex, float sumExpos, float wstates, float b, float pstate) {
		/*
		 * Calculates the gradient of a synapse parameter using cross-entropy. This is
		 * the pure, unadulterated mathematical derivative, does not account for learning rate.
		 */
		
		return  synapseGrad(ex, wstates, b, pstate) 
				* entropyDerivative(ex, sumExpos, wstates + b) * pstate;
	}
	
	public static double biasGradEntropy(float ex, float sumExpos, float wstates, float b) {
		/*
		 * Calculates the gradient of a bias parameter using cross-entropy. This is
		 * the pure, unadulterated mathematical derivative, does not account for learning rate.
		 */
		
		return  biasGrad(ex, wstates, b) 
				* entropyDerivative(ex, sumExpos, wstates + b);
	}
	
	public static double neuronGradEntropy(float ex, float sumExpos, float wstates, float b, float corr_weight) {
		/*
		 * Calculates the gradient of a previous neuron state parameter using cross-entropy. This is
		 * the pure, unadulterated mathematical derivative, does not account for learning rate.
		 */
		
		return  neuronGrad(ex, wstates, b, corr_weight) 
				* entropyDerivative(ex, sumExpos, wstates + b) * corr_weight;
	}
	
	public void insertInput(float[] inputData) {
		/*
		 * Insert data into input layer
		 * 
		 */
		neuronStates[0] = inputData; 
	}
	
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
		/*
		 * Return output layer data vector
		 * 
		 */
		return neuronStates[lastLayer];
	}
	
	public static float[] averageGradients(float[][] gradients) {
		/*
		 * Crunches the average gradient from an array of gradients
		 * 
		 */
		
		float[] average = new float[gradients[0].length];
		float nGradients = gradients.length;
		float avgVal;
		
		for (int i=0;i<average.length;i++) {
			 avgVal = 0.0f;
			 for (float[] grad : gradients) {
				 avgVal += grad[i];
			 }
			 avgVal /= nGradients;
			 average[i] = avgVal;
		}
		
		return average;
	}
	
	public static double entropyDerivative(float ex, float sum, float state) {
		/*
		 * The derivative of the composite cross-entropy - softmax function. Pure
		 * derivative, does not account for learning rate 
		 */
		
		// Maybe this formula needs optimization?
		return sum * sum * (ex * sum - Math.exp(state)) * (sum - Math.exp(state));
	}
	
	public static double softmaxDerivative(float sum, float state) {
		/*
		 * The derivative of the softmax function for an input state.
		 * Does not calculate the denominator sum in the softmax equation.
		 */

		return Math.pow(e, state) * (sum - Math.exp(state)) / Math.pow(sum, 2.0f);
	}

	public float[] backprop(float[] output) {
		/*
		 * Big ass function that returns the gradient of the network given an observed training output
		 * 
		 */
		
		float[] gradient = new float[gradientLength];
		
		// Set update gradient to zero
		for (int i = 0; i < gradientLength; i++ ) {
			gradient[i] = 0;
		}

		int backpropStart;
		int sumWeightedStates;
		int sumPrevStates;
		int biasInsPointer = 0;
		int synapseInsPointer = 0;
		int layersize = 0;
		int previousLayersize = 0;
		float[] outputBuffer;
		
		// If we are doing CCE, start doing residuals squared backprop at the penultimate layer
		backpropStart = (softmaxEnabled ? -2 : -1);
		
		// If softmax is enabled, do cross-entropy
		if (softmaxEnabled) {
			layersize = layersizes[lastLayer];
			previousLayersize = layersizes[numLayers-2];
			
			outputBuffer = new float[previousLayersize];
			for (int i =0; i<previousLayersize;i++) {
				outputBuffer[i] = 0;
			}
			
			// Calculate total exponentials of output layer states
			float sumExpos = 0;
			for (int n=0; n<layersize; n++) {
				sumExpos += Math.pow(e, neuronStates[lastLayer][n]);
			}
			
			for (int n = 0; n<layersize;n++) {
				
				sumWeightedStates = 0;
				sumPrevStates = 0;
				
				for (int prn = 0; prn<layersizes[numLayers-2];prn++) {
					sumWeightedStates += neuronStates[numLayers-2][prn] * synStates[numLayers-2][prn][n];
					sumPrevStates += neuronStates[numLayers-2][prn];
				}
				
				// Calculate the cross-entropy of this output layer neuron
				neuronStates[lastLayer][n] = (float) ((1.0f - output[n]) * Math.log(1.0f - neuronStates[lastLayer][n] + LOG_OFFSET));

				
				for (int prv_n = 0; prv_n<layersizes[numLayers-2];prv_n++)
				{	
					gradient[numSynapses + biasInsPointer] += biasGradEntropy(output[n],sumExpos, sumWeightedStates, biasStates[lastLayer][n]);
					gradient[synapseInsPointer] += synapseGradEntropy(output[n],sumExpos, sumWeightedStates, biasStates[lastLayer][n], neuronStates[numLayers-2][prv_n]);
					outputBuffer[prv_n] += neuronGradEntropy(output[n],sumExpos, sumWeightedStates, biasStates[lastLayer][n], synStates[numLayers-2][prv_n][n]);
					synapseInsPointer++;
				}
				biasInsPointer++;
			}	
			output = outputBuffer;	
			
		}
		
		for (int l = numLayers+backpropStart;l>0;l--) {
			layersize = layersizes[l];
			previousLayersize = layersizes[l-1];
			
			outputBuffer = new float[previousLayersize];
			for (int i =0; i<previousLayersize;i++) {
				outputBuffer[i] = 0;
			}
			
			for (int n = 0; n<layersize;n++) {
				sumWeightedStates = 0;
				sumPrevStates = 0;
				
				for (int prn = 0; prn<layersizes[l-1];prn++) {
					sumWeightedStates += neuronStates[l-1][prn] * synStates[l-1][prn][n];
					sumPrevStates += neuronStates[l-1][prn];
				}

				
				for (int prv_n = 0; prv_n<layersizes[l-1];prv_n++)
				{	
					gradient[synapseInsPointer] += synapseGrad(output[n], sumWeightedStates, biasStates[l][n], neuronStates[l-1][prv_n]);
					gradient[numSynapses + biasInsPointer] += biasGrad(output[n], sumWeightedStates, biasStates[l][n]);
					outputBuffer[prv_n] += neuronGrad(output[n], sumWeightedStates, biasStates[l][n], synStates[l-1][prv_n][n]);
					
					synapseInsPointer++;
				}
				biasInsPointer++;
			}	
			output = outputBuffer;	
		}	
		return gradient;
	}
	
	public void changeNetParams(float[] gradient) {
		int biasInsPointer = 0;
		int synapseInsPointer = 0;
		int layerSize = 0;

		for (int layer = numLayers - 1; layer > 0; layer--) {
			layerSize = layersizes[layer];
			
			for (int n = 0; n < layerSize; n++) {				
				for (int prv_n = 0; prv_n<layersizes[layer - 1]; prv_n++) {	
					synStates[layer-1][prv_n][n] -= alpha * gradient[synapseInsPointer];
					biasStates[layer][n] -= alpha * gradient[numSynapses + biasInsPointer];
					
					synapseInsPointer++;
				}
				biasInsPointer++;
			}	
		}	
	}
	
	public static float[] subtract(float[] a, float[] b) {
		/*
		 * Subtract two vectors
		 */
		float[] retrn = new float[a.length];
		
		for (int i = 0; i < a.length; i++) {
			retrn[i] = a[i]-b[i];
		}

		return retrn;

	}
	
	public static float[] add(float[] a, float[] b) {
		/*
		 * Add two vectors
		 */
		float[] retrn = new float[a.length];
		
		for (int i = 0;i<a.length;i++) {
			retrn[i] = a[i]+b[i];
		}

		return retrn;

	}
	
	public static float[] sqrt(float[] x) {
		/*
		 * Takes the square root of a vector element-wise
		 */

		float[] retrn = new float[x.length];
		
		for (int i = 0; i < x.length; i++) {
			retrn[i] = (float) Math.pow(x[i], 0.5f);
		}
		
		return retrn;
		
	}
	
	public static float scalar_sqrt(float[] x) {
		/*
		 * Square root of the sum of vector elements
		 */
		
		float sum = 0;
		
		for (float v : x) {
			sum += v;
		}
		
		return sum/x.length;
	}
	
	public static float[] square(float[] x) {
		/*
		 * Squares a vector's elements element-wise
		 */
		
		float[] retrn = new float[x.length];
		
		for (int i = 0; i < x.length; i++) {
			retrn[i] = (float) Math.pow(x[i], 2.0f);
		}
		
		return retrn;
		
	}
	
	public static float[] multiply(float[] a, float f) {
		/*
		 * Multiply two vectors element-wise
		 */
		
		float[] retrn = new float[a.length];
		
		for (int i = 0; i < a.length; i++) {
			retrn[i] = a[i]*f;
		}

		return retrn;

	}
	
	public static float[] zeroVector(int size) {
		/*
		 * Null vector
		 */

		return new float[size];
	}
	
	public static float[] divide(float n, float[] v) {
		/*
		 * Divide a vector's elements by another vector's elements
		 */
		
		float[] retrn = new float[v.length];
		
		for (int i = 0; i < v.length; i++) {
			retrn[i] = n / v[i];
		}

		return retrn;
	}
	
	public void trainMomentum(float[][] inputData, float[][] outputData) {
		int samples=inputData.length;

		float[][] gradients = new float[samples][gradientLength]; // Gradients buffer
		float[] epochGrad = new float[gradientLength];
		float[] prevGrad;
		float[] acceleration;
		
		// Momentum training scheme
		velocity = new float[gradientLength];

		for (int e=0;e<epochs;e++)
			for (int datum=0;datum<samples;datum++) {
				insertInput(inputData[datum]);
				propagate();
				gradients[datum] = backprop(outputData[datum]);
				
			}
		
			prevGrad = epochGrad;
			epochGrad = averageGradients(gradients);
			acceleration = subtract(epochGrad, prevGrad);
			velocity = multiply(velocity, momentum);
			velocity = add(velocity, multiply(acceleration,1-momentum));
			
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
			epochGrad = averageGradients(gradients);
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

			epochGrad = averageGradients(gradients);
			gradSquared = add(multiply(gradSquared, momentum), multiply(square(epochGrad), 1-momentum));
			changeNetParams(multiply(epochGrad, 1/(scalar_sqrt(gradSquared)+0.0000000000001f)));
	}
	
	public void trainAdam(float[][] inputData, float[][] outputData) {
		int samples=inputData.length;

		float[][] gradients = new float[samples][gradientLength]; // Gradients buffer
		float[] epochGrad = new float[gradientLength];
		
		// ADAM training scheme
		float b1 = 0.9f;
		float b2 = 0.999f;

		for (int e=0;e<epochs;e++)
			for (int datum=0;datum<samples;datum++) {
				insertInput(inputData[datum]);
				propagate();
				gradients[datum] = backprop(outputData[datum]);				
			}

			epochGrad = averageGradients(gradients);
			mirror = add(multiply(mirror, b1), multiply(epochGrad, 1-b1));
			velocity = add(multiply(velocity, b2), multiply(square(epochGrad), 1-b2));
			mhat = multiply(mirror, (float) (1.0f/(1.0f-Math.pow(b1, epoch+1))));
			vhat = multiply(velocity, (float) (1.0f/(1.0f-Math.pow(b2, epoch+1))));
			
			changeNetParams(multiply(mhat, 1/(scalar_sqrt(vhat)+0.00000000000001f)));
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
					lossTotal -= (1.0f - testOutput[i][j]) * Math.log(1.0f - obsOutput[j]+LOG_OFFSET);
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
				
				// Calculate weighted sum of states and add bias, run through activation
				if (Math.random() > 1.0f - dropoutChance) {
					neuronStates[i][n] = 0.0f;
				} else {
					for (int prv_n=0;prv_n<prevLayersize;prv_n++) {
						neuronStates[i][n] += neuronStates[i-1][prv_n] * synStates[i-1][prv_n][n];
					}
					neuronStates[i][n] = activation(neuronStates[i][n] + biasStates[i][n]);
				}
				
			}
		}
		
		// Run the entire output layer through Softmax
		if (softmaxEnabled) {
			softmaxDenominator = 0.0f;
			for (int n=0;n<layersizes[lastLayer];n++) {
				softmaxDenominator += Math.exp(neuronStates[lastLayer][n]);
			}

			for (int n=0;n<layersizes[lastLayer];n++) {
				neuronStates[lastLayer][n] = (float) (Math.exp(neuronStates[lastLayer][n]) / softmaxDenominator);
				
			}
			
			
			
		}
	}
	
	
	public Net(int[] theLayersizes) {
		numLayers = theLayersizes.length;
		lastLayer = numLayers-1; // The index of the last layer
		inputSize = theLayersizes[0];
		layersizes = theLayersizes;
		gradientLength = 0;
		
		// Declare all of the data layers of the network
		neuronStates = new float[numLayers][];
		biasStates = new float[numLayers][];
		synStates = new float[numLayers][][];
		
		for (int i = 0; i<numLayers; i++) {
			neuronStates[i] = new float[layersizes[i]];
			biasStates[i] = new float[layersizes[i]];
			numBiases += layersizes[i];

			// Set the network to random
			for (int j=0; j<layersizes[i];j++) {
				neuronStates[i][j] = (float) (Math.random() -0.5f) * 2.0f;
			}
		}
		
		for (int i = 0; i<lastLayer; i++) {
			synStates[i] = new float[layersizes[i]][layersizes[i+1]];
			numSynapses += layersizes[i] * layersizes[i+1];
		}

		
		gradientLength = numSynapses + numBiases;
		gradientLength -= inputSize; // We aren't updating the first input layer's biases
		
		mirror = new float[gradientLength];
		velocity  = new float[gradientLength];
		mhat = new float[gradientLength];
		vhat = new float[gradientLength];
		gradSquared = zeroVector(gradientLength);
		
	}
	
}
