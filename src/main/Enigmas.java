package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public class Enigmas {
	int num_tokens;
	static int bits_per_token;
	static int window_size;
	static HashMap<String, Integer> tokens;
	
	public static void testInputWord(String token, Net net) {
		int divisions = 0;
		float[] insertInput = new float[bits_per_token*window_size];
    	for (int i=tokens.get(token);i>0;i=i/2) {
    		insertInput[divisions] = (float) i%2;
    		divisions++;
    		   
    	}
		
		net.insertInput(insertInput);
		net.propagate();
		
		for (float v : net.getLayer(3)) {
			System.out.print(v + ", ");
		}
			
	
	}
	
	public static void con(String[] args) throws IOException {
		tokens = new HashMap<String, Integer>();
		File corpus = new File("/home/eclipse-workspace/xmlfawot/src/main/corpus.txt");

		try (BufferedReader br = new BufferedReader(new FileReader(corpus))) {
		    int tokenIndex = 0;
		    // Go through the corpus, give every token/word an index number (used for One-Hot encoding)
		    for (String line; (line = br.readLine()) != null;) {
		       for (String token : line.toLowerCase().replaceAll("\\n",  "").split(" ")) {
		    	   if (!tokens.containsKey(token)) {
		    		   tokens.put(token, tokenIndex);
		    		   tokenIndex++;
		    	   }
		       }
		    }
		}
		
		
 		// Actually train the autoencoder
 		int num_tokens = tokens.size();
 		bits_per_token = (int) (Math.log(num_tokens) / Math.log(2));
 		window_size = 5;
 		int[] layer_sizes = {bits_per_token * window_size, 400,18,200,40,bits_per_token*window_size};
 		int max_input_bits = bits_per_token * window_size;
 		Net vectorizer = new Net(layer_sizes);
 		vectorizer.setLearningRate(1f);
 		vectorizer.setSoftmax();
 		vectorizer.setDropoutChance(0.0f);
 
 		float[] input_insert = new float[max_input_bits];
 		int divisions;
 		
 		int buffer_data_samples = 2600; // arbitrary, but should be pretty high
 		float[][] trainDataBuffer = new float[buffer_data_samples][max_input_bits];
 		int bufferDataPtr = 0;
 		System.out.println(tokens.size());
 		
 		Net bestNetYet = null;
 		double worstLossYet = Double.MAX_VALUE;
 		float loss;
 		for (int epoch=0;epoch<4;epoch++) {
	 		try (BufferedReader br = new BufferedReader(new FileReader(corpus))) {
	 		    // Go through the corpus, give every token/word an index number (used for One-Hot encoding)
	 		    for (String line; (line = br.readLine()) != null;) {
	 		    	divisions = 0;
	 		    	for (String token : line.toLowerCase().replaceAll("\\n", "").split(" ")) {
	 		    		for (int i=tokens.get(token);i>0;i=i/2) {
	 		    		   input_insert[divisions] = (float) i%2;
	 		    		   divisions++;
	 		    		   
	 		    		   if (divisions >= max_input_bits) {
	 		    			   divisions = 0;
	 		    			   break;
	 		    		   }
	 		    		   
	 		    	   }
	 		    	   trainDataBuffer[bufferDataPtr] = input_insert;
	 		    	   bufferDataPtr++;
	 		    	   
	 		    	   if (bufferDataPtr >= buffer_data_samples) {
	 		    		   bufferDataPtr = 0;
	 		    		   vectorizer.train(trainDataBuffer,trainDataBuffer,"adam");
	 		    		   loss=(float) vectorizer.get_loss(trainDataBuffer, trainDataBuffer);
	 		    		   System.out.println(vectorizer.get_loss(trainDataBuffer, trainDataBuffer));
	 		    		   
	 		    		   if (loss < worstLossYet) {
	 		    			   bestNetYet = vectorizer.copy();
	 		    			   worstLossYet = loss;
	 		    		   }
	 		    		   
	 		    	   }
	 		    	   
	 		    	}
	 		    	
	 		    	
	 		    }
	 		}
 		}

 
		testInputWord("the", bestNetYet);
		
		
	}
}
