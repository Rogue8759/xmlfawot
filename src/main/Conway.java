package main;

import java.awt.Dimension;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;

import display.Display;

public class Conway {
	static final int boardWidth = 20;
	static final int boardHeight = 20;
	static final int boardArea = boardWidth * boardHeight;
	static final int zoomScale = 8;
	static final float populationDensity = 0.15f;
	
	public static float[] argMax(float[] input) {
		float[] ret = new float[input.length];
		
		for (int i=0;i<input.length;i++) {
			if (input[i] > 0.5f) {
				ret[i] = 1;
			} else {
				ret[i] = 0;
			}
		}
		
		return ret;
		
	}
	
	public static float[][] genRandBoard() {
		float[][] random = new float[boardHeight][boardWidth];
		
		for (int y=1;y<boardHeight-1;y++) {
			for (int x=1;x<boardWidth-1;x++) {
				if (Math.random() > 1.0f - populationDensity) {
					random[y][x] = 1;
				} else {
					random[y][x] = 0;
				}
				
			}
		}
		return random;
	}
	
	public static float[][] updateBoard(float[][] current) {
		float[][] cache = new float[boardHeight][boardWidth];
		int num_alive = 0;
		for (int y=1;y<boardHeight-1;y++) {
			for (int x=1;x<boardWidth-1;x++) {
				num_alive = (int) (
						  (current[y][(x+1)]
						+ (current[y][(x-1)])
						+ (current[(y+1)][x])
						+ (current[(y+1)][(x+1)])
						+ (current[(y+1)][(x-1)])
						+ (current[(y-1)][x])
						+ (current[(y-1)][(x+1)])
						+ (current[(y-1)][(x-1)])));
				
				if (current[y][x] == 1) {
					if (num_alive <= 3 && num_alive >= 2) {
						cache[y][x] = 1;
					}
				} else {
					if (num_alive == 3)
					{
						cache[y][x] = 1;
					}
				}
				
			}
		}
		
		return cache;
	}
	
	public static void main(String[] args) throws InterruptedException {
		// Hyperparameters
		int inputSize = boardArea;
		int outputSize = boardArea;

		int samples = 20;
		int depth = 7;
		float alpha = 1000f;
		int[] layerSizes = {inputSize,500,400,300,300,200, outputSize};
		double loss;

		Tensor2D playEnv = new Tensor2D(boardWidth, boardHeight);
		float[][] randPlayboard = genRandBoard();

		Net network = new Net(layerSizes);
		network.setLearningRate(alpha);
		network.setDropoutChance(.01f);
		network.setSoftmax(true);
		network.setEpochs(1);
		
		// Display
		Display display = new Display(boardWidth, boardHeight);
		JFrame frame = new JFrame();
		frame.add(display);
		frame.pack();
		frame.setTitle("CGOL AI (AWOT)");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(new Dimension(400,400));
		frame.setVisible(true);
		frame.setLocationRelativeTo(null);
		frame.setResizable(true);
		display.start();
		display.setZoomFactor(zoomScale);
		
		Tensor2D board = new Tensor2D(boardWidth, boardHeight);
		float[][] inputData = new float[1][boardArea];
		float[][] outputData = new float[1][boardArea];

		for (int e=0;e<samples;e++) {
			// Inject a random board seed
			board.inject(genRandBoard());
			
			/* Do depth updates to the board
				inject the updated board into the display, and train the network 
			*/
			for (int i=0;i<depth;i++) {
				inputData[0]=board.compress();
				board.inject(updateBoard(board.reform()));
				outputData[0]=board.compress();
				display.setScreenPixels(network.getOutputLayer());
					
				network.train(inputData, outputData, "adam");
				loss = network.get_loss(inputData, outputData);
				System.out.println(loss);
			}
		}
		
		
		// Shows the final results of our training by running a few games using our model
		network.setDropoutChance(0.0f);
		for (int i=0;i<10;i++) {
			randPlayboard = genRandBoard();
			playEnv.inject(randPlayboard);
			Thread.sleep(200);
			inputData[0] = playEnv.compress();
	
			for (int r=0;r<30;r++) {
				
				network.insertInput(inputData[0]);
				network.propagate();
				outputData[0] = network.getOutputLayer();
				playEnv.inject(outputData[0]);
				
				display.setScreenPixels(outputData[0]);
				
				for (int k =0;k<outputData[0].length;k++) {
					System.out.print(" " + outputData[0][k]);
				}
				//System.out.println(outputData[0]);
				inputData[0] = outputData[0];

				Thread.sleep(400);
			}
		}
		
	}
}
