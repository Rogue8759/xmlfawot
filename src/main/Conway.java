package main;

import java.awt.Dimension;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;

public class Conway {
	static int boardWidth = 45;
	static int boardHeight = 45;
	static int boardArea = boardWidth * boardHeight;
	static int zoomScale = 7;
	static float populationDensity = 0.12f;
	
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

		int samples = 4;
		int depth = 25;
		float alpha = .5f;
		int[] layerSizes = {inputSize,600,550,560,550,600, outputSize};
		double loss = 0.0f;
		Tensor2D environment = new Tensor2D(boardWidth, boardHeight);
		float[][] randPlayboard = genRandBoard();
//
		Net network = new Net(layerSizes);
		network.setLearningRate(alpha);
		network.setDropoutChance(0.9f);
		network.setSoftmax(false);
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
		
		// Data shitz
		Tensor2D board = new Tensor2D(boardWidth, boardHeight);
		
		float[][] inputData = new float[1][boardArea];
		float[][] outputData = new float[1][boardArea];

		Net bestNetYet = null;
		double worstLossYet = Double.MAX_VALUE;

		for (int e=0;e<samples;e++) {
			board.inject(genRandBoard());
			
			for (int i=0;i<depth;i++) {
				inputData[0]=board.compress();
				board.inject(updateBoard(board.reform()));
				outputData[0]=board.compress();
				display.setScreenPixels(network.getOutputLayer());
					
				network.train(inputData, outputData, "adam");
				loss = network.get_loss(inputData, outputData);
				if (loss < worstLossYet) {
					bestNetYet = network.copy();
					worstLossYet = loss;
				}
				
				System.out.println(loss);
			}
		}
		
		
		// Shows the final results of our training by running a few games using our model
		for (int i=0;i<10;i++) {
			randPlayboard = genRandBoard();
			environment.inject(randPlayboard);
			Thread.sleep(200);
			inputData[0] = environment.compress();
	
			for (int r=0;r<30;r++) {
				
				bestNetYet.insertInput(inputData[0]);
				bestNetYet.propagate();
				outputData[0] = bestNetYet.getOutputLayer();
				environment.inject(outputData[0]);
				
				display.setScreenPixels(outputData[0]);
				inputData[0] = outputData[0];
				
				
	
				Thread.sleep(55);
			}
		}
		
	}
}
