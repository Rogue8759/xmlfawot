package main;

import java.awt.Canvas;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;

import javax.swing.JFrame;

public class Display extends Canvas implements Runnable {
	public int width;
	public int height;
	private Thread thread;
	private boolean running;
	private BufferedImage img;
	private Screen screen;
	private int[] pixels;
	private int zoomFactor;
	
	public void setZoomFactor(float zoomFactor ) {
		this.zoomFactor = (int) zoomFactor;
	}
	
	public Display(int width, int height) {
		// how we display images is that we set the raw pixels in an array in the Display class,
		// and we use our Display's instance of a BufferedImage to set the BufferedImage's pixel data
		// we then render our image by creating a disposable graphics object and using bufferstrategy
		// on our JFrame.
		this.width = width;
		this.height = height;
		screen = new Screen(width, height);
		img = new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB);
		pixels = ((DataBufferInt)img.getRaster().getDataBuffer()).getData();
	}
	
	public void run() {
		while (running) {
			tick();
			render();
		}
		
	}
	
	public void stop() throws InterruptedException {
		if (!running) {
			return;
		}
		
		running = false;
		thread.join();
		
	}
	
	private void tick() {
		
	}
	
	private void render() {
		// The display class serves as a wrapper for a buffered image and a jframe.
		// we render the bufferedimage to the jframe using an on-the-fly graphics object.
		// we then show the bufferstrategy, which has the effect of giving us our image
		
		BufferStrategy bs = this.getBufferStrategy();
		if (bs == null) {
			createBufferStrategy(2);
			return;
		}
		
		for (int i =0;i<width*height;i++) {
			pixels[i] = screen.pixels[i];
		}
		
		screen.render();
		
		Graphics g = bs.getDrawGraphics();
		g.drawImage(img, 0,0,width*zoomFactor,height*zoomFactor,null);
		g.dispose();
		bs.show();
	}
	
	public void setScreenPixels(float[] pixels) {
		screen.setPixels(pixels);
	}
	
	public void start() {
		if (running) {
			return;
		}
		
		running = true;
		thread = new Thread(this);
		thread.start();
	}
	
}
