package main;

import java.util.Random;

public class Screen extends Render {
	private Render screenRender;
	
	public Screen(int width, int height) {
		super(width, height);
		Random random = new Random();
		screenRender = new Render(width,height);
		for (int i =0;i<width*height;i++) {
			screenRender.pixels[i] = random.nextInt();
		}
	}
	
	public void render() {
		// Draw the render object "test" onto this render object's pixels
		draw(screenRender, 0,0);
	}

	public void setPixels(float[] pixels) {
		for (int i =0;i<width*height;i++) {
			screenRender.pixels[i] = (int) (pixels[i]*2000000000);
		}
		
	}

}
