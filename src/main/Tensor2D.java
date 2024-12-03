package main;

public class Tensor2D {
	private int width;
	private int height;
	private float[] data;
	
	public float[] compress() {
		return data;
	}
	
	public Tensor2D(int width_, int height_) {
		width = width_;
		height = height_;
		data = new float[width_ * height_];
	}
	
	public void inject(float[][] array2d) {
		int i = 0;
		
		for (float[] row : array2d) {
			for (float piece: row) {
				data[i] = piece;
				i++;
			}
		}
	}
	
	public void inject(float[] array) {
		int i = 0;
		
		for (float piece: array) {
			data[i] = piece;
			i++;
		}
	}
	
	public float[][] reform() {
		float[][] retrn = new float[height][width];
		
		int i = 0;
		for (int r=0;r<height;r++) {
			for (int c=0;c<width;c++) {
				retrn[r][c] = data[i];
				i++;
			}
		}
		
		
		return retrn;
	}
}
