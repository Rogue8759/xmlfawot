package display;

public class Render {
	public final int width;
	public final int height;
	public int[] pixels;
	
	public Render(int width, int height) {
		this.width = width;
		this.height = height;
		pixels = new int[width*height];
	}
	
	public void draw(Render render, int xOffs, int yOffs) {
		// xPix and yPix are the coordinates of the pixel you shall replace
		// we are in essence copying data from another render object into this render object.
		for (int y=0;y<render.height;y++) {
			int yPix = y + yOffs;
			
			for (int x=0;x<render.width;x++) {
				int xPix = x + xOffs;
				pixels[xPix + width * yPix] = render.pixels[x + y * render.width];
			}
			
		}
	}

}
