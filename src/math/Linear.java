package math;

public class Linear {
	public float relu(float x) {
		if (x < 0.0f) {
			return 0;
		}
		return x;
	}
	
	public float reluD(float x) {
		if (x < 0.0f) {
			return 0;
		}
		return 1;
	}
	
	public float sigmoid(float x) {
		return (float) (1.0f / (1.0f + Math.exp(-x)));
	}
	
	public float sigmoidDv(float x) {
		return (float) -Math.pow(1+Math.exp(-x), -2.0f);
	}
	
	public float[] subtract(float[] a, float[] b) {
		float[] retrn = new float[a.length];
		
		for (int i = 0; i < a.length; i++) {
			retrn[i] = a[i]-b[i];
		}

		return retrn;

	}
	
	public float[] zeroVector(int size) {
		return new float[size];
	}
	
	public float[] divide(float n, float[] v) {
		float[] retrn = new float[v.length];
		
		for (int i = 0; i < v.length; i++) {
			retrn[i] = n / v[i];
		}

		return retrn;
	}
	
	public static float[] sqrt(float[] x) {
		float[] retrn = new float[x.length];
		
		for (int i = 0; i < x.length; i++) {
			retrn[i] = (float) Math.pow(x[i], 0.5f);
		}
		
		return retrn;
		
	}
	
	public float scalar_sqrt(float[] x) {
		float sum = 0;
		
		for (float v : x) {
			sum += v;
		}
		
		return sum/x.length;
	}
	
	public float[] add(float[] a, float[] b) {
		float[] retrn = new float[a.length];
		
		for (int i = 0;i<a.length;i++) {
			retrn[i] = a[i]+b[i];
		}

		return retrn;

	}
	
	
	public float[] multiply(float[] a, float f) {
		float[] retrn = new float[a.length];
		
		for (int i = 0; i < a.length; i++) {
			retrn[i] = a[i]*f;
		}

		return retrn;

	}
	
	public float[] square(float[] x) {
		float[] retrn = new float[x.length];
		
		for (int i = 0; i < x.length; i++) {
			retrn[i] = (float) Math.pow(x[i], 2.0f);
		}
		
		return retrn;
		
	}
	
	public float[] averageVecs(float[][] gradients) {
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
}
