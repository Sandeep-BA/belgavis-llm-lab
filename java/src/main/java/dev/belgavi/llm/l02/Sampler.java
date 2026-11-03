package dev.belgavi.llm.l02;

/** Softmax with temperature and inverse-CDF sampling. Author: Sandeep Belgavi. */
public final class Sampler {
  private Sampler() {}

  /** softmax(z / T). Temperature is just the one division. */
  public static double[] softmax(double[] z, double temperature) {
    if (!(temperature > 0)) throw new IllegalArgumentException("temperature must be > 0");
    double max = Double.NEGATIVE_INFINITY;
    for (double v : z) max = Math.max(max, v / temperature);
    double sum = 0;
    double[] p = new double[z.length];
    for (int i = 0; i < z.length; i++) {
      p[i] = Math.exp(z[i] / temperature - max); // shifting by max keeps exp() from overflowing
      sum += p[i];
    }
    for (int i = 0; i < p.length; i++) p[i] /= sum;
    return p;
  }

  /** Picks the index whose cumulative probability first exceeds the uniform draw u in [0, 1). */
  public static int sample(double[] probs, double u) {
    double c = 0;
    for (int i = 0; i < probs.length; i++) {
      c += probs[i];
      if (u < c) return i;
    }
    return probs.length - 1;
  }

  public static int argmax(double[] p) {
    int best = 0;
    for (int i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    return best;
  }
}
