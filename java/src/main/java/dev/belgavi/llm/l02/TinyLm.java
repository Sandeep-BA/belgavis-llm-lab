package dev.belgavi.llm.l02;

/**
 * A toy transformer: embeddings, one causal attention head, an MLP, tied output projection.
 * Author: Sandeep Belgavi. Same deterministic weights as the Python {@code TinyLM}.
 */
public final class TinyLm {
  public static final String[] VOCAB = {"the", "cat", "sat", "on", "mat", "."};
  public static final int D_MODEL = 4;
  public static final int D_HIDDEN = 8;

  /** Everything the forward pass computes, so each step can be inspected. */
  public record Trace(double[][] x, double[][] weights, double[][] h1, double[][] mlp,
      double[][] h2, double[][] logits) {}

  private final double[][] table = weight(1, VOCAB.length, D_MODEL);
  private final double[][] wq = weight(2, D_MODEL, D_MODEL);
  private final double[][] wk = weight(3, D_MODEL, D_MODEL);
  private final double[][] wv = weight(4, D_MODEL, D_MODEL);
  private final double[][] wo = weight(5, D_MODEL, D_MODEL);
  private final double[][] w1 = weight(6, D_MODEL, D_HIDDEN);
  private final double[][] w2 = weight(7, D_HIDDEN, D_MODEL);

  /** Entry (i, j) = ((7i + 13j + 5 seed) mod 11 - 5) / 10. */
  public static double[][] weight(int seed, int rows, int cols) {
    double[][] w = new double[rows][cols];
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) w[i][j] = (((7 * i + 13 * j + 5 * seed) % 11) - 5) / 10.0;
    return w;
  }

  /** GELU, tanh approximation (the one GPT-2 uses). */
  public static double gelu(double x) {
    return 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
  }

  private static double[][] add(double[][] a, double[][] b) {
    double[][] c = new double[a.length][a[0].length];
    for (int i = 0; i < a.length; i++)
      for (int j = 0; j < a[0].length; j++) c[i][j] = a[i][j] + b[i][j];
    return c;
  }

  public Trace forward(int[] ids) {
    double[][] x = new double[ids.length][];
    for (int i = 0; i < ids.length; i++) x[i] = table[ids[i]].clone(); // embedding lookup
    Attention.Result a = Attention.head(x, wq, wk, wv, true);
    double[][] h1 = add(x, Attention.matmul(a.output(), wo)); // residual
    double[][] hidden = Attention.matmul(h1, w1);
    for (double[] row : hidden) for (int j = 0; j < row.length; j++) row[j] = gelu(row[j]);
    double[][] m = Attention.matmul(hidden, w2);
    double[][] h2 = add(h1, m); // residual
    double[][] logits = Attention.matmul(h2, Attention.transpose(table)); // tied embeddings
    return new Trace(x, a.weights(), h1, m, h2, logits);
  }

  public double[] nextTokenProbs(int[] ids, double temperature) {
    double[][] logits = forward(ids).logits();
    return Sampler.softmax(logits[logits.length - 1], temperature);
  }
}
