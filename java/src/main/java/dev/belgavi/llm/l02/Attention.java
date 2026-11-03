package dev.belgavi.llm.l02;

/**
 * One self-attention head in plain Java, with a causal mask. Author: Sandeep Belgavi.
 *
 * <p>Matrices are {@code double[rows][cols]}; no ML library. {@link #head} returns the attention
 * weights as well as the output, because the weights are what the lesson looks at.
 */
public final class Attention {
  private Attention() {}

  /** Result of one head: {@code weights[i][j]} is how much token i reads from token j. */
  public record Result(double[][] output, double[][] weights) {}

  public static double[][] matmul(double[][] a, double[][] b) {
    int n = a.length, k = b.length, m = b[0].length;
    double[][] c = new double[n][m];
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) {
        double s = 0;
        for (int t = 0; t < k; t++) s += a[i][t] * b[t][j];
        c[i][j] = s;
      }
    return c;
  }

  public static double[][] transpose(double[][] a) {
    double[][] t = new double[a[0].length][a.length];
    for (int i = 0; i < a.length; i++) for (int j = 0; j < a[0].length; j++) t[j][i] = a[i][j];
    return t;
  }

  /** Q = xWq, K = xWk, V = xWv; scores = QK^T / sqrt(d); mask the future; softmax; weights * V. */
  public static Result head(double[][] x, double[][] wq, double[][] wk, double[][] wv, boolean causal) {
    double[][] q = matmul(x, wq), k = matmul(x, wk), v = matmul(x, wv);
    double[][] scores = matmul(q, transpose(k));
    double scale = Math.sqrt(k[0].length);
    double[][] weights = new double[x.length][];
    for (int i = 0; i < x.length; i++) {
      double[] row = new double[x.length];
      for (int j = 0; j < x.length; j++)
        row[j] = (causal && j > i) ? Double.NEGATIVE_INFINITY : scores[i][j] / scale;
      weights[i] = Sampler.softmax(row, 1.0);
    }
    return new Result(matmul(weights, v), weights);
  }
}
