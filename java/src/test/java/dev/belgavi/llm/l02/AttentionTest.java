package dev.belgavi.llm.l02;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/** Fixed-number checks; the same values appear in the Python tests. Author: Sandeep Belgavi. */
class AttentionTest {
  private static final double[][] X = {{1, 0}, {0, 1}, {1, 1}};
  private static final double[][] I = {{1, 0}, {0, 1}};
  private static final double EPS = 1e-6;

  @Test
  void attentionByHand() {
    Attention.Result r = Attention.head(X, I, I, I, true);
    double e = Math.exp(1 / Math.sqrt(2));
    assertArrayEquals(new double[] {1, 0, 0}, r.weights()[0], 1e-12);
    assertArrayEquals(new double[] {1 / (1 + e), e / (1 + e), 0}, r.weights()[1], 1e-12);
    assertArrayEquals(new double[] {0.248255, 0.248255, 0.503490}, r.weights()[2], EPS);
    assertArrayEquals(new double[] {0.751745, 0.751745}, r.output()[2], EPS);
  }

  @Test
  void causalMaskBlocksTheFuture() {
    double[][] w = Attention.head(X, I, I, I, true).weights();
    for (int i = 0; i < 3; i++) for (int j = i + 1; j < 3; j++) assertEquals(0.0, w[i][j]);
    double[][] x2 = {{1, 0}, {0, 1}, {5, -3}};
    double[][] a = Attention.head(x2, I, I, I, true).output();
    double[][] b = Attention.head(X, I, I, I, true).output();
    assertArrayEquals(b[0], a[0], 1e-12);
    assertArrayEquals(b[1], a[1], 1e-12);
  }

  @Test
  void withoutMaskEveryTokenSeesEveryToken() {
    for (double[] row : Attention.head(X, I, I, I, false).weights())
      for (double v : row) assertEquals(true, v > 0);
  }

  @Test
  void temperatureIsOneDivision() {
    double[] z = {2.0, 1.0, 0.1};
    assertArrayEquals(new double[] {0.659001, 0.242433, 0.098566}, Sampler.softmax(z, 1.0), EPS);
    assertArrayEquals(new double[] {0.863777, 0.116900, 0.019323}, Sampler.softmax(z, 0.5), EPS);
    assertArrayEquals(new double[] {0.501688, 0.304289, 0.194023}, Sampler.softmax(z, 2.0), EPS);
    assertArrayEquals(Sampler.softmax(new double[] {4.0, 2.0, 0.2}, 1.0), Sampler.softmax(z, 0.5), 1e-12);
    assertThrows(IllegalArgumentException.class, () -> Sampler.softmax(z, 0.0));
  }

  @Test
  void softmaxIsStableForHugeLogits() {
    assertArrayEquals(new double[] {0.5, 0.5}, Sampler.softmax(new double[] {1000, 1000}, 1.0), 1e-12);
  }

  @Test
  void inverseCdfSampling() {
    double[] p = {0.5, 0.3, 0.2};
    int[] expected = {0, 0, 1, 1, 2, 2};
    double[] us = {0.0, 0.49, 0.5, 0.79, 0.8, 0.999};
    for (int i = 0; i < us.length; i++) assertEquals(expected[i], Sampler.sample(p, us[i]));
  }

  @Test
  void tinyLmMatchesPythonNumbers() {
    TinyLm lm = new TinyLm();
    int[] ids = {0, 1, 2, 3, 0};
    TinyLm.Trace t = lm.forward(ids);
    assertArrayEquals(new double[] {0.50175, 0.49825, 0, 0, 0}, t.weights()[1], 1e-5);
    assertArrayEquals(new double[] {0.626714, -0.183847, 0.039615, -0.093019, -0.214123, 0.009338},
        t.logits()[4], 1e-5);
    double[] p = lm.nextTokenProbs(ids, 1.0);
    assertArrayEquals(new double[] {0.289173, 0.128569, 0.160762, 0.140793, 0.124735, 0.155968}, p, 1e-5);
    assertEquals("the", TinyLm.VOCAB[Sampler.argmax(p)]);
  }
}
