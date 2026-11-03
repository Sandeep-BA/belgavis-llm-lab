package dev.belgavi.llm.l02;

import java.util.Arrays;

/**
 * CPU walk-through of lesson 2 in Java. Author: Sandeep Belgavi.
 *
 * <pre>mvn -B -ntp -q -f java/pom.xml compile exec:java -Dexec.mainClass=dev.belgavi.llm.l02.Lesson02Demo</pre>
 */
public final class Lesson02Demo {
  private Lesson02Demo() {}

  private static void print(String title, double[][] m) {
    System.out.println("\n" + title);
    for (double[] row : m) {
      StringBuilder sb = new StringBuilder("  ");
      for (double v : row) sb.append(String.format("%8.4f", v));
      System.out.println(sb);
    }
  }

  public static void main(String[] args) {
    int[] ids = {0, 1, 2, 3, 0};
    System.out.println("tokens: " + Arrays.toString(Arrays.stream(ids).mapToObj(i -> TinyLm.VOCAB[i]).toArray()));
    TinyLm lm = new TinyLm();
    TinyLm.Trace t = lm.forward(ids);
    print("1. embeddings", t.x());
    print("2. causal attention weights", t.weights());
    print("3. after attention + residual", t.h1());
    print("4. MLP output", t.mlp());
    double[] p = lm.nextTokenProbs(ids, 1.0);
    System.out.println("\n5. next-token distribution after 'the cat sat on the':");
    double sum = 0;
    for (int i = 0; i < p.length; i++) {
      System.out.printf("   %4s  %.4f%n", TinyLm.VOCAB[i], p[i]);
      sum += p[i];
    }
    System.out.printf("   sum = %.4f%n", sum);
    System.out.println("\nTemperature is one division (logits / T):");
    double[] logits = t.logits()[ids.length - 1];
    for (double temp : new double[] {0.25, 0.5, 1.0, 2.0, 4.0}) {
      double[] q = Sampler.softmax(logits, temp);
      int top = Sampler.argmax(q);
      System.out.printf("   T=%-4s top=%s p_top=%.4f sample(u=0.5)=%s%n", temp, TinyLm.VOCAB[top], q[top],
          TinyLm.VOCAB[Sampler.sample(q, 0.5)]);
    }
  }
}
