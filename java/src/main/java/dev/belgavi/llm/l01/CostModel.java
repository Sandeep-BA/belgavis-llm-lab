// Author: Sandeep Belgavi
package dev.belgavi.llm.l01;

import java.util.ArrayList;
import java.util.List;

/**
 * Token cost and context-window arithmetic. Pure functions: you pass token counts in. Prices are
 * inputs; the lesson ships no vendor price list because prices change. Python twin:
 * {@code belgavi_llm.l01_tokens.cost}.
 */
public final class CostModel {

    private CostModel() {}

    private static final double MILLION = 1_000_000.0;

    /** USD per one million tokens. */
    public record Pricing(double inputPerM, double outputPerM) {
        public double cost(long inputTokens, long outputTokens) {
            return (inputTokens * inputPerM + outputTokens * outputPerM) / MILLION;
        }
    }

    /** Clearly fake numbers so examples are reproducible. Not any vendor's price. */
    public static final Pricing EXAMPLE_PRICING = new Pricing(1.00, 4.00);

    public record ContextWindow(int size, int reservedForOutput) {
        public int usableForInput() {
            return Math.max(0, size - reservedForOutput);
        }

        public boolean fits(long inputTokens) {
            return inputTokens <= usableForInput();
        }

        public double percentUsed(long inputTokens) {
            return 100.0 * inputTokens / size;
        }
    }

    /** How many whole chat turns fit after {@code fixedPrefix} tokens (system prompt, tools). */
    public static int turnsUntilFull(ContextWindow window, int fixedPrefix, int tokensPerTurn) {
        if (tokensPerTurn <= 0) throw new IllegalArgumentException("tokensPerTurn must be positive");
        return Math.max(0, (window.usableForInput() - fixedPrefix) / tokensPerTurn);
    }

    /** One model call: what the model writes, and what its tool returns before the next call. */
    public record Step(int outputTokens, int toolResultTokens) {}

    public record RunCost(
            int calls,
            long inputTokens,
            long outputTokens,
            long peakContextTokens,
            double cost,
            boolean fitsWindow,
            List<Long> perCallInput) {}

    /**
     * Cost of one agent run. Call n is billed for the prefix plus every output and tool result of
     * calls 1..n-1, so input grows with each step. {@code window} may be null (no limit).
     */
    public static RunCost agentRunCost(
            int prefixTokens, List<Step> steps, Pricing pricing, ContextWindow window) {
        if (prefixTokens < 0 || steps.isEmpty()) {
            throw new IllegalArgumentException("need a non-negative prefix and at least one step");
        }
        long history = prefixTokens;
        long outTotal = 0;
        long inTotal = 0;
        List<Long> perCall = new ArrayList<>();
        for (Step s : steps) {
            perCall.add(history);
            inTotal += history;
            outTotal += s.outputTokens();
            history += (long) s.outputTokens() + s.toolResultTokens();
        }
        long peak = perCall.get(perCall.size() - 1) + steps.get(steps.size() - 1).outputTokens();
        boolean fits = window == null || window.fits(peak);
        return new RunCost(steps.size(), inTotal, outTotal, peak, pricing.cost(inTotal, outTotal), fits, perCall);
    }
}
