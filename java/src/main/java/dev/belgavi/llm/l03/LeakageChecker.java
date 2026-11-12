// Author: Sandeep Belgavi
package dev.belgavi.llm.l03;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Reports what a held-out split shares with the training split. */
public final class LeakageChecker {

    public record Report(int rows, int sharedSources, int rowsFromSharedSources,
                         int exactTextOverlap, int nearTextOverlap, double leakyFraction) {}

    private LeakageChecker() {}

    public static Report overlap(List<Row> train, List<Row> other, double nearThreshold) {
        Set<String> trainSources = new HashSet<>();
        Set<String> trainText = new HashSet<>();
        List<Set<String>> trainShingles = train.stream().map(r -> TextOps.shingles(r.userText(), 3)).toList();
        for (Row r : train) {
            trainSources.add(r.source());
            trainText.add(TextOps.normKey(r.userText()));
        }
        Set<String> shared = new HashSet<>();
        int fromShared = 0, exact = 0, near = 0;
        for (Row r : other) {
            if (trainSources.contains(r.source())) {
                fromShared++;
                shared.add(r.source());
            }
            if (trainText.contains(TextOps.normKey(r.userText()))) {
                exact++;
                continue;
            }
            Set<String> s = TextOps.shingles(r.userText(), 3);
            for (Set<String> t : trainShingles) {
                if (TextOps.jaccard(s, t) >= nearThreshold) {
                    near++;
                    break;
                }
            }
        }
        double frac = other.isEmpty() ? 0.0 : Math.round(1000.0 * fromShared / other.size()) / 1000.0;
        return new Report(other.size(), shared.size(), fromShared, exact, near, frac);
    }
}
