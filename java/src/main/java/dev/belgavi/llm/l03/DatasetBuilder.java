// Author: Sandeep Belgavi
package dev.belgavi.llm.l03;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

/**
 * Raw JSONL to clean, deduplicated train/validation/test files, split by source so no source lands
 * on both sides. Python twin: {@code belgavi_llm.l03_data}.
 *
 * <pre>java -cp ... dev.belgavi.llm.l03.DatasetBuilder raw.jsonl out-dir</pre>
 */
public final class DatasetBuilder {

    public record Splits(List<Row> train, List<Row> validation, List<Row> test) {}

    public record Result(int rawRows, int droppedInvalid, int exactDuplicatesRemoved,
                         int nearDuplicatesRemoved, int cleanRows, Splits splits) {}

    private DatasetBuilder() {}

    /** Normalise text, drop rows that fail validation (checked after normalising). */
    public static List<Row> clean(List<JsonNode> raw) {
        List<Row> kept = new ArrayList<>();
        for (JsonNode n : raw) {
            if (!Validator.validateRow(normalised(n)).isEmpty()) continue;
            kept.add(Jsonl.toRow(normalised(n)));
        }
        return kept;
    }

    private static JsonNode normalised(JsonNode n) {
        JsonNode copy = n.deepCopy();
        JsonNode msgs = copy.get("messages");
        if (msgs != null && msgs.isArray()) {
            for (JsonNode m : msgs) {
                JsonNode c = m.get("content");
                if (c != null && c.isTextual()) {
                    ((com.fasterxml.jackson.databind.node.ObjectNode) m).put("content", TextOps.normalize(c.asText()));
                }
            }
        }
        return copy;
    }

    public static List<Row> exactDedup(List<Row> rows) {
        Set<String> seen = new HashSet<>();
        List<Row> out = new ArrayList<>();
        for (Row r : rows) if (seen.add(TextOps.normKey(r.userText()))) out.add(r);
        return out;
    }

    /** Keeps the first of each cluster of rows whose shingle Jaccard is at least {@code threshold}. */
    public static List<Row> nearDedup(List<Row> rows, double threshold) {
        List<Row> out = new ArrayList<>();
        List<Set<String>> kept = new ArrayList<>();
        for (Row r : rows) {
            Set<String> s = TextOps.shingles(r.userText(), 3);
            boolean dup = false;
            for (Set<String> k : kept) {
                if (TextOps.jaccard(s, k) >= threshold) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                kept.add(s);
                out.add(r);
            }
        }
        return out;
    }

    /** The tempting default: shuffles rows, so one source appears on both sides. Kept to show the leak. */
    public static Splits splitByRow(List<Row> rows, long seed, double val, double test) {
        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < rows.size(); i++) idx.add(i);
        Collections.shuffle(idx, new Random(seed));
        int nTest = (int) (rows.size() * test), nVal = (int) (rows.size() * val);
        List<Row> tr = new ArrayList<>(), va = new ArrayList<>(), te = new ArrayList<>();
        for (int i = 0; i < idx.size(); i++) {
            (i < nTest ? te : i < nTest + nVal ? va : tr).add(rows.get(idx.get(i)));
        }
        return new Splits(tr, va, te);
    }

    /** Whole sources go to one split. */
    public static Splits splitByGroup(List<Row> rows, long seed, double val, double test) {
        List<String> groups = new ArrayList<>(new TreeSet<>(rows.stream().map(Row::source).toList()));
        Collections.shuffle(groups, new Random(seed));
        int nTest = (int) Math.round(groups.size() * test), nVal = (int) Math.round(groups.size() * val);
        Set<String> testG = new HashSet<>(groups.subList(0, nTest));
        Set<String> valG = new HashSet<>(groups.subList(nTest, nTest + nVal));
        List<Row> tr = new ArrayList<>(), va = new ArrayList<>(), te = new ArrayList<>();
        for (Row r : rows) (testG.contains(r.source()) ? te : valG.contains(r.source()) ? va : tr).add(r);
        return new Splits(tr, va, te);
    }

    public static Result build(List<JsonNode> raw, long seed) {
        List<Row> cleaned = clean(raw);
        List<Row> exact = exactDedup(cleaned);
        List<Row> near = nearDedup(exact, 0.7);
        return new Result(raw.size(), raw.size() - cleaned.size(), cleaned.size() - exact.size(),
                exact.size() - near.size(), near.size(), splitByGroup(near, seed, 0.1, 0.2));
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("usage: DatasetBuilder <raw.jsonl> <out-dir>");
            System.exit(2);
        }
        Path out = Path.of(args[1]);
        List<JsonNode> raw = Jsonl.readTree(Path.of(args[0]));
        Result r = build(raw, 1);
        Jsonl.write(out.resolve("train.jsonl"), r.splits().train());
        Jsonl.write(out.resolve("validation.jsonl"), r.splits().validation());
        Jsonl.write(out.resolve("test.jsonl"), r.splits().test());
        System.out.printf("raw=%d invalid=%d exact_dups=%d near_dups=%d clean=%d%n", r.rawRows(),
                r.droppedInvalid(), r.exactDuplicatesRemoved(), r.nearDuplicatesRemoved(), r.cleanRows());
        System.out.printf("train=%d validation=%d test=%d%n", r.splits().train().size(),
                r.splits().validation().size(), r.splits().test().size());
        System.out.println("group split vs test: " + LeakageChecker.overlap(r.splits().train(), r.splits().test(), 0.5));
        Splits byRow = splitByRow(near(r), 1, 0.1, 0.2);
        System.out.println("row split   vs test: " + LeakageChecker.overlap(byRow.train(), byRow.test(), 0.5));
    }

    private static List<Row> near(Result r) {
        List<Row> all = new ArrayList<>(r.splits().train());
        all.addAll(r.splits().validation());
        all.addAll(r.splits().test());
        return all;
    }
}
