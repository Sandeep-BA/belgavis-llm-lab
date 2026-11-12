// Author: Sandeep Belgavi
package dev.belgavi.llm.l03;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class Lesson03Test {
    private static final ObjectMapper M = new ObjectMapper();

    private static JsonNode raw(String id, String src, String user) throws Exception {
        return M.readTree("{\"id\":\"" + id + "\",\"source\":\"" + src + "\",\"label\":\"login\",\"messages\":["
                + "{\"role\":\"user\",\"content\":" + M.writeValueAsString(user) + "},"
                + "{\"role\":\"assistant\",\"content\":\"ok\"}]}");
    }

    private static Row row(String id, String src, String user) {
        return new Row(id, src, "login", List.of(new Row.Message("user", user), new Row.Message("assistant", "ok")));
    }

    private static List<Row> manyRows(int sources, int perSource) {
        List<Row> rows = new ArrayList<>();
        for (int s = 0; s < sources; s++) {
            for (int i = 0; i < perSource; i++) {
                rows.add(row("CASE-" + s + "-" + i, "cust-" + s, "case number " + s + " item " + i + " words here"));
            }
        }
        return rows;
    }

    @Test
    void validatorAcceptsGoodRowAndFlagsBadOnes() throws Exception {
        assertTrue(Validator.validateRow(raw("1", "c", "hello")).isEmpty());
        JsonNode badRole = M.readTree("{\"id\":\"2\",\"source\":\"c\",\"label\":\"x\",\"messages\":"
                + "[{\"role\":\"customer\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"ok\"}]}");
        assertTrue(Validator.validateRow(badRole).stream().anyMatch(e -> e.contains("role")));
        assertTrue(Validator.validateRow(M.readTree("{\"id\":\"3\",\"messages\":[]}")).contains("missing source"));
        assertTrue(Validator.validateRow(raw("4", "c", "   ")).contains("message 0: empty content"));
        assertTrue(Validator.validate(List.of(raw("5", "c", "a"), raw("5", "c", "b"))).get("5").contains("duplicate id"));
    }

    @Test
    void normalizeCollapsesWhitespaceAndMasksEmail() {
        assertEquals("Mail me at <email>", TextOps.normalize("  Mail  me at a.b@x.org  "));
    }

    @Test
    void cleanDropsInvalidAndDedupFindsExactAndNear() throws Exception {
        List<JsonNode> in = List.of(
                raw("1", "c1", "the app crashes when I export my report to pdf today"),
                raw("2", "c1", "THE APP crashes  when I export my report to pdf today"),
                raw("3", "c1", "the app crashes when I export my report to pdf today please"),
                raw("4", "c2", "my receipt shows two charges for the same month"),
                raw("5", "c2", "   "));
        List<Row> cleaned = DatasetBuilder.clean(in);
        assertEquals(4, cleaned.size());
        List<Row> exact = DatasetBuilder.exactDedup(cleaned);
        assertEquals(3, exact.size());
        List<Row> near = DatasetBuilder.nearDedup(exact, 0.7);
        assertEquals(List.of("1", "4"), near.stream().map(Row::id).toList());
    }

    @Test
    void splitByGroupNeverSharesASource() {
        List<Row> rows = manyRows(40, 10);
        DatasetBuilder.Splits s = DatasetBuilder.splitByGroup(rows, 1, 0.1, 0.2);
        Set<String> tr = s.train().stream().map(Row::source).collect(Collectors.toSet());
        Set<String> va = s.validation().stream().map(Row::source).collect(Collectors.toSet());
        Set<String> te = s.test().stream().map(Row::source).collect(Collectors.toSet());
        assertTrue(Collections_disjoint(tr, va) && Collections_disjoint(tr, te) && Collections_disjoint(va, te));
        assertEquals(rows.size(), s.train().size() + s.validation().size() + s.test().size());
        assertEquals(0, LeakageChecker.overlap(s.train(), s.test(), 0.5).sharedSources());
    }

    private static boolean Collections_disjoint(Set<String> a, Set<String> b) {
        return java.util.Collections.disjoint(a, b);
    }

    @Test
    void splitByRowLeaksAndCheckerSeesIt() {
        DatasetBuilder.Splits s = DatasetBuilder.splitByRow(manyRows(40, 10), 1, 0.1, 0.2);
        LeakageChecker.Report r = LeakageChecker.overlap(s.train(), s.test(), 0.5);
        assertTrue(r.sharedSources() > 0);
        assertEquals(1.0, r.leakyFraction());
    }

    @Test
    void checkerCountsExactAndNearText() {
        List<Row> train = List.of(row("1", "c1", "the app crashes when I export my report to pdf today"));
        List<Row> test = List.of(
                row("2", "c2", "The app crashes when I export my report to pdf today"),
                row("3", "c3", "the app crashes when I export my report to pdf now"),
                row("4", "c4", "pre-order charged twice"));
        LeakageChecker.Report r = LeakageChecker.overlap(train, test, 0.5);
        assertEquals(1, r.exactTextOverlap());
        assertEquals(1, r.nearTextOverlap());
    }

    @Test
    void jsonlRoundTripAndBadLineReportsNumber(@TempDir Path dir) throws Exception {
        List<Row> rows = manyRows(3, 3);
        assertEquals(9, Jsonl.write(dir.resolve("x/rows.jsonl"), rows));
        List<Row> back = Jsonl.readTree(dir.resolve("x/rows.jsonl")).stream().map(Jsonl::toRow).toList();
        assertEquals(rows, back);
        Path bad = dir.resolve("bad.jsonl");
        Files.writeString(bad, "{\"a\":1}\n{oops\n");
        Exception e = assertThrows(java.io.IOException.class, () -> Jsonl.readTree(bad));
        assertTrue(e.getMessage().contains(":2:"));
    }
}
