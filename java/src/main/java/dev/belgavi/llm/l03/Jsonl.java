// Author: Sandeep Belgavi
package dev.belgavi.llm.l03;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Reads and writes one-JSON-object-per-line files with Jackson. */
public final class Jsonl {
    private static final ObjectMapper MAPPER =
            new ObjectMapper().configure(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS, true);

    private Jsonl() {}

    /** Raw rows as a JSON tree, so the validator can describe a bad row instead of failing on it. */
    public static List<JsonNode> readTree(Path path) throws IOException {
        List<JsonNode> out = new ArrayList<>();
        int n = 0;
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            n++;
            if (line.isBlank()) continue;
            try {
                out.add(MAPPER.readTree(line));
            } catch (IOException e) {
                throw new IOException(path + ":" + n + ": invalid JSON", e);
            }
        }
        return out;
    }

    public static Row toRow(JsonNode n) {
        List<Row.Message> msgs = new ArrayList<>();
        for (JsonNode m : n.get("messages")) {
            msgs.add(new Row.Message(m.get("role").asText(), m.get("content").asText()));
        }
        return new Row(n.get("id").asText(), n.get("source").asText(), n.get("label").asText(), msgs);
    }

    public static JsonNode toTree(Row r) {
        return MAPPER.valueToTree(r);
    }

    public static int write(Path path, List<Row> rows) throws IOException {
        if (path.getParent() != null) Files.createDirectories(path.getParent());
        try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (Row r : rows) {
                ObjectNode node = (ObjectNode) toTree(r);
                w.write(MAPPER.writeValueAsString(node));
                w.newLine();
            }
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
        return rows.size();
    }
}
