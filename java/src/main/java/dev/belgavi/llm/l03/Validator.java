// Author: Sandeep Belgavi
package dev.belgavi.llm.l03;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Checks raw rows against the chat format. Same rules as the Python {@code jsonl.validate_row}. */
public final class Validator {
    public static final Set<String> ROLES = Set.of("system", "user", "assistant");
    public static final int MAX_CHARS = 2000;

    private Validator() {}

    public static List<String> validateRow(JsonNode row) {
        List<String> errs = new ArrayList<>();
        for (String key : List.of("id", "source", "label")) {
            JsonNode v = row.get(key);
            if (v == null || !v.isTextual() || v.asText().isBlank()) errs.add("missing " + key);
        }
        JsonNode msgs = row.get("messages");
        if (msgs == null || !msgs.isArray() || msgs.isEmpty()) {
            errs.add("messages must be a non-empty list");
            return errs;
        }
        boolean hasUser = false;
        String lastRole = null;
        int i = 0;
        for (JsonNode m : msgs) {
            JsonNode role = m.get("role");
            JsonNode content = m.get("content");
            if (role == null || !role.isTextual() || !ROLES.contains(role.asText())) {
                errs.add("message " + i + ": role must be one of " + ROLES.stream().sorted().toList());
            } else {
                lastRole = role.asText();
                hasUser |= lastRole.equals("user");
                if (content == null || !content.isTextual() || content.asText().isBlank()) {
                    errs.add("message " + i + ": empty content");
                } else if (content.asText().length() > MAX_CHARS) {
                    errs.add("message " + i + ": longer than " + MAX_CHARS + " characters");
                }
            }
            i++;
        }
        if (!hasUser) errs.add("no user message");
        if (lastRole != null && !lastRole.equals("assistant")) errs.add("last message must be from the assistant");
        return errs;
    }

    /** Row id (or {@code #index}) to its problems; valid rows are omitted. Duplicate ids are flagged. */
    public static Map<String, List<String>> validate(List<JsonNode> rows) {
        Map<String, List<String>> problems = new LinkedHashMap<>();
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < rows.size(); i++) {
            JsonNode r = rows.get(i);
            JsonNode idNode = r.get("id");
            String key = idNode != null && idNode.isTextual() && !idNode.asText().isEmpty()
                    ? idNode.asText() : "#" + i;
            List<String> errs = validateRow(r);
            if (!seen.add(key)) errs.add("duplicate id");
            if (!errs.isEmpty()) problems.computeIfAbsent(key, k -> new ArrayList<>()).addAll(errs);
        }
        return problems;
    }
}
