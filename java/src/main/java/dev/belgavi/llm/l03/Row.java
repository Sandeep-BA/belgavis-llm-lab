// Author: Sandeep Belgavi
package dev.belgavi.llm.l03;

import java.util.List;

/** One chat-format instruction row. {@code source} is the group (here: customer) it came from. */
public record Row(String id, String source, String label, List<Message> messages) {

    public record Message(String role, String content) {}

    /** The text a classifier sees: all user turns joined by a space. */
    public String userText() {
        return messages.stream().filter(m -> "user".equals(m.role())).map(Message::content)
                .reduce((a, b) -> a + " " + b).orElse("");
    }
}
