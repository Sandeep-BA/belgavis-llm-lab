// Author: Sandeep Belgavi
package dev.belgavi.llm.l01;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Counts and inspects tokens with a Hugging Face {@code tokenizer.json}, through DJL. The Python
 * twin is {@code belgavi_llm.l01_tokens.counting}; both load the same file and give the same ids.
 */
public final class TokenCounter implements AutoCloseable {

    /** One token: its id and the exact bytes it stands for. */
    public record Piece(long id, byte[] data) {
        /** Readable text if the piece is whole UTF-8, otherwise the raw bytes in hex. */
        public String show() {
            try {
                return StandardCharsets.UTF_8.newDecoder()
                        .onMalformedInput(CodingErrorAction.REPORT)
                        .onUnmappableCharacter(CodingErrorAction.REPORT)
                        .decode(java.nio.ByteBuffer.wrap(data))
                        .toString();
            } catch (CharacterCodingException e) {
                StringBuilder sb = new StringBuilder("<");
                for (int i = 0; i < data.length; i++) {
                    if (i > 0) sb.append(' ');
                    sb.append(String.format("0x%02X", data[i] & 0xFF));
                }
                return sb.append('>').toString();
            }
        }
    }

    /** GPT-2's byte to printable-character map, inverted: character -> byte. */
    private static final Map<Integer, Integer> CHAR_TO_BYTE = charToByte();

    private final HuggingFaceTokenizer tokenizer;

    private TokenCounter(HuggingFaceTokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    public static TokenCounter load(Path tokenizerJson) throws IOException {
        return new TokenCounter(HuggingFaceTokenizer.builder()
                .optTokenizerPath(tokenizerJson)
                .optAddSpecialTokens(false)
                // DJL truncates at 512 tokens by default; a counter must never drop tokens silently.
                .optTruncation(false)
                .build());
    }

    public int count(String text) {
        return text.isEmpty() ? 0 : tokenizer.encode(text).getIds().length;
    }

    public long[] ids(String text) {
        return text.isEmpty() ? new long[0] : tokenizer.encode(text).getIds();
    }

    public List<Piece> pieces(String text) {
        List<Piece> out = new ArrayList<>();
        if (text.isEmpty()) return out;
        Encoding enc = tokenizer.encode(text);
        long[] ids = enc.getIds();
        String[] tokens = enc.getTokens();
        for (int i = 0; i < ids.length; i++) {
            out.add(new Piece(ids[i], tokenBytes(tokens[i])));
        }
        return out;
    }

    public static int utf8Bytes(String text) {
        return text.getBytes(StandardCharsets.UTF_8).length;
    }

    static byte[] tokenBytes(String token) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        token.codePoints().forEach(cp -> {
            Integer b = CHAR_TO_BYTE.get(cp);
            if (b == null) throw new IllegalArgumentException("not a byte-level token: " + token);
            out.write(b);
        });
        return out.toByteArray();
    }

    private static Map<Integer, Integer> charToByte() {
        List<Integer> keep = new ArrayList<>();
        for (int b = 33; b <= 126; b++) keep.add(b);
        for (int b = 161; b <= 172; b++) keep.add(b);
        for (int b = 174; b <= 255; b++) keep.add(b);
        Map<Integer, Integer> map = new HashMap<>();
        for (int b : keep) map.put(b, b);
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            if (!keep.contains(b)) map.put(256 + extra++, b);
        }
        return map;
    }

    @Override
    public void close() {
        tokenizer.close();
    }
}
