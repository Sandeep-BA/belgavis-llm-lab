// Author: Sandeep Belgavi
package dev.belgavi.llm.l01;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Offline, CPU-only. Reads the toy tokenizer shared with the Python tests. */
class TokenCounterTest {

    // tests run with the java/ module as working directory
    static final Path TESTDATA = Path.of("..", "testdata", "l01");
    static final Path TOY = TESTDATA.resolve("toy-tokenizer.json");
    static final Path CACHE = Path.of("..", "data", "cache", "l01", "tokenizers");

    @Test
    void toyTokenizerGivesTheSameIdsAsPython() throws Exception {
        JsonNode expected = new ObjectMapper().readTree(Files.readString(TESTDATA.resolve("toy-expected.json")));
        try (TokenCounter tc = TokenCounter.load(TOY)) {
            Iterator<Map.Entry<String, JsonNode>> it = expected.fields();
            int checked = 0;
            while (it.hasNext()) {
                Map.Entry<String, JsonNode> e = it.next();
                long[] want = new long[e.getValue().size()];
                for (int i = 0; i < want.length; i++) want[i] = e.getValue().get(i).asLong();
                assertArrayEquals(want, tc.ids(e.getKey()), e.getKey());
                checked++;
            }
            assertTrue(checked >= 5);
        }
    }

    @Test
    void piecesRoundTripToTheOriginalBytes() throws Exception {
        try (TokenCounter tc = TokenCounter.load(TOY)) {
            for (String text : List.of("My book order has not arrived.", "ಧನ್ಯವಾದಗಳು", "धन्यवाद", "")) {
                ByteArrayOutputStream joined = new ByteArrayOutputStream();
                for (TokenCounter.Piece p : tc.pieces(text)) joined.writeBytes(p.data());
                assertArrayEquals(text.getBytes(StandardCharsets.UTF_8), joined.toByteArray(), text);
            }
        }
    }

    @Test
    void longTextIsNeverTruncated() throws Exception {
        try (TokenCounter tc = TokenCounter.load(TOY)) {
            String sentence = "My book order has not arrived. ";
            int one = tc.count(sentence);
            assertEquals(one * 400, tc.count(sentence.repeat(400)));
            assertTrue(tc.count(sentence.repeat(400)) > 512);
        }
    }

    @Test
    void emptyTextIsZeroTokens() throws Exception {
        try (TokenCounter tc = TokenCounter.load(TOY)) {
            assertEquals(0, tc.count(""));
        }
    }

    @Test
    void unseenScriptCostsMoreTokensPerCharacter() throws Exception {
        try (TokenCounter tc = TokenCounter.load(TOY)) {
            String english = "The team will reply within one working day.";
            String kannada = "ತಂಡವು ಒಂದು ಕೆಲಸದ ದಿನದೊಳಗೆ ಉತ್ತರಿಸುತ್ತದೆ.";
            assertTrue((double) tc.count(kannada) / kannada.length()
                    > (double) tc.count(english) / english.length());
        }
    }

    @Test
    void partialUtf8PieceIsShownAsHex() {
        TokenCounter.Piece partial = new TokenCounter.Piece(1, new byte[] {(byte) 0xE0, (byte) 0xB2});
        assertEquals("<0xE0 0xB2>", partial.show());
        assertEquals("ತ", new TokenCounter.Piece(2, "ತ".getBytes(StandardCharsets.UTF_8)).show());
    }

    @Test
    void nonByteLevelTokenIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> TokenCounter.tokenBytes("ತ"));
    }

    @Test
    void cliPrintsTokensForASentence() throws Exception {
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        TokenCli.tokens(TOY, "My book order has not arrived.", new PrintStream(buf, true, StandardCharsets.UTF_8));
        String s = buf.toString(StandardCharsets.UTF_8);
        assertTrue(s.contains("tokens: 7"), s);
        assertTrue(s.contains("ids:    300 386 383 346 337 389 13"), s);
    }

    /** Needs `python -m belgavi_llm.l01_tokens.fetch`; the numbers are the ones Python measured. */
    @Test
    void realTokenizersMatchThePythonMeasurements() throws Exception {
        assumeTrue(Files.isRegularFile(CACHE.resolve("o200k_base.json")), "run the fetch step first");
        Map<String, int[]> expected = Map.of(
                "r50k_base", new int[] {44, 326, 626},
                "cl100k_base", new int[] {43, 214, 405},
                "o200k_base", new int[] {43, 67, 79});
        for (var e : expected.entrySet()) {
            try (TokenCounter tc = TokenCounter.load(CACHE.resolve(e.getKey() + ".json"))) {
                for (int i = 0; i < Samples.LANGUAGES.size(); i++) {
                    assertEquals(e.getValue()[i], tc.count(Samples.paragraph(Samples.LANGUAGES.get(i))),
                            e.getKey() + " " + Samples.LANGUAGES.get(i));
                }
            }
        }
    }
}
