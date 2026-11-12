// Author: Sandeep Belgavi
package dev.belgavi.llm.l03;

import java.text.Normalizer;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

/** Text normalisation and word n-gram shingles, shared by cleaning, dedup and the leakage checker. */
public final class TextOps {
    private static final Pattern WS = Pattern.compile("\\s+", Pattern.UNICODE_CHARACTER_CLASS);
    private static final Pattern EMAIL = Pattern.compile("[\\w.+-]+@[\\w-]+\\.[\\w.-]+");

    private TextOps() {}

    /** NFKC, mask e-mail addresses, collapse whitespace. Case is kept. */
    public static String normalize(String s) {
        String t = Normalizer.normalize(s, Normalizer.Form.NFKC);
        t = EMAIL.matcher(t).replaceAll("<email>");
        return WS.matcher(t).replaceAll(" ").strip();
    }

    /** Case- and whitespace-insensitive key: what "the same text" means for exact dedup. */
    public static String normKey(String s) {
        return WS.matcher(s.toLowerCase(Locale.ROOT)).replaceAll(" ").strip();
    }

    public static Set<String> shingles(String text, int n) {
        String[] w = normKey(text).isEmpty() ? new String[0] : normKey(text).split(" ");
        Set<String> out = new HashSet<>();
        if (w.length <= n) {
            out.add(String.join(" ", w));
            return out;
        }
        for (int i = 0; i + n <= w.length; i++) {
            out.add(String.join(" ", java.util.Arrays.copyOfRange(w, i, i + n)));
        }
        return out;
    }

    public static double jaccard(Set<String> a, Set<String> b) {
        if (a.isEmpty() && b.isEmpty()) return 1.0;
        int inter = 0;
        for (String s : a) if (b.contains(s)) inter++;
        return (double) inter / (a.size() + b.size() - inter);
    }
}
