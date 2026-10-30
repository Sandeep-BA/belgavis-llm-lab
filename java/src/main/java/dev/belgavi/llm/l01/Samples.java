// Author: Sandeep Belgavi
package dev.belgavi.llm.l01;

import java.util.List;
import java.util.Map;

/**
 * Synthetic parallel sentences in English, Hindi and Kannada: short bookstore help-desk lines written for
 * this lesson, no real people or data. Same text as {@code belgavi_llm.l01_tokens.samples}.
 */
public final class Samples {

    private Samples() {}

    public static final List<String> LANGUAGES = List.of("English", "Hindi", "Kannada");

    private static final Map<String, List<String>> SENTENCES = Map.of(
            "English", List.of(
                    "My book order has not arrived. Please check the status.",
                    "I cannot sign in after resetting my password.",
                    "The team will reply within one working day.",
                    "Thank you for your patience. We are looking into the problem."),
            "Hindi", List.of(
                    "मेरी किताब का ऑर्डर अभी तक नहीं आया है। कृपया स्थिति जाँच लें।",
                    "पासवर्ड रीसेट करने के बाद मैं साइन इन नहीं कर पा रहा हूँ।",
                    "टीम एक कार्य दिवस के भीतर उत्तर देगी।",
                    "आपके धैर्य के लिए धन्यवाद। हम समस्या की जाँच कर रहे हैं।"),
            "Kannada", List.of(
                    "ನನ್ನ ಪುಸ್ತಕದ ಆರ್ಡರ್ ಇನ್ನೂ ಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ಥಿತಿಯನ್ನು ಪರಿಶೀಲಿಸಿ.",
                    "ಪಾಸ್‌ವರ್ಡ್ ಮರುಹೊಂದಿಸಿದ ನಂತರ ನನಗೆ ಸೈನ್ ಇನ್ ಮಾಡಲು ಆಗುತ್ತಿಲ್ಲ.",
                    "ತಂಡವು ಒಂದು ಕೆಲಸದ ದಿನದೊಳಗೆ ಉತ್ತರಿಸುತ್ತದೆ.",
                    "ನಿಮ್ಮ ತಾಳ್ಮೆಗೆ ಧನ್ಯವಾದಗಳು. ನಾವು ಸಮಸ್ಯೆಯನ್ನು ಪರಿಶೀಲಿಸುತ್ತಿದ್ದೇವೆ."));

    public static List<String> sentences(String language) {
        return SENTENCES.get(language);
    }

    /** All sample sentences of one language joined into one paragraph. */
    public static String paragraph(String language) {
        return String.join(" ", sentences(language));
    }
}
