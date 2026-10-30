"""Synthetic parallel sentences in English, Hindi and Kannada. Author: Sandeep Belgavi.

Written for this lesson: short bookstore help-desk sentences, no real people or data. Each row says
the same thing in the three languages, so token counts are comparable.
"""

from __future__ import annotations

LANGUAGES = ("English", "Hindi", "Kannada")

PARALLEL: tuple[dict[str, str], ...] = (
    {
        "English": "My book order has not arrived. Please check the status.",
        "Hindi": "मेरी किताब का ऑर्डर अभी तक नहीं आया है। कृपया स्थिति जाँच लें।",
        "Kannada": "ನನ್ನ ಪುಸ್ತಕದ ಆರ್ಡರ್ ಇನ್ನೂ ಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ಥಿತಿಯನ್ನು ಪರಿಶೀಲಿಸಿ.",
    },
    {
        "English": "I cannot sign in after resetting my password.",
        "Hindi": "पासवर्ड रीसेट करने के बाद मैं साइन इन नहीं कर पा रहा हूँ।",
        "Kannada": "ಪಾಸ್‌ವರ್ಡ್ ಮರುಹೊಂದಿಸಿದ ನಂತರ ನನಗೆ ಸೈನ್ ಇನ್ ಮಾಡಲು ಆಗುತ್ತಿಲ್ಲ.",
    },
    {
        "English": "The team will reply within one working day.",
        "Hindi": "टीम एक कार्य दिवस के भीतर उत्तर देगी।",
        "Kannada": "ತಂಡವು ಒಂದು ಕೆಲಸದ ದಿನದೊಳಗೆ ಉತ್ತರಿಸುತ್ತದೆ.",
    },
    {
        "English": "Thank you for your patience. We are looking into the problem.",
        "Hindi": "आपके धैर्य के लिए धन्यवाद। हम समस्या की जाँच कर रहे हैं।",
        "Kannada": "ನಿಮ್ಮ ತಾಳ್ಮೆಗೆ ಧನ್ಯವಾದಗಳು. ನಾವು ಸಮಸ್ಯೆಯನ್ನು ಪರಿಶೀಲಿಸುತ್ತಿದ್ದೇವೆ.",
    },
)


def sentences(language: str) -> list[str]:
    return [row[language] for row in PARALLEL]


def paragraph(language: str) -> str:
    """All sample sentences of one language joined into one paragraph."""
    return " ".join(sentences(language))
