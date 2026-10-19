"""Conservative, stateless language choices for automatic voice recognition.

Whisper probabilities remain global probabilities: restricting the supported
languages never turns a weak score into a confident one. Text evidence can
repair a label for a complete known phrase, but never rewrites the transcript.
"""

import math
import re
import unicodedata
from collections.abc import Mapping


SUPPORTED_LANGUAGES = {
    "en": "Inggris", "zh": "Mandarin", "hi": "Hindi", "es": "Spanyol",
    "ar": "Arab Standar Modern", "fr": "Prancis", "pt": "Portugis",
    "id": "Indonesia", "ja": "Jepang", "ko": "Korea",
}


def build_language_instruction():
    """Shared reply-language rules for the supported multilingual voice path."""
    languages = ", ".join(f"{code}={name}" for code, name in SUPPORTED_LANGUAGES.items())
    return (
        f"Bahasa yang didukung: {languages}. "
        "Tag [bahasa: ...] pada pesan terbaru dari pengguna menentukan bahasa reply; dahulukan tag "
        "tersebut daripada bahasa riwayat dan jangan tampilkan tag itu. Pahami teks input asli tanpa "
        "mengganti atau menerjemahkannya. Untuk ar, jawab dalam Arab Standar "
        "Modern (Fusha); zh dalam Mandarin dengan aksara Tionghoa sederhana; "
        "hi dengan aksara Devanagari; ja dalam bahasa Jepang; ko dengan Hangul. "
        "Jangan terjemahkan kode command. Pertahankan arti larangan, jumlah, "
        "durasi, dan urutan aksi."
    )


def _probability(value):
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def _language(value):
    return value.strip().lower() if isinstance(value, str) else None


def _probabilities(all_probs):
    """Ignore malformed entries; duplicate languages keep their maximum score."""
    if isinstance(all_probs, Mapping):
        entries = all_probs.items()
    elif isinstance(all_probs, (list, tuple)):
        entries = all_probs
    else:
        entries = ()
    result = {}
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        language, probability = _language(entry[0]), _probability(entry[1])
        if language and probability is not None:
            result[language] = max(result.get(language, 0), probability)
    return result


def select_language(detected, probability, all_probs):
    """Choose a supported acoustic candidate without rescaling its score.

    Clearly recognized unsupported speech is rejected when every supported
    alternative is weak. Missing full distributions only permit the original
    supported language; they never imply Indonesian or the previous language.
    """
    detected, detected_probability = _language(detected), _probability(probability)
    probabilities = _probabilities(all_probs)
    distribution_available = bool(probabilities)
    if detected_probability is None:
        detected_probability = probabilities.get(detected)
    if detected in SUPPORTED_LANGUAGES and detected_probability is not None:
        probabilities.setdefault(detected, detected_probability)
    supported = sorted(
        ((language, probabilities[language]) for language in SUPPORTED_LANGUAGES
         if language in probabilities),
        key=lambda item: -item[1],
    )
    mass = sum(score for _, score in supported) if distribution_available else None
    # Malformed distributions must not advertise an impossible probability.
    if mass is not None and mass > 1 + 1e-6:
        mass = None
    elif mass is not None:
        mass = min(mass, 1.0)
    language, score = supported[0] if supported else (None, None)
    reason = "supported_detection" if language == detected else "supported_candidate"
    if not supported:
        language = detected if detected in SUPPORTED_LANGUAGES else None
        score = detected_probability if language else None
        reason = "supported_detection" if language else "no_supported_candidate"
    elif (detected not in SUPPORTED_LANGUAGES
          and detected_probability is not None and detected_probability >= 0.8
          and score <= 0.1):
        language, score, reason = None, None, "unsupported_confident"
    alternatives = [
        {"language": code, "probability": value}
        for code, value in supported if code != language
    ][:2]
    return {
        "language": language,
        "probability": score,
        "detected_language": detected,
        "detected_probability": detected_probability,
        "supported_mass": mass,
        "reason": reason,
        "alternatives": alternatives,
    }


def _phrase(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text).casefold()
    # Remove punctuation only at the edges. Internal punctuation/negations are
    # not stripped into an unrelated command or a shorter matching fragment.
    text = text.strip()
    while text and unicodedata.category(text[0]).startswith("P"):
        text = text[1:].lstrip()
    while text and unicodedata.category(text[-1]).startswith("P"):
        text = text[:-1].rstrip()
    return re.sub(r"\s+", " ", text)


# Entire phrases only; these must never match a fragment of a longer sentence.
# These are label hints, not a command parser; arbitrary speech remains intact.
_PHRASES = {
    "en": ("sit", "stand", "walk", "look right", "look left", "thank you"),
    "zh": ("坐下", "站起来", "向右看", "向左看", "谢谢"),
    "hi": ("बैठो", "खड़े हो जाओ", "दाईं ओर देखो", "बाईं ओर देखो", "धन्यवाद"),
    "es": ("siéntate", "sientate", "ponte de pie", "mira a la derecha", "mira a la izquierda", "gracias"),
    "ar": ("اجلس", "قف", "انظر يمينا", "انظر يسارا", "شكرا"),
    "fr": ("assieds-toi", "levez-vous", "regarde à droite", "regarde à gauche", "merci"),
    "pt": ("sente-se", "levante-se", "olhe para direita", "olhe para esquerda", "olhe para a direita", "olhe para a esquerda", "obrigado", "obrigada"),
    "id": ("duduk", "berdiri", "jalan", "maju", "mundur", "berhenti", "tengok kanan", "tengok kiri", "terima kasih"),
    "ja": ("座って", "すわって", "立って", "たって", "右を見て", "左を見て", "ありがとう"),
    "ko": ("앉아", "일어서", "오른쪽을 봐", "왼쪽을 봐", "고마워"),
}


def reconcile_text_language(
    text, selected, selected_probability, all_probs, detected_probability=None,
):
    """Return (language, raw_probability, reason), leaving text untouched.

    Confident selected languages are preserved. A whole phrase shared between
    languages retains the selected language, otherwise its acoustic ranking is
    used. Unknown phrases, scripts alone, and rejected languages are unchanged.
    """
    selected = _language(selected)
    probability = _probability(selected_probability)
    if selected not in SUPPORTED_LANGUAGES:
        return selected, probability, "unchanged"
    confidence = probability
    if confidence is None:
        confidence = _probability(detected_probability)
    if confidence is not None and confidence >= 0.85:
        return selected, probability, "confident_language"
    phrase = _phrase(text)
    candidates = [code for code, phrases in _PHRASES.items() if phrase in phrases]
    if not candidates:
        return selected, probability, "unchanged"
    if selected in candidates:
        return selected, probability, "shared_phrase" if len(candidates) > 1 else "unchanged"
    probabilities = _probabilities(all_probs)
    if len(candidates) > 1:
        ranked = [code for code in candidates if code in probabilities]
        if not ranked:
            return selected, probability, "ambiguous_phrase"
        language = max(ranked, key=lambda code: probabilities[code])
    else:
        language = candidates[0]
    return language, probabilities.get(language), "phrase_evidence"
