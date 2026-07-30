"""Translate transcribed text into English using deep-translator (free
Google Translate endpoint, no API key needed)."""

from deep_translator import GoogleTranslator

LANGUAGE_CODE_MAP = {
    "yoruba": "yo",
    "hausa": "ha",
    "igbo": "ig",
}


def translate_to_english(text: str, lang: str) -> str:
    if not text or not text.strip():
        return ""

    source = LANGUAGE_CODE_MAP.get(lang.lower(), "auto")

    try:
        return GoogleTranslator(source=source, target="en").translate(text)
    except Exception as e:
        return f"[translation failed: {e}] {text}"


def translate_from_english(text: str, lang: str) -> str:
    """Translate an English response back into the target language."""
    if not text or not text.strip():
        return ""

    target = LANGUAGE_CODE_MAP.get(lang.lower())
    if target is None:
        return text  # unknown language code -- return English rather than fail

    try:
        return GoogleTranslator(source="en", target=target).translate(text)
    except Exception as e:
        return f"[translation failed: {e}] {text}"
