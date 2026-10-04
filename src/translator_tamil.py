"""
English ↔ Tamil Toggle Translation Layer.
Translates recognized ISL semantic sign tokens and English sentences into Tamil (தமிழ்).
Only 21 ISL sign classes.
"""

from config import TAMIL_VOCAB_MAP, TAMIL_SENTENCE_MAP


class TamilTranslator:
    def __init__(self):
        self.vocab_map = TAMIL_VOCAB_MAP

    def translate_word(self, english_word):
        """Translates a single ISL sign token to Tamil."""
        clean_word = english_word.lower().strip()
        return self.vocab_map.get(clean_word, clean_word)

    def translate_sentence(self, english_sentence, raw_words=None):
        """Translates an English sentence or list of raw ISL words to Tamil."""
        if not english_sentence and not raw_words:
            return ""

        clean_eng = english_sentence.strip()

        # 1. Exact match in sentence-level Tamil rules
        if clean_eng in TAMIL_SENTENCE_MAP:
            return TAMIL_SENTENCE_MAP[clean_eng]

        # 2. Word-by-word Tamil translation mapping
        words_to_translate = raw_words if raw_words else clean_eng.replace(".", "").split()
        translated_parts = []

        for w in words_to_translate:
            w_clean = w.lower().strip()
            t_val = self.vocab_map.get(w_clean, w_clean)
            translated_parts.append(t_val)

        return " ".join(translated_parts)
