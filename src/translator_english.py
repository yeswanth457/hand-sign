"""
Sign to English Translation Layer and Sentence Buffer.
Uses the 21 ISL sign class translations.
"""

from config import ENGLISH_TRANSLATIONS, ISL_GRAMMAR_RULES


class EnglishSentenceBuilder:
    def __init__(self):
        self.word_buffer = []

    def add_sign(self, sign_word):
        """Adds a confirmed sign word to the sentence buffer if not an immediate duplicate."""
        if sign_word and (len(self.word_buffer) == 0 or self.word_buffer[-1] != sign_word):
            self.word_buffer.append(sign_word)
        return self.get_sentence()

    def get_raw_words(self):
        return list(self.word_buffer)

    def get_sentence(self):
        """Translates raw ISL word sequence buffer into a grammatically refined English sentence."""
        if len(self.word_buffer) == 0:
            return ""

        raw_key = " ".join([w.upper() for w in self.word_buffer])

        # 1. Exact match in ISL Grammar dictionary
        if raw_key in ISL_GRAMMAR_RULES:
            return ISL_GRAMMAR_RULES[raw_key]

        # 2. Single word translation
        if len(self.word_buffer) == 1:
            word = self.word_buffer[0]
            return ENGLISH_TRANSLATIONS.get(word, word.replace("_", " ").capitalize() + ".")

        # 3. Multi-word: join translations
        parts = []
        for w in self.word_buffer:
            translation = ENGLISH_TRANSLATIONS.get(w, w.replace("_", " ").capitalize())
            # Remove trailing period for joining
            if translation.endswith("."):
                translation = translation[:-1]
            parts.append(translation)
        
        return " ".join(parts) + "."

    def pop_last(self):
        if len(self.word_buffer) > 0:
            self.word_buffer.pop()
        return self.get_sentence()

    def clear_buffer(self):
        self.word_buffer.clear()
        return ""
