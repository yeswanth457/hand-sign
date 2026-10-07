"""
Phase 11: Local Sentence Processor & Language Toggle Pipeline.
Combines English sentence builder and Tamil translator into a local, API-free translation engine.
"""

from src.translator_english import EnglishSentenceBuilder
from src.translator_tamil import TamilTranslator


class LocalSentenceProcessor:
    def __init__(self, current_language="english"):
        self.english_builder = EnglishSentenceBuilder()
        self.tamil_translator = TamilTranslator()
        self.current_language = current_language

    def set_language(self, language_code):
        """Sets target translation language: 'english' or 'tamil'."""
        if language_code.lower() in ["english", "tamil"]:
            self.current_language = language_code.lower()
        return self.current_language

    def add_sign_token(self, sign_word):
        """
        Adds a recognized ISL sign word token to sentence buffer and returns translation state.
        """
        eng_sentence = self.english_builder.add_sign(sign_word)
        raw_words = self.english_builder.get_raw_words()
        tamil_sentence = self.tamil_translator.translate_sentence(eng_sentence, raw_words)

        output_text = eng_sentence if self.current_language == "english" else tamil_sentence

        return {
            "english": eng_sentence,
            "tamil": tamil_sentence,
            "raw_words": raw_words,
            "current_language": self.current_language,
            "display_text": output_text
        }

    def get_current_translation(self):
        """Returns current sentence translation in both English and Tamil."""
        eng_sentence = self.english_builder.get_sentence()
        raw_words = self.english_builder.get_raw_words()
        tamil_sentence = self.tamil_translator.translate_sentence(eng_sentence, raw_words)

        output_text = eng_sentence if self.current_language == "english" else tamil_sentence

        return {
            "english": eng_sentence,
            "tamil": tamil_sentence,
            "raw_words": raw_words,
            "current_language": self.current_language,
            "display_text": output_text
        }

    def pop_last_word(self):
        eng_sentence = self.english_builder.pop_last()
        raw_words = self.english_builder.get_raw_words()
        tamil_sentence = self.tamil_translator.translate_sentence(eng_sentence, raw_words)
        return {
            "english": eng_sentence,
            "tamil": tamil_sentence,
            "raw_words": raw_words,
            "current_language": self.current_language,
            "display_text": eng_sentence if self.current_language == "english" else tamil_sentence
        }

    def clear(self):
        self.english_builder.clear_buffer()
        return {
            "english": "",
            "tamil": "",
            "raw_words": [],
            "current_language": self.current_language,
            "display_text": ""
        }
