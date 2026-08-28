"""
Phase 9: Sign to English Translation Layer and Sentence Buffer.
Maintains recognized word stream buffer and applies deterministic ISL grammar rules (SOV to SVO).
"""

from config import ISL_GRAMMAR_RULES


class EnglishSentenceBuilder:
    def __init__(self):
        self.word_buffer = []

    def add_sign(self, sign_word):
        """
        Adds a confirmed sign word to the sentence buffer if not an immediate duplicate.
        """
        if sign_word and (len(self.word_buffer) == 0 or self.word_buffer[-1] != sign_word):
            self.word_buffer.append(sign_word)
        return self.get_sentence()

    def get_raw_words(self):
        return list(self.word_buffer)

    def get_sentence(self):
        """
        Translates raw ISL word sequence buffer into a grammatically refined English sentence.
        """
        if len(self.word_buffer) == 0:
            return ""

        raw_key = " ".join([w.upper() for w in self.word_buffer])

        # 1. Exact match in ISL Grammar dictionary
        if raw_key in ISL_GRAMMAR_RULES:
            return ISL_GRAMMAR_RULES[raw_key]

        # 2. Heuristic grammar rules for SOV -> SVO translation
        words = list(self.word_buffer)
        
        # Replace token underscores with spaces (e.g. thank_you -> thank you)
        clean_words = [w.replace("_", " ") for w in words]

        # Handle basic SOV reordering if 'want' or 'need' or 'go' is at end
        if len(clean_words) >= 3 and clean_words[-1] in ["want", "need"]:
            subject = clean_words[0]
            obj = " ".join(clean_words[1:-1])
            verb = clean_words[-1]
            return f"{subject.capitalize()} {verb} {obj}."

        # Handle 'going to'
        if len(clean_words) >= 2 and clean_words[-1] == "go":
            subject = clean_words[0]
            location = " ".join(clean_words[1:-1]) if len(clean_words) > 2 else "there"
            return f"{subject.capitalize()} is going to {location}."

        # Fallback default joins words and capitalizes first letter
        sentence = " ".join(clean_words)
        return sentence[0].upper() + sentence[1:] + "."

    def pop_last(self):
        if len(self.word_buffer) > 0:
            self.word_buffer.pop()
        return self.get_sentence()

    def clear_buffer(self):
        self.word_buffer.clear()
        return ""
