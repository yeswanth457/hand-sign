"""
Phase 10: English ↔ Tamil Toggle Translation Layer.
Translates recognized ISL semantic sign tokens and English sentences into Tamil (தமிழ்).
"""

from config import TAMIL_VOCAB_MAP


# Contextual Tamil Sentence Rules Map
TAMIL_SENTENCE_RULES = {
    "I want water.": "எனக்கு தண்ணீர் வேண்டும் (Enakku thanneer vaendum)",
    "I want water": "எனக்கு தண்ணீர் வேண்டும் (Enakku thanneer vaendum)",
    "Please give me water.": "எனக்கு தண்ணீர் தரவும் (Enakku thanneer tharavum)",
    "What is your name?": "உங்களது பெயர் என்ன? (Ungaladhu peyar enna?)",
    "My name is": "எனது பெயர் (Enadhu peyar)",
    "I want to eat food.": "நான் உணவு சாப்பிட விரும்புகிறேன் (Naan unavu saappida virumbugiraen)",
    "I am going to school.": "நான் பள்ளிக்கு செல்கிறேன் (Naan pallikku selgiraen)",
    "I am going home.": "நான் வீட்டிற்கு செல்கிறேன் (Naan veettirku selgiraen)",
    "I need help.": "எனக்கு உதவி வேண்டும் (Enakku uthavi vaendum)",
    "I am happy.": "நான் மகிழ்ச்சியாக இருக்கிறேன் (Naan magizhchiyaaga irukkiraen)",
    "I am sad.": "நான் வருத்தமாக இருக்கிறேன் (Naan varuthamaaga irukkiraen)",
    "Mother is at home.": "அம்மா வீட்டில் இருக்கிறார் (Amma veettil irukkiraar)",
    "Father is at work.": "அப்பா வேலையில் இருக்கிறார் (Appa vaelaiyil irukkiraar)",
    "Need medicine.": "மருந்து வேண்டும் (Marundhu vaendum)",
}


class TamilTranslator:
    def __init__(self):
        self.vocab_map = TAMIL_VOCAB_MAP

    def translate_word(self, english_word):
        """Translates a single ISL sign token to Tamil."""
        clean_word = english_word.lower().strip()
        return self.vocab_map.get(clean_word, clean_word)

    def translate_sentence(self, english_sentence, raw_words=None):
        """
        Translates an English sentence or list of raw ISL words to Tamil.
        """
        if not english_sentence and not raw_words:
            return ""

        clean_eng = english_sentence.strip()

        # 1. Exact match in contextual Tamil sentence rules
        if clean_eng in TAMIL_SENTENCE_RULES:
            return TAMIL_SENTENCE_RULES[clean_eng]

        # 2. Word-by-word Tamil translation mapping
        words_to_translate = raw_words if raw_words else clean_eng.replace(".", "").split()
        translated_parts = []

        for w in words_to_translate:
            w_clean = w.lower().strip().replace("_", " ")
            t_val = self.vocab_map.get(w_clean, w_clean)
            translated_parts.append(t_val)

        return " ".join(translated_parts)
