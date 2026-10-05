"""
Configuration parameters for RT-STAMP-SLR ISL Translation System.
EXACTLY 21 classes for real ISL recognition.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Phase 13 Real ISL Dataset Pipeline Paths
RAW_DATA_DIR = os.path.join(DATASET_DIR, "raw")
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks")
TOKENS_DIR = os.path.join(DATASET_DIR, "tokens")
METADATA_DIR = os.path.join(DATASET_DIR, "metadata")
METADATA_CSV_PATH = os.path.join(METADATA_DIR, "dataset.csv")

# Ensure required directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(LANDMARKS_DIR, exist_ok=True)
os.makedirs(TOKENS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "processed"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "test"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# EXACTLY 21-CLASS ISL VOCABULARY
# ═══════════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "hello",       # 0
    "thank_you",   # 1
    "welcome",     # 2
    "goodbye",     # 3
    "yes",         # 4
    "no",          # 5
    "please",      # 6
    "sorry",       # 7
    "help",        # 8
    "stop",        # 9
    "water",       # 10
    "food",        # 11
    "school",      # 12
    "teacher",     # 13
    "mother",      # 14
    "father",      # 15
    "sister",      # 16
    "brother",     # 17
    "friend",      # 18
    "house",       # 19
    "work",        # 20
]

CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)  # Exactly 21 classes
INDEX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# Active vocabulary is the 21-class set
ACTIVE_VOCABULARY = CLASS_NAMES
ACTIVE_VOCAB_SIZE = NUM_CLASSES
ACTIVE_WORD_TO_ID = CLASS_TO_INDEX
ACTIVE_ID_TO_WORD = INDEX_TO_CLASS

# Legacy aliases for backward compatibility with existing code
ISL_VOCABULARY = CLASS_NAMES
VOCAB_SIZE = NUM_CLASSES
WORD_TO_ID = CLASS_TO_INDEX
ID_TO_WORD = INDEX_TO_CLASS

_best_model_path = os.path.join(MODEL_DIR, "isl_cnn_gru_21class_best.pt")
MODEL_PATH = _best_model_path if os.path.exists(_best_model_path) else os.path.join(MODEL_DIR, "isl_cnn_gru.pt")

# Pipeline hyperparameters
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Motion Energy Parameters
MOTION_WINDOW_SIZE = 15
LAMBDA_SIGMA = 0.5   # τ = μ_E + λ * σ_E
IDLE_ENERGY_THRESHOLD = 0.008  # Energy threshold to declare idle state (calibrated for stationary hand)
SIGNING_MOTION_THRESHOLD = 0.012  # Minimum motion energy required to initiate/continue gesture collection

# Tokenizer Parameters
TOKEN_DIM = 6        # [Hx, Hy, Mx, My, Rx, Ry]

# Model Architecture Hyperparameters
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LEN = 25     # Fixed at 25 tokens to match training data shape
DROPOUT = 0.1

# Early Decision & Cooldown Engine
CONFIDENCE_THRESHOLD = 0.45
SUSTAINED_FRAMES = 2
COOLDOWN_FRAMES = 25

# NO Gesture Detector & Calibration Hyperparameters
NO_RANGE_TOLERANCE = 0.20        # Configurable percentile range expansion tolerance (20%)
NO_DETECTOR_THRESHOLD = 0.65     # Recommended similarity score threshold for NO gesture

# ═══════════════════════════════════════════════════════════════════
# TRANSLATION MAPPINGS (21 classes)
# ═══════════════════════════════════════════════════════════════════

# English translations for each sign class
ENGLISH_TRANSLATIONS = {
    "hello":     "Hello.",
    "thank_you": "Thank you.",
    "welcome":   "Welcome.",
    "goodbye":   "Goodbye.",
    "yes":       "Yes.",
    "no":        "No (இல்லை)",
    "please":    "Please.",
    "sorry":     "Sorry.",
    "help":      "Help.",
    "stop":      "Stop.",
    "water":     "Water (தண்ணீர்)",
    "food":      "Food.",
    "school":    "School",
    "teacher":   "Teacher.",
    "mother":    "Mother.",
    "father":    "Father.",
    "sister":    "Sister.",
    "brother":   "Brother.",
    "friend":    "Friend.",
    "house":     "House.",
    "work":      "Work.",
}

# Dynamic Sentence Grammar & Translation Setup
ISL_GRAMMAR_RULES = {
    "I WATER WANT": "I want water",
    "ME WATER GIVE": "Please give me water",
    "I FOOD EAT": "I want to eat food",
    "I HELP NEED": "I need help",
}

# Tamil translations for each sign class
TAMIL_VOCAB_MAP = {
    "hello":     "வணக்கம் (Vanakkam)",
    "thank_you": "நன்றி (Nandri)",
    "welcome":   "நல்வரவு (Nalvaravu)",
    "goodbye":   "சென்று வருகிறேன் (Sentru varugiren)",
    "yes":       "ஆம் (Aam)",
    "no":        "இல்லை",
    "please":    "தயவுசெய்து (Thayavuseythu)",
    "sorry":     "மன்னிக்கவும் (Mannikkavum)",
    "help":      "உதவி (Uthavi)",
    "stop":      "நில் (Nil)",
    "water":     "தண்ணீர்",
    "food":      "உணவு (Unavu)",
    "school":    "பள்ளி (Palli)",
    "teacher":   "ஆசிரியர் (Aasiriyar)",
    "mother":    "அம்மா (Amma)",
    "father":    "அப்பா (Appa)",
    "sister":    "சகோதரி (Sagodhari)",
    "brother":   "சகோதரன் (Sagodharan)",
    "friend":    "நண்பன் (Nanban)",
    "house":     "வீடு (Veedu)",
    "work":      "வேலை (Velai)",
}

# Tamil sentence-level translations
TAMIL_SENTENCE_MAP = {
    "Hello.":           "வணக்கம் (Vanakkam)",
    "Thank you.":       "நன்றி (Nandri)",
    "Welcome.":         "நல்வரவு (Nalvaravu)",
    "Goodbye.":         "சென்று வருகிறேன் (Sentru varugiren)",
    "Yes.":             "ஆம் (Aam)",
    "No.":              "இல்லை",
    "No (இல்லை)":        "இல்லை",
    "Please.":          "தயவுசெய்து (Thayavuseythu)",
    "Sorry.":           "மன்னிக்கவும் (Mannikkavum)",
    "Help.":            "உதவி (Uthavi)",
    "Stop.":            "நில் (Nil)",
    "Water (தண்ணீர்)":   "தண்ணீர்",
    "Water.":           "தண்ணீர்",
    "Water":            "தண்ணீர்",
    "Food.":            "உணவு (Unavu)",
    "School.":          "பள்ளி (Palli)",
    "Teacher.":         "ஆசிரியர் (Aasiriyar)",
    "Mother.":          "அம்மா (Amma)",
    "Father.":          "அப்பா (Appa)",
    "Sister.":          "சகோதரி (Sagodhari)",
    "Brother.":         "சகோதரன் (Sagodharan)",
    "Friend.":          "நண்பன் (Nanban)",
    "House.":           "வீடு (Veedu)",
    "Work.":            "வேலை (Velai)",
}

