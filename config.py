"""
Configuration parameters for RT-STAMP-SLR ISL Translation System.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "isl_temporal_transformer.pt")

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

# 50 ISL Sign Vocabulary definitions
ISL_VOCABULARY = [
    # Greetings & Courtesy (0-7)
    "hello", "thank_you", "welcome", "goodbye", "please", "sorry", "yes", "no",
    # Basic Needs & Daily Life (8-17)
    "water", "eat", "drink", "food", "help", "need", "want", "sleep", "toilet", "medicine",
    # People & Family (18-25)
    "father", "mother", "brother", "sister", "friend", "teacher", "doctor", "name",
    # Places & Time (26-33)
    "home", "school", "hospital", "today", "tomorrow", "time", "where", "what",
    # Actions & States (34-42)
    "go", "come", "learn", "read", "write", "work", "play", "stop", "wait",
    # Emotions & Descriptors (43-49)
    "good", "bad", "happy", "sad", "beautiful", "hot", "cold"
]

VOCAB_SIZE = len(ISL_VOCABULARY)
WORD_TO_ID = {word: i for i, word in enumerate(ISL_VOCABULARY)}
ID_TO_WORD = {i: word for i, word in enumerate(ISL_VOCABULARY)}

# ─── 43-Class ISL Sign Vocabulary ────────────────────────────────
# Canonical 43 sign/action classes spanning Greetings, Needs, Family, Places, and Actions (indices 0-42)
ISL_43_VOCABULARY = ISL_VOCABULARY[:43]
VOCAB_43_SIZE = len(ISL_43_VOCABULARY)
WORD_TO_ID_43 = {word: i for i, word in enumerate(ISL_43_VOCABULARY)}
ID_TO_WORD_43 = {i: word for i, word in enumerate(ISL_43_VOCABULARY)}

# Active vocabulary mapped to the full 43-sign vocabulary
ACTIVE_VOCABULARY = ISL_43_VOCABULARY
ACTIVE_VOCAB_SIZE = VOCAB_43_SIZE
ACTIVE_WORD_TO_ID = WORD_TO_ID_43
ACTIVE_ID_TO_WORD = ID_TO_WORD_43

# Pipeline hyperparameters
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Motion Energy Parameters
MOTION_WINDOW_SIZE = 15
LAMBDA_SIGMA = 0.5   # τ = μ_E + λ * σ_E
IDLE_ENERGY_THRESHOLD = 0.015  # Energy threshold to declare idle state

# Tokenizer Parameters
TOKEN_DIM = 6        # [Hx, Hy, Mx, My, Rx, Ry]

# Temporal Memory Transformer Hyperparameters
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LEN = 30
DROPOUT = 0.1

# Early Decision & Cooldown Engine
CONFIDENCE_THRESHOLD = 0.80
SUSTAINED_FRAMES = 2
COOLDOWN_FRAMES = 10

# Dynamic Sentence Grammar & Translation Setup
ISL_GRAMMAR_RULES = {
    # ISL SOV / Keyword ordering -> Natural English SVO translation
    "I WATER WANT": "I want water",
    "ME WATER GIVE": "Please give me water",
    "YOU NAME WHAT": "What is your name?",
    "MY NAME": "My name is",
    "I FOOD EAT": "I want to eat food",
    "I SCHOOL GO": "I am going to school",
    "I HOME GO": "I am going home",
    "I HELP NEED": "I need help",
    "I HAPPY": "I am happy",
    "I SAD": "I am sad",
    "MOTHER HOME": "Mother is at home",
    "FATHER WORK": "Father is at work",
    "TODAY SCHOOL GO": "Going to school today",
    "MEDICINE NEED": "Need medicine",
}

# English to Tamil Semantic Mapping Dictionary
TAMIL_VOCAB_MAP = {
    "hello": "வணக்கம் (Vanakkam)",
    "thank_you": "நன்றி (Nandri)",
    "welcome": "நல்வரவு (Nalvaravu)",
    "goodbye": "சென்று வருகிறேன் (Sentru varugiren)",
    "please": "தயவுசெய்து (Thayavuseythu)",
    "sorry": "மன்னிக்கவும் (Mannikkavum)",
    "yes": "ஆம் (Aam)",
    "no": "இல்லை (Illai)",
    "water": "தண்ணீர் (Thanneer)",
    "eat": "சாப்பிடு (Saappidu)",
    "drink": "குடி (Kudi)",
    "food": "உணவு (Unavu)",
    "help": "உதவி (Uthavi)",
    "need": "வேண்டும் (Vaendum)",
    "want": "விரும்புகிறேன் (Virumbugiraen)",
    "sleep": "தூக்கம் (Thookkam)",
    "toilet": "கழிப்பறை (Kazhipparai)",
    "medicine": "மருந்து (Marundhu)",
    "father": "அப்பா (Appa)",
    "mother": "அம்மா (Amma)",
    "brother": "சகோதரன் (Sagodharan)",
    "sister": "சகோதரி (Sagodhari)",
    "friend": "நண்பன் (Nanban)",
    "teacher": "ஆசிரியர் (Aasiriyar)",
    "doctor": "மருத்துவர் (Maruthuvar)",
    "name": "பெயர் (Peyar)",
    "home": "வீடு (Veedu)",
    "school": "பள்ளி (Palli)",
    "hospital": "மருத்துவமனை (Maruthuvamanai)",
    "today": "இன்று (Indru)",
    "tomorrow": "நாளை (Naalai)",
    "time": "நேரம் (Neram)",
    "where": "எங்கே (Engae)",
    "what": "என்ன (Enna)",
    "go": "செல் (Sel)",
    "come": "வா (Vaa)",
    "learn": "கற்றுக்கொள் (Katrukkol)",
    "read": "படி (Padi)",
    "write": "எழுது (Ezhuthu)",
    "work": "வேலை (Vaelai)",
    "play": "விளையாடு (Vilaiyaadu)",
    "stop": "நில் (Nil)",
    "wait": "காத்திரு (Kaathiru)",
    "good": "நல்லது (Nallathu)",
    "bad": "கெட்டது (Kettathu)",
    "happy": "மகிழ்ச்சி (Magizhchi)",
    "sad": "வருத்தம் (Varutham)",
    "beautiful": "அழகான (Azhagaana)",
    "hot": "சூடான (Sudaana)",
    "cold": "குளிர்ந்த (Kulirndha)"
}
