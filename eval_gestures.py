import os
import glob
import numpy as np
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.no_binary_model import NOBinaryInferenceEngine
from src.build_real_split import resample_tokens

def evaluate_models():
    cnn = CNNGRUInferenceEngine()
    bin_no = NOBinaryInferenceEngine()

    classes = ['no', 'hello', 'yes', 'thank_you', 'please', 'stop']
    print("\n" + "="*85)
    print(f"{'CLASS':<12} | {'FILE':<40} | {'CNN-GRU':<15} {'CONF':<8} | {'BIN NO':<8}")
    print("="*85)

    for c in classes:
        files = glob.glob(f"dataset/tokens/{c}/*/*.npz")
        for f in files[:3]:
            toks = np.load(f)['tokens']
            if len(toks) != 25:
                toks = resample_tokens(toks, 25)
            p = cnn.predict_sequence(toks)
            b = bin_no.predict_sequence(toks)
            base = os.path.basename(f)
            w = p['word']
            conf = p['confidence']
            no_p = b['no_probability']
            print(f"{c:<12} | {base:<40} | {w:<15} {conf:<8.2f} | {no_p:<8.2f}")

if __name__ == "__main__":
    evaluate_models()
