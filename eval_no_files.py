import os
import glob
import numpy as np
from src.cnn_gru_model import CNNGRUInferenceEngine
from src.no_binary_model import NOBinaryInferenceEngine
from src.build_real_split import resample_tokens

def evaluate_all_no_files():
    cnn = CNNGRUInferenceEngine()
    bin_no = NOBinaryInferenceEngine()

    files = glob.glob("dataset/tokens/no/*/*.npz")
    print(f"Total NO token files: {len(files)}")
    print(f"{'FILE':<45} | {'CNN TOP PRED':<15} {'CONF':<8} | {'BIN NO':<8}")
    print("="*85)

    for f in files:
        toks = np.load(f)['tokens']
        if len(toks) != 25:
            toks = resample_tokens(toks, 25)
        # raw probs without threshold cutoff
        p = cnn.predict_sequence(toks)
        b = bin_no.predict_sequence(toks)
        base = os.path.basename(f)
        w = p['word']
        conf = p['confidence']
        no_p = b['no_probability']
        top_probs = p.get('probabilities', {})
        print(f"{base:<45} | {w:<15} {conf:<8.2f} | {no_p:<8.2f} | top: {top_probs}")

if __name__ == "__main__":
    evaluate_all_no_files()
