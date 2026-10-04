import os
import sys
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.cnn_gru_model import CNNGRUInferenceEngine
from src.no_binary_model import NOBinaryInferenceEngine
from src.build_real_split import resample_tokens

def main():
    cnn_engine = CNNGRUInferenceEngine()
    no_engine = NOBinaryInferenceEngine()

    gestures = ['no', 'friend', 'hello', 'yes']

    print("\n" + "=" * 95)
    print(f"{'TARGET':<10} | {'FILE':<35} | {'21-CLASS':<15} {'CONF':<8} | {'NO_PROB':<8} | {'TOP 3 PROBS'}")
    print("=" * 95)

    for g in gestures:
        pattern = os.path.join(BASE_DIR, 'dataset', 'tokens', g, '**', '*.npz')
        token_files = glob.glob(pattern, recursive=True)
        if not token_files:
            print(f"No files found for {g}")
            continue
        for f in token_files[:5]:
            data = np.load(f)
            toks = data['tokens']
            if len(toks) != 25:
                toks = resample_tokens(toks, 25)

            pred_21 = cnn_engine.predict_sequence(toks)
            pred_no = no_engine.predict_sequence(toks)

            w21 = pred_21.get('word', '--')
            c21 = pred_21.get('confidence', 0.0)
            p_no = pred_no.get('no_probability', 0.0)
            top3 = [(k, v) for k, v in list(pred_21.get('probabilities', {}).items())[:3]]

            base = os.path.basename(f)
            print(f"{g:<10} | {base:<35} | {w21:<15} {c21:<8.4f} | {p_no:<8.4f} | {top3}")

if __name__ == '__main__':
    main()
