import numpy as np
from src.cnn_gru_model import CNNGRUInferenceEngine
from config import CLASS_NAMES
from collections import Counter

engine = CNNGRUInferenceEngine()
test_x = np.load('dataset/test/X.npy')
test_y = np.load('dataset/test/y.npy')

correct = 0
preds = []
for i in range(len(test_x)):
    res = engine.predict_sequence(test_x[i])
    pred_word = res['word']
    true_word = CLASS_NAMES[test_y[i]]
    preds.append(pred_word)
    if pred_word == true_word:
        correct += 1
    top3 = [(k, v) for k, v in list(res['probabilities'].items())[:3]]
    conf = res['confidence']
    print(f"{i:2d} True: {true_word:<12} Pred: {pred_word:<12} Conf: {conf:.3f} Top3: {top3}")

print(f"\nTotal correct: {correct}/{len(test_x)} ({correct/len(test_x)*100:.1f}%)")
print("Prediction distribution:", Counter(preds))
