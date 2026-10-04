import os
import unittest
import glob
import numpy as np
from fastapi.testclient import TestClient
from app import app
from src.build_real_split import resample_tokens

class TestGatedDecisionSystem(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _feed_gesture(self, token_file):
        toks = np.load(token_file)['tokens']
        if len(toks) != 25:
            toks = resample_tokens(toks, 25)

        self.client.post("/api/clear_sentence")

        responses = []
        for i in range(len(toks)):
            tok = toks[i].copy()
            # Set non-zero velocity during gesture execution
            if abs(tok[2]) < 0.01 and abs(tok[3]) < 0.01:
                tok[2] = 0.015
            payload = {
                "token": tok.tolist(),
                "frame_id": i + 1,
                "timestamp_ms": 1000.0 + i * 66.0,
                "has_hand": True,
                "hand_center": [float(tok[0]), float(tok[1])],
                "shoulder_center": [0.5, 0.35]
            }
            res = self.client.post("/api/process_token", json=payload)
            self.assertEqual(res.status_code, 200)
            responses.append(res.json())

        # Send hold frame after gesture
        last_tok = toks[-1].copy()
        last_tok[2] = 0.0
        last_tok[3] = 0.0
        payload = {
            "token": last_tok.tolist(),
            "frame_id": len(toks) + 1,
            "timestamp_ms": 1000.0 + (len(toks) + 1) * 66.0,
            "has_hand": True,
            "hand_center": [float(last_tok[0]), float(last_tok[1])],
            "shoulder_center": [0.5, 0.35]
        }
        res = self.client.post("/api/process_token", json=payload)
        self.assertEqual(res.status_code, 200)
        responses.append(res.json())

        # Select the response containing the evaluation decision
        valid = [r for r in responses if r.get("final_class") not in ["--", "COLLECTING GESTURE...", None]]
        if valid:
            return valid[-1]
        return responses[-1]

    def test_no_gesture(self):
        no_file = "dataset/tokens/no/signNO/noFAST.npz"
        res = self._feed_gesture(no_file)
        print("\n" + "="*50)
        print("TEST 1: NO GESTURE")
        print(f"File:    {no_file}")
        print(f"Primary: {res.get('primary_class')} ({res.get('primary_confidence')})")
        print(f"NO Prob: {res.get('no_probability')} | Confirmed: {res.get('binary_no_confirmed')}")
        print(f"Final:   {res.get('final_class')}")
        print("="*50)

        self.assertEqual(res.get("final_class"), "no")

    def test_hello_gesture(self):
        hello_file = "dataset/tokens/hello/hf_real/hello__INCLUDE__00569__MVI_0032.npz"
        res = self._feed_gesture(hello_file)
        print("\n" + "="*50)
        print("TEST 2: HELLO GESTURE")
        print(f"File:    {hello_file}")
        print(f"Primary: {res.get('primary_class')} ({res.get('primary_confidence')})")
        print(f"NO Prob: {res.get('no_probability')} | Confirmed: {res.get('binary_no_confirmed')}")
        print(f"Final:   {res.get('final_class')}")
        print("="*50)

        self.assertEqual(res.get("final_class"), "hello")
        self.assertFalse(res.get("binary_no_confirmed"))

    def test_yes_gesture(self):
        yes_file = "dataset/tokens/yes/hf_real/yes__CISLR__00033__LAT-u9Ww5ZU.npz"
        res = self._feed_gesture(yes_file)
        print("\n" + "="*50)
        print("TEST 3: YES GESTURE")
        print(f"File:    {yes_file}")
        print(f"Primary: {res.get('primary_class')} ({res.get('primary_confidence')})")
        print(f"NO Prob: {res.get('no_probability')} | Confirmed: {res.get('binary_no_confirmed')}")
        print(f"Final:   {res.get('final_class')}")
        print("="*50)

        # Critical: Must NOT be overwritten to NO
        self.assertNotEqual(res.get("final_class"), "no")
        self.assertFalse(res.get("binary_no_confirmed"))

    def test_thank_you_gesture(self):
        ty_file = "dataset/tokens/thank_you/hf_real/thank_you__CISLR__00027__IRmlHHtW_Q0_1.npz"
        res = self._feed_gesture(ty_file)
        print("\n" + "="*50)
        print("TEST 4: THANK YOU GESTURE")
        print(f"File:    {ty_file}")
        print(f"Primary: {res.get('primary_class')} ({res.get('primary_confidence')})")
        print(f"NO Prob: {res.get('no_probability')} | Confirmed: {res.get('binary_no_confirmed')}")
        print(f"Final:   {res.get('final_class')}")
        print("="*50)

        # Critical: Must NOT be overwritten to NO
        self.assertNotEqual(res.get("final_class"), "no")
        self.assertEqual(res.get("final_class"), "thank_you")
        self.assertFalse(res.get("binary_no_confirmed"))

    def test_please_gesture(self):
        please_file = "dataset/tokens/please/hf_real/please__ISL500__00120__Please__session106__clip004.npz"
        res = self._feed_gesture(please_file)
        print("\n" + "="*50)
        print("TEST 5: PLEASE GESTURE")
        print(f"File:    {please_file}")
        print(f"Primary: {res.get('primary_class')} ({res.get('primary_confidence')})")
        print(f"NO Prob: {res.get('no_probability')} | Confirmed: {res.get('binary_no_confirmed')}")
        print(f"Final:   {res.get('final_class')}")
        print("="*50)

        self.assertEqual(res.get("final_class"), "please")
        self.assertFalse(res.get("binary_no_confirmed"))

    def test_stop_gesture(self):
        stop_file = "dataset/tokens/stop/hf_real/stop__CISLR__00021__FEi50jcTxxI_1.npz"
        res = self._feed_gesture(stop_file)
        print("\n" + "="*50)
        print("TEST 6: STOP GESTURE")
        print(f"File:    {stop_file}")
        print(f"Primary: {res.get('primary_class')} ({res.get('primary_confidence')})")
        print(f"NO Prob: {res.get('no_probability')} | Confirmed: {res.get('binary_no_confirmed')}")
        print(f"Final:   {res.get('final_class')}")
        print("="*50)

        # Critical: Must NOT be overwritten to NO
        self.assertNotEqual(res.get("final_class"), "no")
        self.assertEqual(res.get("final_class"), "stop")
        self.assertFalse(res.get("binary_no_confirmed"))

if __name__ == "__main__":
    unittest.main()
