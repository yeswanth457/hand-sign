import urllib.request
import json
import glob
import numpy as np

def run_test():
    welcome_files = glob.glob('dataset/tokens/welcome/*/*.npz')
    if not welcome_files:
        print('FAIL: No welcome token files found.')
        return

    print(f"Total welcome token recordings found: {len(welcome_files)}")

    # Test sample recordings
    for sample_idx in range(min(3, len(welcome_files))):
        req_clear = urllib.request.Request(
            'http://127.0.0.1:8000/api/clear_sentence',
            data=b'',
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        urllib.request.urlopen(req_clear)

        filepath = welcome_files[sample_idx]
        tokens = np.load(filepath)['tokens']
        print(f"\n--- Testing Sample {sample_idx+1}: {filepath.split('/')[-1]} ({len(tokens)} frames) ---")

        final_res = None
        for i in range(min(25, len(tokens))):
            tok = tokens[i]
            payload = {
                'token': tok.tolist(),
                'frame_id': (sample_idx + 1) * 1000 + i,
                'timestamp_ms': 10000.0 + i * 66.0,
                'has_hand': True,
                'hand_center': [float(tok[0]), float(tok[1])],
                'shoulder_center': [0.5, 0.35]
            }
            req = urllib.request.Request(
                'http://127.0.0.1:8000/api/process_token',
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as res:
                final_res = json.loads(res.read().decode())

        pred = final_res['prediction']['word']
        conf = final_res['prediction']['confidence']
        state = final_res['early_decision']['state']
        accepted = final_res['early_decision']['accepted']
        text = final_res['translation']['display_text']
        print(f"Outcome: Pred='{pred}' | Conf={conf*100:.2f}% | State={state} | English Output='{text}'")

if __name__ == '__main__':
    run_test()
