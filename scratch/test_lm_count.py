import os
import sys
import numpy as np
from pathlib import Path

lm_dir = Path("dataset/landmarks")
classes = sorted([d.name for d in lm_dir.iterdir() if d.is_dir()])

print(f"Testing 12D token generation on {len(classes)} classes in dataset/landmarks:")

total_files = 0
shape_counts = {}

for c in classes:
    files = list((lm_dir / c).glob("**/*.npz"))
    total_files += len(files)
    
print(f"Total landmark files to process: {total_files}")
