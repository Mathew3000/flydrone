"""
Run once after downloading the two MaleCNS bulk files into data/raw/ (see
docs/connectome-data-access.md) -- prints columns and a few rows of each so
we can confirm exact column names before writing a loader against them
(rather than guessing and finding out later that it silently used the wrong
column).
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ANNOT = os.path.join(DATA_DIR, "body-annotations-male-cns-v1.0-minconf-0.5.feather")
WEIGHTS = os.path.join(DATA_DIR, "connectome-weights-male-cns-v1.0-minconf-0.5.feather")

for label, path in [("annotations", ANNOT), ("weights", WEIGHTS)]:
    if not os.path.isfile(path):
        print(f"[MISSING] {label}: {path}")
        continue
    df = pd.read_feather(path)
    print(f"\n=== {label} ({path}) ===")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(3).to_string())
