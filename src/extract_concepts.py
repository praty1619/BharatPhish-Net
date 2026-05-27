"""
Phase 3 — Step 1: Global SHAP Concept Extraction
=================================================
Runs SHAP over all scam messages in the dataset.
Collects the most influential words (model-discovered, not manually defined).
Saves to outputs/models/learned_concepts.csv

Usage:
    py -m src.extract_concepts
"""

import joblib
import shap
import pandas as pd
from collections import Counter
from tqdm import tqdm
from pathlib import Path

from src.preprocess import get_layer2_data

MODEL_PATH  = "outputs/models/layer2_model.pkl"
VEC_PATH    = "outputs/models/layer2_vectorizer.pkl"
OUTPUT_PATH = "outputs/models/learned_concepts.csv"


def load_artifacts():
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer


def build_background(vectorizer, X, n=500):
    """
    Use a real sample of training data as SHAP background.
    This is critical — a dummy string produces garbage SHAP values.
    """
    sample = X.sample(min(n, len(X)), random_state=42)
    return vectorizer.transform(sample)


def extract_global_shap_words(sample_size=400, top_per_message=8):
    print("Loading Layer-2 data...")
    X, y = get_layer2_data()

    model, vectorizer = load_artifacts()
    feature_names     = vectorizer.get_feature_names_out()

    print(f"Building SHAP explainer with {min(500, len(X))} background samples...")
    background = build_background(vectorizer, X)
    explainer  = shap.LinearExplainer(model, background)

    # Only process scam messages (label == 1)
    scam_mask  = y == 1
    scam_texts = X[scam_mask].sample(min(sample_size, scam_mask.sum()), random_state=42)

    print(f"Running SHAP on {len(scam_texts)} scam messages...")
    word_counter = Counter()

    for text in tqdm(scam_texts, desc="SHAP analysis"):
        vec         = vectorizer.transform([text])
        shap_values = explainer(vec)
        values      = shap_values.values[0]   # shape: (n_features,)

        # Only consider words that actually appear in this message
        nonzero_idx    = vec.nonzero()[1]
        present        = [(i, values[i]) for i in nonzero_idx if values[i] > 0]
        present_sorted = sorted(present, key=lambda x: x[1], reverse=True)

        top_words = [feature_names[i] for i, _ in present_sorted[:top_per_message]]
        word_counter.update(top_words)

    return word_counter


def save_concepts(counter, top_n=80):
    Path("outputs/models").mkdir(parents=True, exist_ok=True)

    top_words = counter.most_common(top_n)
    df = pd.DataFrame(top_words, columns=["word", "frequency"])
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTop 20 model-discovered scam vocabulary:")
    print(df.head(20).to_string(index=False))
    print(f"\nFull list saved → {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    counter = extract_global_shap_words()
    save_concepts(counter)