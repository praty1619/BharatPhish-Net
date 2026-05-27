"""
Phase 3 — Step 3: Per-Message SHAP Explanation
===============================================
Takes a single SMS/email message and produces a structured explanation:
  - Model prediction + confidence
  - Top SHAP words present in the message
  - Which semantic clusters those words belong to
  - Structured output ready to pass into Phase 4 (LLM translator)

Usage:
    py -m src.explain_shap
    or import explain_message() in other modules
"""

import joblib
import shap
import pandas as pd
from pathlib import Path

from src.preprocess import get_layer2_data

MODEL_PATH    = "outputs/models/layer2_model.pkl"
VEC_PATH      = "outputs/models/layer2_vectorizer.pkl"
CONCEPTS_PATH = "outputs/models/concept_clusters.csv"


def load_artifacts():
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer


def load_concept_map() -> dict:
    """
    Returns {word: cluster} mapping from data-driven vocabulary.
    Falls back to empty dict if file doesn't exist yet.
    """
    path = Path(CONCEPTS_PATH)
    if not path.exists():
        print(f"[WARN] {CONCEPTS_PATH} not found. Run extract_concepts.py first.")
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["word"], df["cluster"]))


def build_background(vectorizer, X, n=500):
    sample = X.sample(min(n, len(X)), random_state=42)
    return vectorizer.transform(sample)


def explain_message(text: str, top_n: int = 10) -> dict:
    """
    Produces a structured explanation for a single message.

    Returns
    -------
    dict with keys:
        prediction       : "Scam" or "Spam"
        confidence       : float (0-1)
        top_shap_words   : list of {word, shap_score, cluster}
        cluster_summary  : dict {cluster: [words]}
        raw_text         : original input
    """
    model, vectorizer = load_artifacts()
    concept_map       = load_concept_map()

    # ── Get prediction ──────────────────────────────────────────────────────
    X_vec      = vectorizer.transform([text])
    pred_label = model.predict(X_vec)[0]
    pred_proba = model.predict_proba(X_vec)[0]
    confidence = round(pred_proba.max(), 4)
    prediction = "Scam" if pred_label == 1 else "Spam"

    # ── Build SHAP explainer with real background ───────────────────────────
    X_all, _ = get_layer2_data()
    background = build_background(vectorizer, X_all)
    explainer  = shap.LinearExplainer(model, background)

    shap_values   = explainer(X_vec)
    feature_names = vectorizer.get_feature_names_out()
    values        = shap_values.values[0]   # shape: (n_features,)

    # ── Only look at words present in THIS message ──────────────────────────
    nonzero_idx = X_vec.nonzero()[1]

    word_scores = []
    for i in nonzero_idx:
        word       = feature_names[i]
        shap_score = float(values[i])
        cluster    = concept_map.get(word, "OTHER")
        word_scores.append({
            "word":       word,
            "shap_score": round(shap_score, 5),
            "cluster":    cluster,
        })

    # Sort by absolute SHAP — most influential first
    word_scores.sort(key=lambda x: abs(x["shap_score"]), reverse=True)
    top_words = word_scores[:top_n]

    # ── Build cluster summary (only positive/scam-pushing words) ────────────
    cluster_summary = {}
    for item in top_words:
        if item["shap_score"] > 0:                        # pushes toward Scam
            c = item["cluster"]
            cluster_summary.setdefault(c, []).append(item["word"])

    return {
        "raw_text":       text,
        "prediction":     prediction,
        "confidence":     confidence,
        "top_shap_words": top_words,
        "cluster_summary": cluster_summary,
    }


def pretty_print(result: dict):
    print(f"\n{'─'*55}")
    print(f"  Input     : {result['raw_text'][:80]}")
    print(f"  Prediction: {result['prediction']}  ({result['confidence']*100:.1f}% confidence)")
    print(f"{'─'*55}")

    print("\nTop SHAP words (present in message):")
    print(f"  {'WORD':<20} {'SHAP SCORE':>12}  {'CLUSTER'}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*12}")
    for item in result["top_shap_words"]:
        direction = "→ Scam" if item["shap_score"] > 0 else "→ Spam"
        print(f"  {item['word']:<20} {item['shap_score']:>+10.5f}  {item['cluster']} ({direction})")

    if result["cluster_summary"]:
        print("\nCluster Summary (scam-pushing signals):")
        for cluster, words in result["cluster_summary"].items():
            print(f"  [{cluster}] → {', '.join(words)}")
    print()


if __name__ == "__main__":
    test_messages = [
        "Your bank account will be blocked. Verify your OTP immediately.",
        "Dear customer, your KYC details have expired. Update immediately to avoid service suspension.",
        "Congratulations! You have won a free iPhone. Click here to claim your prize now.",
        "Your Aadhaar card is linked to suspicious activity. Call 9876543210 to verify your PAN details.",
    ]

    print("=" * 55)
    print("  Bharat-Phish-Net | Phase 3: SHAP Explanation Demo")
    print("=" * 55)

    choice = input("\nEnter SMS to explain (or press Enter to run demo messages): ").strip()

    if choice:
        result = explain_message(choice)
        pretty_print(result)
    else:
        for msg in test_messages:
            result = explain_message(msg)
            pretty_print(result)