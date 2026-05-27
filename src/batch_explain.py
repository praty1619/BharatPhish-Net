"""
Phase 3 — Step 4: Batch SHAP Analysis
======================================
Runs explain_message() across all scam messages in the dataset.
Saves a structured CSV for thesis analysis and figures.

Usage:
    py -m src.batch_explain
"""

import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.preprocess import get_layer2_data
from src.explain_shap import explain_message

OUTPUT_PATH = "outputs/analysis/batch_shap_results.csv"


def run_batch(sample_size=200):
    print("Loading scam messages...")
    X, y = get_layer2_data()

    scam_texts = X[y == 1].sample(min(sample_size, (y == 1).sum()), random_state=42)

    rows = []
    print(f"Running SHAP on {len(scam_texts)} scam messages...")

    for text in tqdm(scam_texts, desc="Batch SHAP"):
        try:
            result = explain_message(text, top_n=5)

            # Flatten for CSV
            top_words = [w["word"] for w in result["top_shap_words"] if w["shap_score"] > 0]
            clusters  = list(result["cluster_summary"].keys())

            rows.append({
                "text":            text[:100],
                "prediction":      result["prediction"],
                "confidence":      result["confidence"],
                "top_scam_words":  ", ".join(top_words[:5]),
                "active_clusters": ", ".join(clusters),
                "n_clusters":      len(clusters),
            })
        except Exception as e:
            print(f"[SKIP] Error on message: {e}")

    Path("outputs/analysis").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nBatch analysis saved → {OUTPUT_PATH}")
    print(f"\nCluster frequency across all scam messages:")
    all_clusters = ", ".join(df["active_clusters"]).split(", ")
    from collections import Counter
    cluster_counts = Counter([c.strip() for c in all_clusters if c.strip()])
    for cluster, count in cluster_counts.most_common():
        print(f"  {cluster:<15}: {count}")

    return df


if __name__ == "__main__":
    run_batch()
