"""
Phase 3 — Step 2: Auto-Cluster Concepts into Semantic Groups
=============================================================
Takes learned_concepts.csv (from extract_concepts.py) and groups words
into semantic clusters using word-boundary regex matching.

Fix over previous version:
  - OLD: substring match caused false positives
      'holiday' → IDENTITY (because 'id' inside 'holiday')
      'video'   → IDENTITY (because 'id' inside 'video')
      'rs'      → IDENTITY (because 'rs' inside 'personal')

  - NEW: regex word-boundary match — seed must appear as a whole word
      'holiday' → OTHER   ✓
      'video'   → OTHER   ✓
      'kyc'     → IDENTITY ✓
      'blocked' → URGENCY  ✓

Usage:
    py -m src.cluster_concepts
"""

import re
import pandas as pd
from pathlib import Path

INPUT_PATH  = "outputs/models/learned_concepts.csv"
OUTPUT_PATH = "outputs/models/concept_clusters.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Seeds define CATEGORY NAMES only — not the vocabulary.
# Vocabulary comes entirely from learned_concepts.csv (model-discovered).
# Expanded with India-specific terms from augmented dataset.
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_SEEDS = {

    "FINANCIAL":  ["bank", "account", "upi", "payment", "loan", "credit", "debit",
                   "wallet", "money", "fund", "transfer", "amount", "cash", "rupee",
                   "pay", "transaction", "balance", "finance", "neft", "imps",
                   "atm", "cheque", "savings", "deposit", "withdraw", "bill",
                   "recharge", "electricity", "disconnected", "dues"],

    "URGENCY":    ["urgent", "immediately", "block", "blocked", "suspend", "suspended",
                   "expire", "expired", "deadline", "warning", "alert", "notice",
                   "required", "limited", "temporary", "freeze", "frozen",
                   "deactivate", "deactivated", "disconnected", "hours", "tonight",
                   "midnight", "today", "last", "final", "action"],

    "IDENTITY":   ["aadhaar", "kyc", "pan", "uidai", "identity", "detail",
                   "document", "card", "passport", "licence", "voter",
                   "information", "data", "profile", "biometric", "ekyc",
                   "epfo", "esic", "ckyc", "sim", "mobile number"],

    "ACTION":     ["verify", "verification", "click", "update", "submit", "login",
                   "confirm", "enter", "provide", "send", "share", "call",
                   "download", "install", "open", "link", "url", "visit",
                   "register", "activate", "claim", "apply", "contact", "reply"],

    "REWARD":     ["free", "offer", "win", "won", "prize", "reward", "gift",
                   "lucky", "congratulation", "selected", "bonus", "discount",
                   "cashback", "scheme", "lottery", "jackpot", "awarded",
                   "receive", "draw", "eligible", "approved", "refund"],

    "AUTHORITY":  ["government", "irdai", "rbi", "sebi", "trai", "npci", "uidai",
                   "police", "court", "official", "ministry", "authority",
                   "department", "helpline", "customer", "support", "team",
                   "epfo", "sbi", "hdfc", "icici", "axis", "kotak",
                   "airtel", "jio", "bsnl", "bescom", "msedcl"],

    "CREDENTIAL": ["otp", "password", "pin", "cvv", "code", "token", "secret",
                   "credential", "username", "access", "passcode", "mpin",
                   "security", "authenticate", "2fa"],
}


def match_cluster(word: str, seeds: dict) -> str:
    """
    Assign word to a cluster using regex word-boundary matching.
    Seed must appear as a complete word — not as a substring inside another word.

    Examples:
        'kyc'     matches seed 'kyc'     → IDENTITY  ✓
        'holiday' does NOT match 'id'    → OTHER     ✓  (was false positive before)
        'blocked' matches seed 'blocked' → URGENCY   ✓
        'bank account' matches 'bank'    → FINANCIAL ✓
    """
    word_lower = word.lower().strip()

    for cluster, seed_list in seeds.items():
        for seed in seed_list:
            # Exact full match
            if seed == word_lower:
                return cluster
            # Seed appears as a whole word inside multi-word phrases
            pattern = r'(?<![a-z])' + re.escape(seed) + r'(?![a-z])'
            if re.search(pattern, word_lower):
                return cluster

    return "OTHER"


def cluster_concepts():
    print(f"Loading concepts from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)

    df["cluster"] = df["word"].apply(lambda w: match_cluster(w, CLUSTER_SEEDS))

    print("\nCluster distribution:")
    print(df["cluster"].value_counts().to_string())

    print("\nWords per cluster:")
    for cluster in list(CLUSTER_SEEDS.keys()) + ["OTHER"]:
        words = df[df["cluster"] == cluster]["word"].tolist()
        preview = words[:8]
        print(f"  {cluster:12s} ({len(words):2d} words): {preview}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nClustered vocabulary saved → {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    cluster_concepts()