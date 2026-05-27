import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/Bharat_Phish_Net_v1.csv")


def load_dataset():
    """
    Load the curated Bharat-Phish-Net dataset.
    """
    df = pd.read_csv(DATA_PATH)

    # Basic safety cleaning
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)

    return df


def create_threat_label(df):
    """
    Layer 1:
    0 -> Legitimate
    1 -> Threat (Spam + Scam + Phishing)
    """
    df["threat_label"] = df["intent_label"].apply(lambda x: 0 if x == 0 else 1)
    return df


def create_layer2_label(df):
    """
    Layer 2A:
    0 -> Spam (Promotional)
    1 -> Scam (Generic Scam + Phishing)
    Only applied on threat messages
    """
    threat_df = df[df["threat_label"] == 1].copy()

    def map_label(x):
        if x == 1:
            return 0  # Spam
        elif x in [2, 3]:
            return 1  # Scam
        else:
            return None

    threat_df["layer2_label"] = threat_df["intent_label"].apply(map_label)

    return threat_df


def get_layer1_data():
    """
    Returns X, y for Threat vs Non-Threat
    """
    df = load_dataset()
    df = create_threat_label(df)

    X = df["clean_text"]
    y = df["threat_label"]

    return X, y


def get_layer2_data():
    """
    Returns X, y for Spam vs Scam
    """
    df = load_dataset()
    df = create_threat_label(df)
    threat_df = create_layer2_label(df)

    X = threat_df["clean_text"]
    y = threat_df["layer2_label"]

    return X, y
