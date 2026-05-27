from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

from src.preprocess import get_layer1_data


MODEL_PATH = "outputs/models/layer1_model.pkl"
VEC_PATH = "outputs/models/layer1_vectorizer.pkl"


def train_layer1():
    print("Loading data...")
    X, y = get_layer1_data()

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Vectorizing...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    print("Training model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_vec, y_train)

    print("\nEvaluation:\n")
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, target_names=["Non-Threat", "Threat"]))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test_vec,
        y_test,
        display_labels=["Non-Threat", "Threat"],
        cmap="Blues"
    )
    plt.title("Layer 1: Threat Detection")
    plt.savefig("outputs/figures/layer1_confusion_matrix.png")
    plt.close()

    print("\nSaving model...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, VEC_PATH)

    print("Layer 1 model saved successfully!")


if __name__ == "__main__":
    train_layer1()
