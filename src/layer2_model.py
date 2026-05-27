from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

from src.preprocess import get_layer2_data


MODEL_PATH = "outputs/models/layer2_model.pkl"
VEC_PATH = "outputs/models/layer2_vectorizer.pkl"


def train_layer2():
    print("Loading Layer-2 data...")
    X, y = get_layer2_data()

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

    print("Training Layer-2 model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_vec, y_train)

    print("\nEvaluation:\n")
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, target_names=["Spam", "Scam"]))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test_vec,
        y_test,
        display_labels=["Spam", "Scam"],
        cmap="Purples"
    )
    plt.title("Layer 2: Spam vs Scam")
    plt.savefig("outputs/figures/layer2_confusion_matrix.png")
    plt.close()

    print("\nSaving Layer-2 model...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf, VEC_PATH)

    print("Layer 2 model saved successfully!")


if __name__ == "__main__":
    train_layer2()
