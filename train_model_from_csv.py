import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from sklearn.pipeline import make_pipeline

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
def load_and_preprocess_data(filename):
    df = pd.read_csv(filename)

    if df.isnull().any().any():
        print("Warning: Missing data found. Dropping rows...")
        df = df.dropna()

    X = df.iloc[:, :-1].values     # 126 columns
    y = df.iloc[:, -1].values      # label column

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


# -------------------------------------------------------
# FEATURE EXTRACTION FOR DOUBLE-HAND DATA
# -------------------------------------------------------
def extract_combined_features(row):
    """
    row = shape (126,) → [hand1(63) + hand2(63)]
    If hand2 is all zeros → treat as single-hand
    """
    row = np.array(row)

    # Split into 2 hands
    h1 = row[:63].reshape(21, 3)
    h2 = row[63:].reshape(21, 3)

    # Detect if second hand exists (if all zeros → no second hand)
    second_hand_exists = not np.all(h2 == 0)

    # ----- Common: Left hand features -----
    h1_rel = h1 - h1[0]
    h1_fingertips = [4, 8, 12, 16, 20]
    h1_dist = [np.linalg.norm(h1[i] - h1[0]) for i in h1_fingertips]

    def angles(hand):
        vecs = [(hand[i] - hand[0]) for i in h1_fingertips]
        angs = []
        for i in range(len(vecs) - 1):
            cos_val = np.dot(vecs[i], vecs[i+1]) / (
                np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i+1])
            )
            angs.append(np.arccos(np.clip(cos_val, -1, 1)))
        return angs

    h1_angles = angles(h1)

    # ----- If ONLY ONE HAND is used -----
    if not second_hand_exists:
        return np.concatenate([
            h1_rel.flatten(),
            np.zeros_like(h1_rel.flatten()),
            np.array(h1_dist),
            np.zeros_like(h1_dist),
            np.array(h1_angles),
            np.zeros_like(h1_angles)
        ])

    # ----- TWO HANDS -----
    h2_rel = h2 - h2[0]
    h2_dist = [np.linalg.norm(h2[i] - h2[0]) for i in h1_fingertips]
    h2_angles = angles(h2)

    return np.concatenate([
        h1_rel.flatten(),
        h2_rel.flatten(),
        np.array(h1_dist),
        np.array(h2_dist),
        np.array(h1_angles),
        np.array(h2_angles)
    ])


# -------------------------------------------------------
# TRAINING MODEL
# -------------------------------------------------------
def train_model(X, y):
    print("Extracting enhanced features...")
    X_enhanced = np.array([extract_combined_features(x) for x in X])

    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42
    )

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight='balanced',
            random_state=42
        )
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, preds))

    return model


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    X, y, label_encoder = load_and_preprocess_data("double_hand_dataset.csv")

    model = train_model(X, y)

    joblib.dump(model, "gesture_recognition_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("\nModel TRAINING complete and saved!")


if __name__ == "__main__":
    main()