import numpy as np
import joblib
import mediapipe as mp
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# model = joblib.load("gesture/model/gesture_recognition_model.pkl")
# label_encoder = joblib.load("gesture/model/label_encoder.pkl")

model = joblib.load(BASE_DIR / "gesture/model/gesture_recognition_model.pkl")
label_encoder = joblib.load(BASE_DIR / "gesture/model/label_encoder.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)

def extract_features(all_hands):
    if len(all_hands) == 0:
        return np.zeros(144)

    fingertips = [4,8,12,16,20]

    def one_hand(hand):
        pts = np.array([[p.x,p.y,p.z] for p in hand])
        rel = pts - pts[0]
        dist = [np.linalg.norm(pts[i]-pts[0]) for i in fingertips]
        vecs = [(pts[i]-pts[0]) for i in fingertips]
        ang = []
        for i in range(len(vecs)-1):
            cos = np.dot(vecs[i],vecs[i+1]) / (
                np.linalg.norm(vecs[i])*np.linalg.norm(vecs[i+1])
            )
            ang.append(np.arccos(np.clip(cos,-1,1)))
        return rel.flatten(), dist, ang

    h1_rel, h1_dist, h1_ang = one_hand(all_hands[0])

    if len(all_hands) > 1:
        h2_rel, h2_dist, h2_ang = one_hand(all_hands[1])
    else:
        h2_rel = np.zeros_like(h1_rel)
        h2_dist = np.zeros_like(h1_dist)
        h2_ang = np.zeros_like(h1_ang)

    return np.concatenate([
        h1_rel, h2_rel,
        h1_dist, h2_dist,
        h1_ang, h2_ang
    ])
