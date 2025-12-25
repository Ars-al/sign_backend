import json
import base64
import cv2
import numpy as np
from channels.generic.websocket import WebsocketConsumer
from .utils import hands, extract_features, model, label_encoder


class GestureConsumer(WebsocketConsumer):
    def connect(self):
        # WebSocket accept
        self.accept()

    def receive(self, text_data):
        data = json.loads(text_data)

        # ---------- IMAGE DECODE ----------
        img_b64 = data.get("image")
        if not img_b64:
            return

        img_bytes = base64.b64decode(img_b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # ---------- MEDIAPIPE ----------
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        all_hands = []
        if results.multi_hand_landmarks:
            for h in results.multi_hand_landmarks:
                all_hands.append(h.landmark)

        # ---------- NO HAND ----------
        if len(all_hands) == 0:
            self.send(json.dumps({
                "gesture": "None",
                "confidence": 0.0
            }))
            return

        # ---------- FEATURE EXTRACTION ----------
        features = extract_features(all_hands).reshape(1, -1)

        # ---------- MODEL PREDICTION ----------
        proba = model.predict_proba(features)[0]
        idx = int(np.argmax(proba))

        gesture_name = label_encoder.inverse_transform([idx])[0]
        confidence = round(float(proba[idx]), 2)

        # ---------- SEND RESULT ----------
        self.send(json.dumps({
            "gesture": gesture_name,
            "confidence": confidence
        }))
