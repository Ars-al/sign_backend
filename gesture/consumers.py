# import json
# import base64
# import cv2
# import numpy as np
# from channels.generic.websocket import WebsocketConsumer
# from .utils import hands, extract_features, model, label_encoder


# class GestureConsumer(WebsocketConsumer):
#     def connect(self):
#         # WebSocket accept
#         self.accept()

#     def receive(self, text_data):
#         data = json.loads(text_data)

#         # ---------- IMAGE DECODE ----------
#         img_b64 = data.get("image")
#         if not img_b64:
#             return

#         img_bytes = base64.b64decode(img_b64)
#         arr = np.frombuffer(img_bytes, np.uint8)
#         frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

#         # ---------- MEDIAPIPE ----------
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)

#         all_hands = []
#         if results.multi_hand_landmarks:
#             for h in results.multi_hand_landmarks:
#                 all_hands.append(h.landmark)

#         # ---------- NO HAND ----------
#         if len(all_hands) == 0:
#             self.send(json.dumps({
#                 "gesture": "None",
#                 "confidence": 0.0
#             }))
#             return

#         # ---------- FEATURE EXTRACTION ----------
#         features = extract_features(all_hands).reshape(1, -1)

#         # ---------- MODEL PREDICTION ----------
#         proba = model.predict_proba(features)[0]
#         idx = int(np.argmax(proba))

#         gesture_name = label_encoder.inverse_transform([idx])[0]
#         confidence = round(float(proba[idx]), 2)

#         # ---------- SEND RESULT ----------
#         self.send(json.dumps({
#             "gesture": gesture_name,
#             "confidence": confidence
#         }))



import json
import base64
import numpy as np
from channels.generic.websocket import WebsocketConsumer
from .utils import hands, extract_features, model, label_encoder


class GestureConsumer(WebsocketConsumer):

    def connect(self):
        # WebSocket connection accept
        self.accept()

    def receive(self, text_data):
        # ⚠️ Lazy imports (server crash se bachne ke liye)
        import cv2

        data = json.loads(text_data)

        # ---------- IMAGE DECODE ----------
        # Frontend se base64 image aa rahi hoti hai
        img_b64 = data.get("image")
        if not img_b64:
            return

        # base64 → bytes
        img_bytes = base64.b64decode(img_b64)

        # bytes → numpy array
        arr = np.frombuffer(img_bytes, np.uint8)

        # numpy array → OpenCV image
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # ---------- MEDIAPIPE HAND DETECTION ----------
        # OpenCV BGR image ko RGB mein convert karna zaroori hota hai
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe se hands detect kar rahe hain
        results = hands.process(img_rgb)

        all_hands = []

        # Agar hands detect huay hain
        if results.multi_hand_landmarks:
            for h in results.multi_hand_landmarks:
                all_hands.append(h.landmark)

        # ---------- NO HAND DETECTED ----------
        # Agar frame mein koi hand nahi mila
        if len(all_hands) == 0:
            self.send(json.dumps({
                "gesture": "None",
                "confidence": 0.0
            }))
            return

        # ---------- FEATURE EXTRACTION ----------
        # Hand landmarks ko ML model ke input format mein convert karna
        features = extract_features(all_hands).reshape(1, -1)

        # ---------- MODEL PREDICTION ----------
        # predict_proba har class ki probability deta hai
        proba = model.predict_proba(features)[0]

        # Sab se zyada probability wali class ka index
        idx = int(np.argmax(proba))

        # Gesture ka naam nikalna
        gesture_name = label_encoder.inverse_transform([idx])[0]

        # Confidence (probability) ko round kar rahe hain
        confidence = round(float(proba[idx]), 2)

        # ---------- CONFIDENCE THRESHOLD ----------
        # Agar model ka confidence kam ho
        # to gesture ko ignore kar dete hain
        # Example: 0.75 = 75% confidence
        CONFIDENCE_THRESHOLD = 0.75

        if confidence < CONFIDENCE_THRESHOLD:
            self.send(json.dumps({
                "gesture": "Uncertain",   # low confidence output
                "confidence": confidence
            }))
            return

        # ---------- SEND FINAL RESULT ----------
        # Sirf tab send hoga jab confidence threshold pass kare
        self.send(json.dumps({
            "gesture": gesture_name,
            "confidence": confidence
        }))
