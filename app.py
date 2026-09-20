import json
import pickle
from collections import deque
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from hand_features import extract_hand_keypoints, extract_static_features

ROOT = Path(__file__).resolve().parent
SEQUENCE_LENGTH = 30
SVM_THRESHOLD = 0.30
LSTM_THRESHOLD = 0.55
PREDICTION_HOLD_SECONDS = 3.0
SVM_FEATURE_SIZE = 63


def load_models():
    lstm_dir = ROOT / "model" / "lstm"
    svm_dir = ROOT / "model" / "svm"
    lstm = tf.keras.models.load_model(lstm_dir / "action_model.h5")
    svm = pickle.loads((svm_dir / "svm_model.pkl").read_bytes())
    lstm_labels = np.array(json.loads((lstm_dir / "labels.json").read_text(encoding="utf-8")))
    svm_labels = np.array(json.loads((svm_dir / "labels.json").read_text(encoding="utf-8")))
    return lstm, lstm_labels, svm, svm_labels

def predict_lstm(model, sample, labels):
    probabilities = model.predict(sample, verbose=0)[0]
    best = int(np.argmax(probabilities))
    return (labels[best], probabilities[best]) if probabilities[best] >= LSTM_THRESHOLD else ("...", probabilities[best])

def predict_svm(model, features, labels):
    probabilities = model.predict_proba(np.expand_dims(features, 0))[0]
    best = int(np.argmax(probabilities))
    return (labels[best], probabilities[best]) if probabilities[best] >= SVM_THRESHOLD else ("...", probabilities[best])

def main():
    lstm, lstm_labels, svm, svm_labels = load_models()
    mode, result = "LSTM", "..."
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    last_result, last_confidence, last_prediction_time = "...", 0.0, 0.0
    svm_feature_size = getattr(svm, "n_features_in_", None)
    svm_needs_retrain = svm_feature_size != SVM_FEATURE_SIZE
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không thể mở camera.")
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for landmarks in results.multi_hand_landmarks or []:
                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            has_hand = bool(results.multi_hand_landmarks)
            features = extract_hand_keypoints(results)
            if mode == "SVM" and has_hand and not svm_needs_retrain:
                static_features = extract_static_features(results)
                if static_features is not None:
                    result, confidence = predict_svm(svm, static_features, svm_labels)
                else:
                    result, confidence = "...", 0.0
            elif mode == "LSTM" and has_hand:
                sequence.append(features)
                if len(sequence) == SEQUENCE_LENGTH:
                    result, confidence = predict_lstm(lstm, np.expand_dims(np.array(sequence), 0), lstm_labels)
                else:
                    confidence = 0.0
            else:
                result, confidence = "...", 0.0
            if result != "...":
                last_result = result
                last_confidence = confidence
                last_prediction_time = cv2.getTickCount() / cv2.getTickFrequency()
            now = cv2.getTickCount() / cv2.getTickFrequency()
            show_saved_result = now - last_prediction_time <= PREDICTION_HOLD_SECONDS

            title = "LSTM - cu chi dong" if mode == "LSTM" else "SVM - cu chi tinh"
            cv2.putText(frame, f"Mode: {title}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
            cv2.putText(frame, "M: doi che do | Q: thoat", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if mode == "SVM" and svm_needs_retrain:
                cv2.putText(frame, "SVM can train lai: chay train_svm.ipynb", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif mode == "LSTM" and len(sequence) < SEQUENCE_LENGTH:
                cv2.putText(frame, f"Dang lay chuoi: {len(sequence)}/{SEQUENCE_LENGTH}", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            elif show_saved_result:
                cv2.putText(frame, f"Ket qua: {last_result} ({last_confidence:.0%})", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Ket qua: ...", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Nhan dang ngon ngu ky hieu", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                mode = "SVM" if mode == "LSTM" else "LSTM"
                sequence.clear()
                result = "..."
                last_result, last_confidence, last_prediction_time = "...", 0.0, 0.0
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
