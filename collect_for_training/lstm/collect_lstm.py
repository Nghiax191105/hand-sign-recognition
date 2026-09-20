import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hand_features import extract_hand_keypoints

ACTIONS = ["J", "Z", "XinLoi"]
SAMPLES_PER_ACTION = 30
SEQUENCE_LENGTH = 30
DATA_DIR = ROOT / "data" / "processed" / "lstm"


def draw_skeleton(frame, results, mp_hands):
    for landmarks in results.multi_hand_landmarks or []:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, landmarks, mp_hands.HAND_CONNECTIONS
        )


def wait_for_start(cap, hands, action, number):
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw_skeleton(frame, results, mp.solutions.hands)
        cv2.putText(frame, f"{action}: mau {number}/{SAMPLES_PER_ACTION}", (15, 40), 0, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "S: bat dau | Q: thoat", (15, 75), 0, 0.7, (0, 0, 255), 2)
        cv2.imshow("Thu thap LSTM", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            return True
        if key == ord("q"):
            return False


def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for action in ACTIONS:
            for sample in range(SAMPLES_PER_ACTION):
                if not wait_for_start(cap, hands, action, sample + 1):
                    cap.release(); cv2.destroyAllWindows(); return
                folder = DATA_DIR / action / str(sample)
                folder.mkdir(parents=True, exist_ok=True)
                saved_frames = 0
                while saved_frames < SEQUENCE_LENGTH:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.flip(frame, 1)
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw_skeleton(frame, results, mp_hands)
                    if results.multi_hand_landmarks:
                        np.save(folder / f"{saved_frames}.npy", extract_hand_keypoints(results))
                        saved_frames += 1
                    cv2.putText(frame, f"{action}: {saved_frames}/{SEQUENCE_LENGTH}", (15, 40), 0, 0.8, (255, 0, 0), 2)
                    if not results.multi_hand_landmarks:
                        cv2.putText(frame, "Dua tay vao camera", (15, 75), 0, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Thu thap LSTM", frame)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        cap.release(); cv2.destroyAllWindows(); return
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
