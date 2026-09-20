import math
import shutil
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hand_features import extract_static_features

ACTIONS = ["0"]
SAMPLES_PER_ACTION = 120
SAMPLES_PER_BATCH = 20
RESET_EXISTING_DATA = True
DATA_DIR = ROOT / "data" / "processed" / "svm"


def draw_skeleton(frame, results, mp_hands):
    for landmarks in results.multi_hand_landmarks or []:
        mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)


def wait_for_start(cap, hands, action, batch, total_batches, mp_hands):
    """S để ghi lại nhãn đó"""
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw_skeleton(frame, results, mp_hands)
        cv2.putText(frame, f"Chu {action} - batch {batch}/{total_batches}", (15, 40), 0, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, "Doi tu the/goc nhin, roi bam S | Q: thoat", (15, 75), 0, 0.55, (0, 0, 255), 2)
        cv2.imshow("Thu thap SVM", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            return True
        if key == ord("q"):
            return False


def prepare_folder(action):
    folder = DATA_DIR / action
    if RESET_EXISTING_DATA and folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Khong the mo camera.")

    mp_hands = mp.solutions.hands
    total_batches = math.ceil(SAMPLES_PER_ACTION / SAMPLES_PER_BATCH)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        for action in ACTIONS:
            folder = prepare_folder(action)
            saved = 0
            for batch in range(1, total_batches + 1):
                if not wait_for_start(cap, hands, action, batch, total_batches, mp_hands):
                    cap.release(); cv2.destroyAllWindows(); return

                batch_end = min(saved + SAMPLES_PER_BATCH, SAMPLES_PER_ACTION)
                while saved < batch_end:
                    ok, frame = cap.read()
                    if not ok:
                        cap.release(); cv2.destroyAllWindows(); return
                    frame = cv2.flip(frame, 1)
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw_skeleton(frame, results, mp_hands)
                    features = extract_static_features(results)
                    if features is not None:
                        np.save(folder / f"{saved}.npy", features)
                        saved += 1
                    cv2.putText(frame, f"{action}: {saved}/{SAMPLES_PER_ACTION}", (15, 40), 0, 0.75, (255, 0, 0), 2)
                    if features is None:
                        cv2.putText(frame, "Khong thay tay - khong luu", (15, 75), 0, 0.6, (0, 0, 255), 2)
                    cv2.imshow("Thu thap SVM", frame)
                    if cv2.waitKey(45) & 0xFF == ord("q"):
                        cap.release(); cv2.destroyAllWindows(); return
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
