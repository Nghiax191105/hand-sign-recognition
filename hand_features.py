"""Các hàm chung để đổi MediaPipe Hands thành vector đặc trưng 126 chiều."""

import numpy as np


def extract_hand_keypoints(results):
    """Trả về [tay trái (63), tay phải (63)]; tay không thấy được điền 0."""
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if not results.multi_hand_landmarks:
        return np.concatenate([left, right])

    for index, landmarks in enumerate(results.multi_hand_landmarks):
        label = results.multi_handedness[index].classification[0].label
        points = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32).flatten()
        if label == "Left":
            left = points
        else:
            right = points
    return np.concatenate([left, right])


def extract_static_features(results):
    """Tạo 63 đặc trưng hình dạng cho một bàn tay dùng trong SVM.

    Tọa độ được đưa về gốc cổ tay và chia theo kích thước bàn tay. Vì vậy
    SVM học hình dạng ngón tay thay vì học tay đang ở đâu trong khung hình.
    Tay trái được lật theo trục x để cùng một cử chỉ hai tay có cùng dạng.
    """
    if not results.multi_hand_landmarks:
        return None

    points = np.array(
        [[point.x, point.y, point.z] for point in results.multi_hand_landmarks[0].landmark],
        dtype=np.float32,
    )
    points -= points[0] 
    scale = np.linalg.norm(points[9, :2])
    if scale < 1e-6:
        return None
    points /= scale

    hand_label = results.multi_handedness[0].classification[0].label
    if hand_label == "Left":
        points[:, 0] *= -1
    return points.flatten()
