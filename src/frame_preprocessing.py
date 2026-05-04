from __future__ import annotations

import cv2
import numpy as np


PREPROCESS_STEPS = ("light", "edge", "glare_light")


def apply_gamma_bgr(frame, gamma: float = 1.1):
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)


def apply_clahe_l_channel(frame, clip_limit: float = 1.5, tile_grid_size: tuple[int, int] = (8, 8)):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def apply_mild_sharpen(frame, amount: float = 0.35):
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


def compress_highlights(frame, threshold: int = 235, strength: float = 0.25):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]
    mask = v > threshold
    v[mask] = threshold + (v[mask] - threshold) * (1.0 - strength)
    hsv[:, :, 2] = np.clip(v, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def preprocess_light(frame):
    out = apply_gamma_bgr(frame, gamma=1.1)
    out = apply_clahe_l_channel(out, clip_limit=1.5)
    return out


def preprocess_edge(frame):
    out = apply_gamma_bgr(frame, gamma=1.05)
    out = apply_mild_sharpen(out, amount=0.35)
    return out


def preprocess_glare_light(frame):
    out = compress_highlights(frame, threshold=235, strength=0.25)
    out = apply_clahe_l_channel(out, clip_limit=1.2)
    return out


def normalize_preprocess_steps(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [item.strip().lower() for item in value.replace("+", ",").split(",")]
        return [item for item in parts if item and item != "none"]
    if isinstance(value, (list, tuple)):
        rows: list[str] = []
        for item in value:
            rows.extend(normalize_preprocess_steps(item))
        return rows
    return []


def preprocess_frame(frame, steps) -> np.ndarray:
    normalized_steps = normalize_preprocess_steps(steps)
    if not normalized_steps:
        return frame
    out = frame.copy()
    for step in normalized_steps:
        if step == "light":
            out = preprocess_light(out)
        elif step == "edge":
            out = preprocess_edge(out)
        elif step == "glare_light":
            out = preprocess_glare_light(out)
    return out
