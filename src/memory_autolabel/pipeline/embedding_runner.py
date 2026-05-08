from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class EmbeddingRunner:
    """Crop embedding adapter.

    Uses a local Hugging Face CLIP/SigLIP model when available, otherwise falls
    back to a deterministic color histogram so memory matching still works.
    """

    def __init__(
        self,
        backend: str = "clip",
        model_id: str = "openai/clip-vit-base-patch32",
        local_files_only: bool = True,
        device: str = "auto",
        log=None,
    ) -> None:
        self.backend = backend
        self.model_id = model_id
        self.local_files_only = local_files_only
        self.device = self._resolve_device(device)
        self.log = log or (lambda message: None)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._load_error = ""

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self) -> bool:
        if self._model is not None and self._processor is not None:
            return True
        if self._load_error:
            return False
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=self.local_files_only)
            self._model = AutoModel.from_pretrained(self.model_id, local_files_only=self.local_files_only).to(self.device).eval()
            self._torch = torch
            self.log(f"Embedding model ready: {self.model_id}")
            return True
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            self.log(f"Embedding model unavailable; using color histogram memory. {self._load_error}")
            return False

    def embed_crop(self, frame, bbox_xyxy: list[float], mask=None) -> list[float]:
        crop = self._crop(frame, bbox_xyxy, mask)
        if crop.size == 0:
            return []
        if self._load():
            try:
                from PIL import Image

                image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with self._torch.inference_mode():
                    if hasattr(self._model, "get_image_features"):
                        features = self._model.get_image_features(**inputs)
                    else:
                        outputs = self._model(**inputs)
                        features = getattr(outputs, "pooler_output", None)
                        if features is None:
                            features = outputs.last_hidden_state.mean(dim=1)
                vec = features[0].detach().float().cpu().numpy()
                vec = vec / max(1e-8, float(np.linalg.norm(vec)))
                return vec.astype(float).tolist()
            except Exception as exc:
                self.log(f"Embedding inference failed; using color histogram. {type(exc).__name__}: {exc}")
        return self._histogram_embedding(crop)

    def _crop(self, frame, bbox_xyxy: list[float], mask=None):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return frame[0:0, 0:0]
        crop = frame[y1:y2, x1:x2].copy()
        if mask is not None:
            crop_mask = mask[y1:y2, x1:x2] > 0
            if crop_mask.shape[:2] == crop.shape[:2]:
                crop[~crop_mask] = 0
        return crop

    def _histogram_embedding(self, crop) -> list[float]:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        hist = hist.astype("float32")
        hist /= max(1e-8, float(np.linalg.norm(hist)))
        return hist.astype(float).tolist()
