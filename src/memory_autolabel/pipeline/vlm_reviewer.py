from __future__ import annotations

from pathlib import Path
from typing import Any
import json


class VLMReviewer:
    """VLM adapter interface with optional local Qwen2.5-VL."""

    def __init__(
        self,
        backend: str = "qwen2.5-vl",
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        local_files_only: bool = True,
        max_new_tokens: int = 256,
        device: str = "auto",
        log=None,
    ) -> None:
        self.backend = backend
        self.model_id = model_id
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self.device = self._resolve_device(device)
        self.log = log or (lambda message: None)
        self._model: Any | None = None
        self._processor: Any | None = None
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

    def _load_qwen(self) -> bool:
        if self._model is not None and self._processor is not None:
            return True
        if self._load_error:
            return False
        if "qwen" not in self.backend.lower():
            self._load_error = f"Unsupported VLM backend: {self.backend}"
            return False
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=self.local_files_only,
            ).eval()
            if self.device == "cpu":
                self._model = self._model.to(self.device)
            self._processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=self.local_files_only)
            self._torch = torch
            self.log(f"VLM ready: {self.model_id}")
            return True
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            self.log(f"VLM unavailable; using stub reviewer. {self._load_error}")
            return False

    def review(self, packet_path: Path) -> dict[str, Any]:
        if self._load_qwen():
            try:
                return self._review_with_qwen(packet_path)
            except Exception as exc:
                self.log(f"VLM review failed; using stub response. {type(exc).__name__}: {exc}")
        return {
            "decision": "uncertain",
            "issue_type": "none",
            "severity": "low",
            "corrected_label": None,
            "recommended_action": "send_to_human_review",
            "sam2_prompt": {"box_xyxy": None, "positive_points": [], "negative_points": [], "polygon_xy": []},
            "candidate_choice": {"accept_candidate_id": None},
            "reason": "stub reviewer; no automatic VLM decision applied",
            "confidence": 0.0,
        }

    def _review_with_qwen(self, packet_path: Path) -> dict[str, Any]:
        from PIL import Image

        prompt = (packet_path / "prompt.txt").read_text(encoding="utf-8") if (packet_path / "prompt.txt").exists() else ""
        schema_prompt = (
            prompt
            + "\n\nReturn JSON only using this schema: "
            '{"decision":"correct|needs_fix|uncertain|reject","issue_type":"none|missing_object|wrong_class|under_segmented|over_segmented|merged_instances|background_false_positive|track_inconsistent","severity":"low|medium|high","corrected_label":null,"recommended_action":"accept|reprompt_sam2|subtract_region|add_region|split_instances|send_to_human_review|reject","sam2_prompt":{"box_xyxy":null,"positive_points":[],"negative_points":[],"polygon_xy":[]},"candidate_choice":{"accept_candidate_id":null},"reason":"short reason","confidence":0.0}'
        )
        image_paths = [
            packet_path / "full_overlay.jpg",
            packet_path / "target_crop_overlay.jpg",
            packet_path / "target_crop_masked.jpg",
            packet_path / "full_frame.jpg",
        ]
        images = [Image.open(path).convert("RGB") for path in image_paths if path.exists()]
        content = [{"type": "image", "image": image} for image in images[:3]]
        content.append({"type": "text", "text": schema_prompt})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=images[:3] if images else None, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        output = self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parsed = self._parse_json(output)
        parsed.setdefault("raw_vlm_output", output)
        return parsed

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.removeprefix("json").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = self._repair_json_like_output(cleaned)
            data = json.loads(repaired)
        if not isinstance(data, dict):
            raise ValueError("VLM output JSON was not an object")
        if "items" in data and isinstance(data["items"], list):
            data.setdefault("decision", "needs_fix" if data["items"] else "correct")
            data.setdefault("issue_type", "mixed_cluster" if data["items"] else "none")
            data.setdefault("recommended_action", "apply_per_crop_changes" if data["items"] else "accept")
            confidences = []
            for item in data["items"]:
                if isinstance(item, dict):
                    try:
                        confidences.append(float(item.get("confidence") or 0.0))
                    except Exception:
                        pass
            data.setdefault("confidence", max(confidences) if confidences else 0.0)
            data.setdefault("reason", f"Per-crop review returned {len(data['items'])} item(s).")
        return data

    def _repair_json_like_output(self, text: str) -> str:
        """Best-effort repair for short Qwen JSON outputs cut off mid-string."""
        cleaned = text.strip()
        start = cleaned.find("{")
        if start >= 0:
            cleaned = cleaned[start:]
        if not cleaned.endswith("}"):
            cleaned = cleaned.rstrip().rstrip(",")
            open_braces = cleaned.count("{") - cleaned.count("}")
            if cleaned.count('"') % 2 == 1:
                cleaned += '"'
            cleaned += "}" * max(1, open_braces)
        return cleaned
