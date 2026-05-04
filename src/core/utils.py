from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def dump_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(make_json_safe(data), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def create_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("prompt_run_%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    ensure_dir(run_dir)
    return run_dir


def bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


_PROMPT_LABEL_EXPANSIONS: dict[str, list[str]] = {
    "hand": ["hand"],
    "person": ["hand", "arm", "person"],
    "cookware": [
        "cookware",
        "pot",
        "cooking pot",
        "saucepan",
        "pan",
        "frying pan",
        "wok",
        "kettle",
        "stovetop kettle",
        "tea kettle",
        "electric kettle",
    ],
    "lid": ["lid", "pot lid", "pan lid"],
    "dishware": ["plate", "bowl", "dish", "cup", "mug", "glass"],
    "utensil": ["spoon", "fork", "ladle", "spatula", "knife", "tongs", "utensil"],
    "appliance": ["stove", "oven", "microwave", "toaster", "rice cooker"],
    "sink": ["sink"],
    "faucet": ["faucet"],
    "countertop": ["countertop", "kitchen counter"],
    "cabinet": ["cabinet", "cabinet door", "drawer"],
    "cooktop": ["cooktop", "stovetop", "electric range", "stove burner", "burner"],
    "wall": ["wall", "kitchen wall", "backsplash"],
}


def parse_prompt_labels(prompt: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for item in prompt.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        expanded = _PROMPT_LABEL_EXPANSIONS.get(normalized.lower(), [normalized])
        for label in expanded:
            cleaned = str(label).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(cleaned)
    return labels


def path_to_text(path: Path | None) -> str:
    return "" if path is None else str(path)


def open_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    command = ["open" if sys.platform == "darwin" else "xdg-open", str(path)]
    subprocess.Popen(command)


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return make_json_safe(value.tolist())
        if isinstance(value, np.generic):
            return make_json_safe(value.item())
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return make_json_safe(value.tolist())
        except Exception:
            pass
    return str(value)
