from __future__ import annotations

from collections import defaultdict
from typing import Any


COARSE_TO_FINE: dict[str, list[str]] = {
    "hand": ["hand", "glove"],
    "cookware": ["pot", "pan", "lid", "cookware", "tray", "kettle"],
    "dishware": ["bowl", "plate", "cup", "glass"],
    "container": ["bottle", "jar", "container", "box", "package", "bag", "carton", "can"],
    "utensil": [
        "knife", "fork", "spoon", "spatula", "tongs", "ladle", "whisk",
        "peeler", "scissors", "cutting board",
    ],
    "ingredient": [
        "pasta", "noodles", "rice", "bread", "vegetable", "fruit", "meat",
        "fish", "egg", "cheese", "ingredient", "food", "dry food", "liquid",
        "water", "milk", "sauce", "oil", "powder", "sugar", "salt", "olive",
    ],
    "kitchen_scene": [
        "sink", "faucet", "stove", "cooktop", "oven", "microwave", "fridge",
        "drawer", "cabinet", "countertop", "table", "rack", "sponge", "towel",
    ],
}


FINE_TO_COARSE: dict[str, str] = {
    fine: coarse
    for coarse, labels in COARSE_TO_FINE.items()
    for fine in labels
}


def normalize_label(label: str | None) -> str:
    text = str(label or "").strip().lower()
    text = " ".join(text.split())
    return text or "unknown"


def fine_to_coarse(label: str | None) -> str:
    fine = normalize_label(label)
    return FINE_TO_COARSE.get(fine, "unknown")


def make_display_label(fine_label: str | None) -> str:
    fine = normalize_label(fine_label)
    return f"{fine_to_coarse(fine)}:{fine}"


def parse_display_label(display_label: str | None) -> tuple[str, str]:
    text = str(display_label or "").strip()
    if ":" in text:
        coarse, fine = text.split(":", 1)
        fine_norm = normalize_label(fine)
        coarse_norm = normalize_label(coarse)
        if coarse_norm == "unknown":
            coarse_norm = fine_to_coarse(fine_norm)
        return coarse_norm, fine_norm
    fine = normalize_label(text)
    return fine_to_coarse(fine), fine


def same_coarse(label_a: str | None, label_b: str | None) -> bool:
    return fine_to_coarse(label_a) == fine_to_coarse(label_b)


def label_conflict_level(predicted_label: str | None, target_label: str | None) -> str:
    pred = normalize_label(predicted_label)
    target = normalize_label(target_label)
    if pred == target:
        return "match"
    if same_coarse(pred, target):
        return "fine_conflict"
    return "coarse_conflict"


def hierarchy_payload(labels: list[str] | set[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    fine_labels = sorted({normalize_label(v) for v in labels or [] if normalize_label(v)})
    fine_map = {label: fine_to_coarse(label) for label in fine_labels}
    for coarse, values in COARSE_TO_FINE.items():
        for fine in values:
            if not fine_labels or fine in fine_map:
                fine_map.setdefault(fine, coarse)

    coarse_map: dict[str, list[str]] = defaultdict(list)
    for fine, coarse in sorted(fine_map.items()):
        coarse_map[coarse].append(fine)
    for coarse in coarse_map:
        coarse_map[coarse] = sorted(set(coarse_map[coarse]))

    return {
        "fine_to_coarse": dict(sorted(fine_map.items())),
        "coarse_to_fine": dict(sorted(coarse_map.items())),
    }


def display_names_payload(labels: list[str] | set[str] | tuple[str, ...]) -> dict[str, str]:
    return {normalize_label(label): make_display_label(label) for label in sorted({normalize_label(v) for v in labels})}


def label_for_display(label: str | None, mode: str = "coarse_fine") -> str:
    coarse, fine = parse_display_label(label)
    if mode == "fine":
        return fine
    if mode == "coarse":
        return coarse
    return f"{coarse}:{fine}"
