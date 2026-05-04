"""
读取 YOLO 训练结果，输出逐类 mAP 排行，找出表现最差的类。

Usage:
    python scripts/debug_per_class_map.py \
        --run  runs/kitchen_visor/nano_full_v1 \
        --data C:/Users/18447/Detector/data/kitchen_visor_yolo/data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",  default="runs/kitchen_visor/nano_full_v1")
    ap.add_argument("--data", default="C:/Users/18447/Detector/data/kitchen_visor_yolo/data.yaml")
    ap.add_argument("--model-override", default="", help="指定 best.pt 路径，留空则自动找")
    args = ap.parse_args()

    import yaml
    with open(args.data, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names: list[str] = data_cfg.get("names", [])

    run_dir = Path(args.run)
    model_path = args.model_override or str(run_dir / "weights" / "best.pt")

    print(f"Loading model: {model_path}")
    import os; os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    from ultralytics import YOLO
    model = YOLO(model_path)

    # 用 val split 跑评估
    data_yaml = args.data
    results = model.val(data=data_yaml, verbose=False)

    # results.maps: per-class mAP50
    maps = getattr(results, "maps", None)    # shape [nc] mAP50 per class
    map50 = getattr(results, "box", None)

    if maps is None:
        print("无法获取 per-class mAP，请检查 ultralytics 版本")
        return

    print(f"\n{'='*60}")
    print(f"{'类名':<35} {'mAP50':>8}")
    print(f"{'-'*60}")

    pairs = sorted(zip(names, maps), key=lambda x: x[1])
    for name, m in pairs:
        bar = "#" * int(m * 40)
        flag = "  ← 关注" if m < 0.3 else ""
        print(f"  {name:<33} {m:>6.3f}  {bar}{flag}")

    poor = [(n, m) for n, m in pairs if m < 0.3]
    print(f"\n表现较差的类 (mAP50 < 0.30): {len(poor)}")
    for n, m in poor:
        print(f"  {n}: {m:.3f}")


if __name__ == "__main__":
    main()
