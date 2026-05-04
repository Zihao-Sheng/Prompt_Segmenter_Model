"""
训练数据集调试工具：
1. 统计每个类的实例数和图片数
2. 列出稀有类（可能学不好）
3. 输出建议忽略/合并的类

Usage:
    python scripts/debug_dataset_stats.py \
        --labels C:/Users/18447/Detector/data/kitchen_visor_yolo/labels/train \
        --yaml   C:/Users/18447/Detector/data/kitchen_visor_yolo/data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="C:/Users/18447/Detector/data/kitchen_visor_yolo/labels/train")
    ap.add_argument("--yaml",   default="C:/Users/18447/Detector/data/kitchen_visor_yolo/data.yaml")
    ap.add_argument("--rare-threshold", type=int, default=300,
                    help="实例数低于此值视为稀有类")
    args = ap.parse_args()

    # 读类名
    import yaml
    with open(args.yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names: list[str] = data.get("names", [])

    instance_count: dict[int, int] = defaultdict(int)
    image_count: dict[int, int] = defaultdict(int)
    total_images = 0
    empty_images = 0

    label_dir = Path(args.labels)
    for txt in label_dir.glob("*.txt"):
        total_images += 1
        seen_in_image: set[int] = set()
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(line.split()[0])
            instance_count[cid] += 1
            seen_in_image.add(cid)
        if not seen_in_image:
            empty_images += 1
        for cid in seen_in_image:
            image_count[cid] += 1

    print(f"\n{'='*65}")
    print(f"总图片数: {total_images}  空标注图: {empty_images}")
    print(f"{'='*65}")
    print(f"{'类名':<35} {'实例数':>8} {'图片数':>8} {'状态'}")
    print(f"{'-'*65}")

    rare: list[str] = []
    ok: list[str] = []

    for cid, name in enumerate(names):
        inst = instance_count.get(cid, 0)
        imgs = image_count.get(cid, 0)
        status = ""
        if inst == 0:
            status = "⚠ 无标注"
        elif inst < args.rare_threshold:
            status = f"⚠ 稀有 (<{args.rare_threshold})"
            rare.append(name)
        else:
            ok.append(name)
        print(f"  {name:<33} {inst:>8,} {imgs:>8,}  {status}")

    print(f"\n{'='*65}")
    print(f"正常类 ({len(ok)}):")
    for n in ok:
        print(f"  {n}")

    print(f"\n稀有类 ({len(rare)}) — 建议观察，考虑降低置信阈值或合并到大类:")
    for n in rare:
        print(f"  {n}")

    # 粗略估算类不平衡比
    counts = [instance_count.get(i, 0) for i in range(len(names))]
    nonzero = [c for c in counts if c > 0]
    if nonzero:
        ratio = max(nonzero) / min(nonzero)
        print(f"\n最大/最小实例比: {ratio:.1f}x  ", end="")
        if ratio > 100:
            print("（严重不平衡，copy_paste 和 mosaic 有帮助）")
        elif ratio > 20:
            print("（中度不平衡，训练后期关注稀有类 mAP）")
        else:
            print("（相对均衡）")


if __name__ == "__main__":
    main()
