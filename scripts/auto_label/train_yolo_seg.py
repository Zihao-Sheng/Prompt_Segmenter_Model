"""
Phase 9 — YOLO11-Seg Training Wrapper.

Fine-tune YOLO11-seg on the exported dataset using the ultralytics library
(already installed — see requirements.txt).

Supported model sizes:
  yolo11n-seg.pt   nano   (~3 MB)
  yolo11s-seg.pt   small  (~12 MB)  ← default, good for RTX 4060 laptop
  yolo11m-seg.pt   medium (~22 MB)

Usage:
    python scripts/auto_label/train_yolo_seg.py \
        --data    data/auto_label_demo/yolo_dataset/dataset.yaml \
        --model   yolo11s-seg.pt \
        --epochs  30 \
        --imgsz   640 \
        --batch   8 \
        --device  0 \
        --project outputs/auto_label_yolo \
        --name    demo_yolo11s_seg
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11-seg on the auto-labeled dataset.")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml.")
    parser.add_argument("--model", default="yolo11s-seg.pt", help="YOLO model size/path.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU index (0) or 'cpu'.")
    parser.add_argument("--project", default="outputs/auto_label_yolo")
    parser.add_argument("--name", default="yolo11_seg_run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--progress-interval", type=int, default=50,
                        help="Print a GUI-friendly progress line every N training batches.")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights (default: yes).")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision (default: yes).")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        parser.error(f"dataset.yaml not found: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "[error] ultralytics not installed.\n"
            "  pip install ultralytics"
        )
        return

    print(f"Model   : {args.model}")
    print(f"Data    : {args.data}")
    print(f"Epochs  : {args.epochs}  imgsz={args.imgsz}  batch={args.batch}")
    print(f"Device  : {args.device}")
    print(f"Project : {args.project}/{args.name}")
    print()

    model = YOLO(args.model)

    progress_interval = max(1, int(args.progress_interval))
    total_epochs = int(args.epochs)

    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _on_train_batch_end(trainer) -> None:
        batch = int(getattr(trainer, "batch", 0)) + 1
        nb = max(1, int(getattr(trainer, "nb", 1)))
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        if batch == 1 or batch == nb or batch % progress_interval == 0:
            pct = 100.0 * batch / nb
            print(f"[progress] epoch {epoch}/{total_epochs} batch {batch}/{nb} ({pct:.1f}%)", flush=True)

    def _on_fit_epoch_end(trainer) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        loss_items = getattr(trainer, "loss_items", None)
        box_loss = _safe_float(loss_items[0]) if loss_items is not None and len(loss_items) > 0 else 0.0
        seg_loss = _safe_float(loss_items[1]) if loss_items is not None and len(loss_items) > 1 else 0.0
        cls_loss = _safe_float(loss_items[2]) if loss_items is not None and len(loss_items) > 2 else 0.0
        metrics = getattr(trainer, "metrics", {}) or {}
        map50 = 0.0
        if isinstance(metrics, dict):
            map50 = _safe_float(metrics.get("metrics/mAP50(M)", metrics.get("metrics/mAP50(B)", 0.0)))
        else:
            try:
                map50 = _safe_float(metrics.seg.map50)
            except Exception:
                try:
                    map50 = _safe_float(metrics.box.map50)
                except Exception:
                    map50 = 0.0
        print(
            f"[epoch] {epoch}/{total_epochs} "
            f"box_loss={box_loss:.4f} seg_loss={seg_loss:.4f} "
            f"cls_loss={cls_loss:.4f} mAP50={map50:.4f}",
            flush=True,
        )

    model.add_callback("on_train_batch_end", _on_train_batch_end)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        pretrained=args.pretrained,
        resume=args.resume,
        amp=args.amp,
        verbose=True,
    )
    print(f"\nTraining complete.")
    print(f"Best weights : {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
