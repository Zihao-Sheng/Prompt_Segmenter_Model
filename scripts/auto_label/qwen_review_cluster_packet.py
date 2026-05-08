from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL review for one cluster packet.")
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    try:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.memory_autolabel.pipeline.vlm_reviewer import VLMReviewer

        packet_dir = Path(args.packet_dir)
        reviewer = VLMReviewer(
            model_id=args.model_id,
            local_files_only=args.local_files_only,
            max_new_tokens=args.max_new_tokens,
            log=lambda message: print(message, file=sys.stderr, flush=True),
        )
        response = reviewer.review(packet_dir)
        print(json.dumps(response, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
