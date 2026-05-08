from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

import cv2

from src.memory_autolabel.pipeline.config import RunConfig
from src.memory_autolabel.pipeline.detector_runner import DetectorRunner
from src.memory_autolabel.pipeline.embedding_runner import EmbeddingRunner
from src.memory_autolabel.pipeline.exporter import Exporter
from src.memory_autolabel.pipeline.mask_repair import MaskRepair
from src.memory_autolabel.pipeline.memory_store import MemoryStore
from src.memory_autolabel.pipeline.quality_scoring import QualityScorer
from src.memory_autolabel.pipeline.sam2_runner import SAM2Runner
from src.memory_autolabel.pipeline.tracker import SimpleTracker
from src.memory_autolabel.pipeline.vlm_packet_builder import VLMPacketBuilder
from src.memory_autolabel.pipeline.vlm_reviewer import VLMReviewer
from src.memory_autolabel.utils.jsonl import append_jsonl, read_json, write_json
from src.memory_autolabel.utils.video_io import sample_video_frames, scan_videos
from src.memory_autolabel.utils.visualization import save_overlay


ProgressFn = Callable[[dict[str, Any]], None]
LogFn = Callable[[str], None]


class StopRequested(Exception):
    pass


class VideoFolderRunner:
    def __init__(self, config: RunConfig, progress: ProgressFn, log: LogFn) -> None:
        self.config = config
        self.progress = progress
        self.log = log
        self.paused = False
        self.stop_after_current_video = False
        self.stop_now = False
        self.output_root = Path(config.output_folder)
        self.memory_root = Path(config.memory_save_path) if config.memory_save_path else self.output_root / "memory"
        self.detector = DetectorRunner(
            use_real=config.use_real_detector,
            threshold=config.bbox_threshold,
            device=config.detector_device,
            checkpoint_path=config.groundingdino_checkpoint_path,
            config_path=config.groundingdino_config_path,
            log=self.log,
        )
        self.sam2 = SAM2Runner(
            use_real=config.use_real_sam2,
            checkpoint_path=config.sam2_checkpoint_path,
            model_cfg=config.sam2_model_cfg,
            device=config.detector_device,
            run_dir=self.output_root / "_sam2_runtime",
            log=self.log,
        )
        self.embedding = EmbeddingRunner(
            backend=config.embedding_backend,
            model_id=config.embedding_model_id,
            local_files_only=config.embedding_local_files_only,
            device=config.detector_device,
            log=self.log,
        )
        self.quality = QualityScorer()
        self.vlm_packets = VLMPacketBuilder()
        self.vlm = VLMReviewer(
            backend=config.vlm_backend,
            model_id=config.vlm_model_id,
            local_files_only=config.vlm_local_files_only,
            max_new_tokens=config.vlm_max_new_tokens,
            device=config.detector_device,
            log=self.log,
        )
        self.repair = MaskRepair()
        self.memory = MemoryStore(self.memory_root)
        self.exporter = Exporter(self.output_root)

    def request_pause(self, value: bool) -> None:
        self.paused = value

    def request_stop_after_current_video(self) -> None:
        self.stop_after_current_video = True

    def request_stop_now(self) -> None:
        self.stop_now = True

    def _checkpoint(self) -> None:
        if self.stop_now:
            raise StopRequested()
        while self.paused and not self.stop_now:
            time.sleep(0.2)
        if self.stop_now:
            raise StopRequested()

    def _stage(self, name: str, value: int = 0, maximum: int = 0) -> None:
        self.progress({"stage_name": name, "stage_value": value, "stage_max": maximum})

    def prepare(self) -> tuple[list[Path], dict[str, Any]]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "run_logs").mkdir(parents=True, exist_ok=True)
        (self.output_root / "dataset_export" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_root / "dataset_export" / "masks").mkdir(parents=True, exist_ok=True)
        self.config.save(self.output_root / "run_config.json")
        processed_path = self.output_root / "processed_videos.json"
        processed = read_json(processed_path, {"completed": [], "failed": [], "marked_for_reprocess": []})
        videos = scan_videos(Path(self.config.video_folder), self.config.recursive_scan)
        if self.config.max_total_videos:
            videos = videos[: self.config.max_total_videos]
        completed = set(processed.get("completed", []))
        if not self.config.reprocess_completed:
            videos = [v for v in videos if str(v) not in completed]
        return videos, processed

    def run(self) -> None:
        videos, processed = self.prepare()
        total = len(videos)
        self.log(f"Found {total} videos to process.")
        round_id = 1
        processed_count = 0
        while processed_count < total:
            self._checkpoint()
            batch = videos[processed_count : processed_count + max(1, self.config.videos_per_round)]
            if not batch:
                break
            round_dir = self.output_root / "rounds" / f"round_{round_id:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            round_summaries = []
            self.log(f"Round {round_id}: {len(batch)} video(s)")
            for index_in_round, video in enumerate(batch, start=1):
                self._checkpoint()
                self.progress({
                    "overall_value": processed_count,
                    "overall_max": total,
                    "round_value": index_in_round - 1,
                    "round_max": len(batch),
                    "current_video": video.name,
                })
                try:
                    summary = self.process_video(video, round_dir, round_id)
                    round_summaries.append(summary)
                    processed.setdefault("completed", []).append(str(video))
                    self.log(f"Completed video: {video.name}")
                except StopRequested:
                    raise
                except Exception as exc:
                    processed.setdefault("failed", []).append(str(video))
                    summary = {"video": str(video), "error": str(exc)}
                    round_summaries.append(summary)
                    self.log(f"FAILED video {video.name}: {exc}")
                finally:
                    write_json(self.output_root / "processed_videos.json", processed)
                processed_count += 1
                self.progress({"overall_value": processed_count, "overall_max": total, "round_value": index_in_round, "round_max": len(batch)})
                if self.stop_after_current_video:
                    self.log("Stop requested after current video.")
                    write_json(round_dir / "round_summary.json", {"round": round_id, "videos": round_summaries})
                    return
            memory_summary = self.memory.summary()
            write_json(round_dir / "round_summary.json", {"round": round_id, "videos": round_summaries, "memory": memory_summary})
            self.progress({"memory_summary": memory_summary})
            round_id += 1
        self.progress({"overall_value": total, "overall_max": total})
        self.log("Folder run complete.")

    def process_video(self, video: Path, round_dir: Path, round_id: int) -> dict[str, Any]:
        prompts = [p.strip() for p in self.config.prompts.split(",") if p.strip()]
        safe_name = video.stem.replace(" ", "_")
        video_dir = round_dir / safe_name
        dirs = {
            name: video_dir / name
            for name in [
                "sampled_frames",
                "overlays_initial",
                "overlays_repaired",
                "masks",
                "crops",
                "vlm_packets",
                "vlm_responses",
            ]
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        self._stage("loading video")
        self.log(f"Sampling frames: {video}")
        sampled = sample_video_frames(
            video,
            dirs["sampled_frames"],
            self.config.frame_stride,
            self.config.max_sampled_frames_per_video,
            adaptive_stride=self.config.enable_adaptive_stride,
            high_risk_dense_stride=self.config.high_risk_dense_stride,
            progress=lambda v, m: self.progress({"video_value": v, "video_max": m}),
        )
        tracker = SimpleTracker()
        initial_records: list[dict[str, Any]] = []
        repaired_records: list[dict[str, Any]] = []
        quality_rows: list[dict[str, Any]] = []
        vlm_sent = 0
        vlm_fix_accepted = 0
        vlm_fix_rejected = 0
        accepted = uncertain = rejected = 0
        before_scores: list[float] = []
        after_scores: list[float] = []

        for frame_offset, frame_info in enumerate(sampled):
            self._checkpoint()
            frame_path = Path(frame_info["path"])
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            self.progress({"video_value": frame_offset + 1, "video_max": len(sampled)})
            self._stage("DINO bbox proposal", frame_offset + 1, len(sampled))
            detections = self.detector.detect(frame, prompts, self.config.bbox_threshold, frame_idx=frame_info["frame_id"])
            detections = self._apply_memory_to_detections(frame, detections)
            self._stage("SAM2 mask generation", frame_offset + 1, len(sampled))
            masks = self.sam2.segment(frame, detections, frame_idx=frame_info["frame_id"])
            self._stage("tracking / memory matching", frame_offset + 1, len(sampled))
            masks = tracker.assign(frame_info["frame_id"], masks)
            self._stage("quality scoring", frame_offset + 1, len(sampled))
            frame_records: list[dict[str, Any]] = []
            for rec in masks:
                mask_name = f"frame_{frame_info['frame_id']:06d}_mask_{rec['candidate_id']:04d}.png"
                mask_path = dirs["masks"] / mask_name
                cv2.imwrite(str(mask_path), rec["mask"])
                scored = {k: v for k, v in rec.items() if k != "mask"}
                scored["mask_path"] = str(mask_path)
                scored["frame_id"] = frame_info["frame_id"]
                scored["frame_path"] = str(frame_path)
                score = self.quality.score(rec, {"image_shape": frame.shape})
                if self.config.enable_embedding_memory:
                    embedding = self.embedding.embed_crop(frame, scored["bbox_xyxy"], rec.get("mask"))
                    memory_match = self.memory.query(embedding, class_hint=str(scored.get("label", "")))
                    scored["embedding"] = embedding
                    scored["memory_matches"] = memory_match.get("matches", [])
                    scored["memory_confidence_boost"] = memory_match.get("confidence_boost", 0.0)
                    scored["memory_hard_negative"] = memory_match.get("hard_negative", False)
                    score["memory_score"] = max(0.0, min(1.0, 0.50 + float(memory_match.get("confidence_boost", 0.0))))
                    score["final_quality_score"] = max(0.0, min(1.0, float(score["final_quality_score"]) + float(memory_match.get("confidence_boost", 0.0))))
                    if memory_match.get("hard_negative"):
                        score.setdefault("hard_flags", []).append("memory_hard_negative")
                        score["status"] = "needs_vlm" if score["final_quality_score"] >= 0.45 else "rejected"
                scored.update(score)
                before_scores.append(float(score["final_quality_score"]))
                if scored["status"] == "accepted":
                    accepted += 1
                elif scored["status"] == "rejected":
                    rejected += 1
                else:
                    uncertain += 1
                frame_records.append(scored)
                initial_records.append(scored)
                quality_rows.append(scored)

            initial_overlay = dirs["overlays_initial"] / f"frame_{frame_info['frame_id']:06d}.jpg"
            save_overlay(frame_path, frame_records, initial_overlay)

            self._stage("VLM packet generation", frame_offset + 1, len(sampled))
            repaired_frame_records = []
            for rec in frame_records:
                repaired = dict(rec)
                needs_vlm = (
                    self.config.enable_vlm_review
                    and vlm_sent < self.config.max_vlm_packets_per_video
                    and (float(rec.get("final_quality_score", 0.0)) < self.config.vlm_review_threshold or rec.get("hard_flags"))
                )
                if needs_vlm:
                    packet_dir = dirs["vlm_packets"] / f"packet_{vlm_sent + 1:04d}"
                    self.vlm_packets.build(packet_dir, frame_path, rec, initial_overlay)
                    self.progress({"vlm_value": vlm_sent + 1, "vlm_max": self.config.max_vlm_packets_per_video})
                    self._stage("VLM review", vlm_sent + 1, self.config.max_vlm_packets_per_video)
                    response = self.vlm.review(packet_dir)
                    append_jsonl(dirs["vlm_responses"] / "vlm_responses.jsonl", {"packet": str(packet_dir), "response": response})
                    vlm_sent += 1
                    self._stage("mask repair", vlm_sent, self.config.max_vlm_packets_per_video)
                    repaired, ok = self.repair.apply_safe_repair(
                        repaired,
                        response,
                        frame=frame,
                        sam2=self.sam2,
                        quality_scorer=self.quality,
                        mask_dir=dirs["masks"],
                    )
                    if ok:
                        vlm_fix_accepted += 1
                    else:
                        vlm_fix_rejected += 1
                after_scores.append(float(repaired.get("final_quality_score", 0.0)))
                repaired_frame_records.append(repaired)
                repaired_records.append(repaired)
                self.memory.update_from_record(repaired, str(video), round_id, embedding=repaired.get("embedding"))
                self.exporter.export_dataset_label(repaired)

            repaired_overlay = dirs["overlays_repaired"] / f"frame_{frame_info['frame_id']:06d}.jpg"
            save_overlay(frame_path, repaired_frame_records, repaired_overlay, repaired=True)

        self._stage("export results")
        for row in initial_records:
            append_jsonl(video_dir / "pseudo_labels_initial.jsonl", row)
        for row in repaired_records:
            append_jsonl(video_dir / "pseudo_labels_repaired.jsonl", row)
        for row in quality_rows:
            append_jsonl(video_dir / "quality_scores.jsonl", row)

        summary = {
            "video": str(video),
            "video_name": video.name,
            "sampled_frames": len(sampled),
            "initial_detections": len(initial_records),
            "initial_masks": len(initial_records),
            "accepted_masks": accepted,
            "uncertain_masks": uncertain,
            "rejected_masks": rejected,
            "vlm_packets_sent": vlm_sent,
            "vlm_fixes_accepted": vlm_fix_accepted,
            "vlm_fixes_rejected": vlm_fix_rejected,
            "missing_objects_found": 0,
            "under_segmented_masks_fixed": 0,
            "over_segmented_masks_fixed": 0,
            "wrong_class_corrections": 0,
            "track_consistency_issues": 0,
            "average_quality_score_before_repair": sum(before_scores) / len(before_scores) if before_scores else 0.0,
            "average_quality_score_after_repair": sum(after_scores) / len(after_scores) if after_scores else 0.0,
        }
        self.exporter.export_video_summary(video_dir, summary)
        self.progress({"video_summary": summary, "memory_summary": self.memory.summary()})
        return summary

    def _apply_memory_to_detections(self, frame, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.config.enable_embedding_memory or not self.config.enable_object_memory:
            return detections
        rows = []
        for det in detections:
            embedding = self.embedding.embed_crop(frame, det["bbox_xyxy"])
            match = self.memory.query(embedding, class_hint=str(det.get("label", "")))
            adjusted = dict(det)
            adjusted["raw_score"] = float(det.get("score", 0.0))
            adjusted["score"] = max(0.0, min(1.0, adjusted["raw_score"] + float(match.get("confidence_boost", 0.0))))
            adjusted["memory_confidence_boost"] = match.get("confidence_boost", 0.0)
            adjusted["memory_hard_negative"] = match.get("hard_negative", False)
            adjusted["memory_matches"] = match.get("matches", [])
            if not match.get("hard_negative"):
                rows.append(adjusted)
        return rows
