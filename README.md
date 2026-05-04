# Prompt Video Segmenter

A first-person video object segmentation tool built for kitchen and workflow monitoring scenarios. It supports multiple detector and segmenter backends, with a focus on the YOLO11-seg pipeline fine-tuned on Kitchen VISOR data.

**Key features:**
- GUI-based video processing with real-time preview
- Pluggable detector backends: YOLO11-seg (recommended), GroundingDINO, YOLO-World, RF-DETR
- Pluggable segmenter backends: SAM2, YOLO11-seg passthrough
- ByteTrack multi-object tracking + temporal label smoothing
- COCO-format annotation export
- Separate training GUI for fine-tuning YOLO11-seg on custom datasets

## Demo

<video src="https://github.com/Zihao-Sheng/Prompt_Segmenter_Model/blob/main/annotated_video.mp4" controls width="100%"></video>


---

## Table of Contents

- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Datasets](#datasets)
- [Research Background](#hangar-maintenance-workflow-monitoring-project-research-progress-summary)

---

## Directory Structure

```
Prompt_Segmenter_Model/
├── configs/                    # YAML config files for inference and training
│   ├── yolo11_demo.yaml        # Recommended: YOLO11-seg only, no heavy models needed
│   ├── prompt_segment_demo.yaml# Full pipeline (GroundingDINO + SAM2 + SegFormer)
│   └── train_kitchen_coarse.yaml  # Training config for 8-class coarse model
│
├── models/                     # Model weights (downloaded via Download_Models.bat)
│   ├── kitchen_coarse_v2.pt    # Fine-tuned YOLO11s-seg, 8 kitchen classes
│   ├── sam2/                   # SAM2 checkpoint and config
│   ├── segformer_b0_ade/       # SegFormer scene segmentation model
│   ├── mediapipe/              # Hand landmark model
│   └── ...
│
├── scripts/                    # Utility scripts
│   ├── download_models.py      # Downloads weights from GitHub Releases
│   └── visor_to_yolo.py        # Converts VISOR annotations to YOLO format
│
├── src/                        # Source code
│   ├── gui_app.py              # Main GUI entry point (launched by Launch_GUI.bat)
│   ├── detection/              # Detector backends (YOLO11, GroundingDINO, etc.)
│   ├── segmentation/           # Segmenter backends (SAM2, YOLO11 passthrough)
│   ├── tracking/               # ByteTrack + label smoothing + hand trigger
│   ├── pipeline/               # Frame processing pipeline and export
│   ├── gui/                    # Training GUI
│   └── core/                   # Config loading, types, utilities
│
├── Launch_GUI.bat              # Start the inference GUI
├── Launch_Training_Tool.bat    # Start the training GUI
├── Install_Dependencies.bat    # Set up venv and install packages
└── Download_Models.bat         # Download model weights from GitHub Releases
```

---

## Quick Start

### Prerequisites
- **Python 3.12+** — [python.org/downloads](https://www.python.org/downloads/) (check "Add python.exe to PATH" during install)
- **NVIDIA GPU** with driver supporting CUDA 12.8+ (driver 525+ recommended)
- **~6 GB free disk space**

> No NVIDIA GPU? The app will fall back to CPU automatically — slower but functional.

### Installation

```bash
git clone https://github.com/Zihao-Sheng/Prompt_Segmenter_Model.git
cd Prompt_Segmenter_Model
```

Then double-click in order:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `Install_Dependencies.bat` | Creates `.venv`, installs PyTorch (CUDA) + all packages |
| 2 | `Download_Models.bat` | Downloads model weights (prompts you to choose a pipeline) |
| 3 | `Launch_GUI.bat` | Opens the GUI |

When running `Download_Models.bat`, select the pipeline that matches the config you want to use:

| Input | Pipeline | Download size |
|-------|----------|--------------|
| `yolo11` | YOLO11-seg only *(default, recommended)* | ~26 MB |
| `gdino` | GroundingDINO + SAM2 | ~820 MB |
| `yolo_world` | YOLO-World + SAM2 | ~255 MB |
| `full` | YOLO-World + SegFormer + SAM2 + MediaPipe | ~285 MB |
| `all` | Everything | ~1.1 GB |

### Pipeline Options

There are four inference pipelines available, each with different speed/accuracy trade-offs. Select the corresponding config in the GUI.

---

#### Option 1 — YOLO11-seg (Recommended)

**Config:** `configs/yolo11_demo.yaml`  
**Speed:** ~20 ms/frame on RTX 4060  
**Models needed:** `models/kitchen_coarse_v2.pt` (downloaded automatically)

YOLO11s-seg runs detection and segmentation in a single forward pass. It is the fastest and most stable option for real-time use. The bundled `kitchen_coarse_v2.pt` is fine-tuned on Kitchen VISOR (mAP50 ≈ 44).

**Prompt for the bundled kitchen model:**
```
hand, cookware, lid, dishware, utensil, appliance, cabinet, container
```
> These names exactly match the 8 training classes. YOLO11-seg ignores the prompt for detection — it always detects all trained classes — but the prompt is used for label normalisation and tracking. Using the exact class names gives the best tracking stability.

To use a different `.pt` file, leave the config as-is and pick your model via the **YOLO11 model (.pt)** field in the GUI.

---

#### Option 2 — GroundingDINO + SAM2

**Config:** `configs/prompt_segment_gdino15_edge_rescue.yaml`  
**Speed:** ~2–4 s/frame  
**Models needed:** `models/groundingdino_swint_ogc.pth`, `models/sam2/`

Open-vocabulary pipeline. GroundingDINO detects objects from free-form text prompts; SAM2 produces pixel-level masks. Highest quality masks but too slow for real-time use. Best for offline annotation or generating training data.

**Example prompt:**
```
hand, pot, pan, lid, bowl, plate, spoon, knife, cup
```
> You can use arbitrary natural-language object names — no fixed class list.

---

#### Option 3 — YOLO-World + SAM2

**Config:** `configs/prompt_segment_demo.yaml` (set `segmenter.backend: sam2`, `detector.backend: yolo_world`)  
**Speed:** ~300–500 ms/frame  
**Models needed:** `models/yolov8s-worldv2.pt`, `models/sam2/`

YOLO-World handles open-vocabulary detection much faster than GroundingDINO; SAM2 provides the segmentation masks. A middle ground between speed and prompt flexibility. Detection accuracy is lower than GroundingDINO without fine-tuning.

**Example prompt:**
```
hand, pot, pan, lid, bowl, plate, spoon, knife
```

---

#### Option 4 — YOLO-World + SegFormer + SAM2 (Full Pipeline)

**Config:** `configs/prompt_segment_demo.yaml`  
**Speed:** ~300–500 ms/frame  
**Models needed:** `models/yolov8s-worldv2.pt`, `models/sam2/`, `models/segformer_b0_ade/`, `models/mediapipe/`

The complete pipeline. SegFormer adds semantic scene/background segmentation (countertop, wall, floor, etc.) to separate foreground objects from background. MediaPipe hand detection enables hand-triggered object persistence. GroundingDINO runs as a rescue detector every 16 frames to recover lost objects.

This is the most accurate but also the heaviest option. Recommended only if you need full scene understanding.

**Example prompt (foreground objects only — scene labels are handled internally):**
```
hand, pot, pan, lid, bowl, plate, spoon, knife, cup, bottle
```

---

### Steps to Run

1. Launch `Launch_GUI.bat`
2. **Config** — select one of the configs above
3. **Video** — pick any `.mp4` / `.avi` file
4. **Prompt** — enter object names (see examples above)
5. Click **Start**

---

## Datasets

| Dataset | Used for | Link |
|---------|----------|------|
| **Kitchen VISOR** | Training YOLO11-seg (mask annotations for 8 kitchen object classes) | [datasetninja.com/epic-kitchens-visor](https://datasetninja.com/epic-kitchens-visor) |
| **EGTEA Gaze+** | Inference testing (first-person kitchen activity videos) | [cbs.ic.gatech.edu/fpv](https://cbs.ic.gatech.edu/fpv/) |

The VISOR annotations were converted to YOLO segmentation format using `scripts/visor_to_yolo.py`, then remapped to 8 coarse classes (appliance, cabinet, container, cookware, dishware, hand, lid, utensil) using `scripts/remap_to_coarse.py`.

---

# Hangar Maintenance Workflow Monitoring Project: Research Progress Summary

> **Purpose**: This document summarizes the current research progress for teammates. Since the partner has not yet provided a real hangar maintenance dataset, the current work uses public first-person workflow datasets and existing vision models to test whether real-time workflow monitoring is technically feasible, and to identify the main bottlenecks before moving to the actual maintenance scenario.

---

## 0. Project Background and Current Data Choice

The long-term goal of this project is to build a system that can monitor maintenance or assembly workflows from video and identify:

- key objects and tools;
- hand-object interactions;
- object states and movement;
- step-level workflow progress;
- possible missing, incorrect, or abnormal actions.

In the final hangar maintenance setting, the system should help compare what happens in the video against manuals or standard operating procedures.

Because the partner has not yet provided real hangar maintenance video data, the current experiments mainly use **EGTEA Gaze+** as a substitute testing dataset.

**EGTEA Gaze+** is a first-person kitchen activity dataset. Although the domain is different from aircraft maintenance, it is still useful for early testing because it contains many similar technical challenges:

- first-person or close-range video;
- frequent hand-object interaction;
- object occlusion by hands;
- visually similar objects;
- fast movement;
- cluttered working surfaces;
- step-like workflow structure.

Therefore, EGTEA Gaze+ is being used as a temporary proxy to evaluate whether different modeling routes are suitable for real-time workflow monitoring.

---

## 1. Stage 1: Understanding the Overall Workflow with Video Classification

### 1.1 Main Idea

The first stage focused on using **CNN-based video classification and action recognition** to understand the general meaning of a video clip.

**CNN (Convolutional Neural Network)** is a neural network architecture commonly used for image and video recognition. It extracts local visual patterns such as edges, textures, shapes, and object parts through convolutional filters. For video, CNN-based methods usually sample frames or extend the network with temporal modeling to classify what action is happening in a short clip.

In this stage, the goal was to answer a relatively high-level question:

> Can we classify what the worker is doing from a short first-person video clip?

The experiments mainly included:

- RGB-only action recognition;
- single-clip classification;
- causal sequence modeling over multiple clips;
- reproducible training and inference scripts for teammates.

### 1.2 Models and Routes Tested

#### TSM

**TSM (Temporal Shift Module)** is a lightweight video action recognition method. Instead of using expensive 3D convolution everywhere, it shifts part of the feature channels along the time dimension, allowing a 2D CNN-style backbone to capture temporal motion information.

In the current EGTEA Gaze+ experiments, TSM became the most practical single-clip baseline because it is relatively stable, not too heavy, and effective enough to serve as a feature extractor for later temporal modeling.

#### X3D-S

**X3D-S** is a lightweight 3D video network. It expands model capacity across several dimensions such as temporal resolution, width, depth, and spatial resolution, aiming for a better speed-accuracy tradeoff.

It is useful for efficiency-focused experiments, but in the current setup it was weaker than TSM as the main backbone.

#### SlowFast-R50

**SlowFast** is a two-branch video understanding model. The Slow branch captures semantic and scene-level information at a lower frame rate, while the Fast branch captures motion details at a higher frame rate.

Although it is theoretically strong for action understanding, it was heavier to run and did not outperform TSM in the current EGTEA Gaze+ setup.

#### Causal GRU

**GRU (Gated Recurrent Unit)** is a recurrent neural network structure that can use previous time steps to predict the current state. In this project, the GRU was used in a **causal** way, meaning it only looks at past and current clips, never future clips.

This is important because real-time monitoring cannot use future information.

The current recommended Stage 1 route is:

```text
single-clip encoder: TSM
context model: causal GRU
context input: embedding_only
context window: K = 5
```

This route gives a realistic online baseline: first classify individual clips, then use a lightweight temporal model to smooth and improve predictions using past context.

### 1.3 Main Finding from Stage 1

The Stage 1 result is useful, but also exposes a major limitation.

Video classification can understand the **general meaning** of a clip, such as “cutting”, “mixing”, or “putting something down”. However, it struggles with fine-grained details that are important for maintenance monitoring, such as:

- which exact object is being touched;
- whether the hand is holding, rotating, placing, or removing an object;
- whether a small part is missing;
- whether the object is partially occluded;
- whether the worker performed the correct micro-action.

This means CNN/action-recognition methods are suitable for estimating the broad workflow state, but they are not enough for detailed real-time verification.

### 1.4 VLM Trial

I also briefly tested mature **VLMs (Vision-Language Models)** such as **Qwen2.5-VL**.

**VLM (Vision-Language Model)** is a model that can take images or video frames together with text prompts and generate language-based understanding. In theory, it can describe what is happening in a frame more flexibly than a fixed classifier.

However, in practice:

- inference was too slow for real-time monitoring;
- local RTX 4060 laptop hardware could not run it smoothly for this use case;
- larger or higher-bit models may work better on stronger partner hardware, but this is not currently reproducible on my machine.

Therefore, VLMs are not the main real-time route at the moment. They may still be useful later for offline analysis, report generation, or high-level reasoning after the visual evidence has already been extracted.

### 1.5 Stage 1 Conclusion

Stage 1 shows that clip-level action recognition can provide a general workflow understanding, but it does not reliably capture the detailed visual evidence needed for maintenance verification.

For real-time monitoring, we need more explicit object-level information, especially bounding boxes, segmentation masks, tracking, and hand-object interaction signals.

---

## 2. Stage 2: Moving from Abstract Video Features to Object Detection, Segmentation, and Tracking

### 2.1 Why Stage 2 Was Needed

Stage 1 made it clear that abstract video features alone are not enough.

If the system needs to monitor what is happening in detail, it must know:

- where each important object is;
- whether the object is still visible;
- whether it is attached to or near the hand;
- whether it has moved, disappeared, or changed state;
- whether it is foreground or background.

Therefore, Stage 2 shifted toward an explicit visual pipeline based on **bounding boxes**, **segmentation**, and **tracking**.

**Bounding box detection** identifies the rectangular region containing an object. It is fast and useful for localization, but it is often too coarse when objects overlap or when the system needs precise shape information.

**Instance segmentation** predicts a pixel-level mask for each object instance. It is more precise than a bounding box and helps separate foreground objects from background surfaces.

**Tracking** links the same object across frames and assigns it a stable identity over time.

### 2.2 Roboflow Rapid-Inspired Direction

After reviewing the Roboflow Rapid online labeling and learning workflow, I decided to test a similar idea locally:

> Use strong foundation models to produce high-quality masks and labels first, then use those outputs to build a lighter real-time model.

This motivated the initial attempt with **GroundingDINO + SAM2**.

### 2.3 GroundingDINO + SAM2 Trial

**GroundingDINO** is an open-vocabulary object detection model. It can detect objects from text prompts, such as “knife”, “bowl”, or “pot”, without being trained specifically on the target dataset.

Its principle is to align image regions with language prompts and output bounding boxes for regions that match the text.

**SAM2 (Segment Anything Model 2)** is a general-purpose segmentation model. Given prompts such as points, boxes, or previous masks, it can generate pixel-level object masks. In video, it can also propagate object masks across frames.

The initial Stage 2 test used GroundingDINO to find objects and SAM2 to segment them.

The result was encouraging in terms of quality:

- segmentation quality was already good without fine-tuning;
- masks were often much better than simple bounding boxes;
- the system could identify detailed object regions.

However, the speed was not acceptable:

- on an RTX 4060 laptop GPU, each frame could take more than 2 seconds;
- this is far from real-time;
- using this directly for live monitoring is not practical.

### 2.4 YOLO-World + SAM2 + ByteTrack + SegFormer Pipeline

To improve speed, the next attempt replaced GroundingDINO with **YOLO-World** for most frames while keeping SAM2 for segmentation.

**YOLO-World** is an open-vocabulary object detection model based on the YOLO detection family. It can use text prompts to detect objects without training on a fixed closed label set. Compared with GroundingDINO, it is much faster, but in this project it was less accurate without fine-tuning.

The idea was:

- use YOLO-World for regular fast detection;
- run GroundingDINO only on the first frame and every 16 frames afterward;
- use GroundingDINO as a rescue detector to recover objects that YOLO-World missed;
- use SAM2 to produce masks from detected boxes;
- use tracking and memory to maintain object identities.

This pipeline also included several other components.

#### ByteTrack

**ByteTrack** is a multi-object tracking algorithm. It links detections across frames by matching bounding boxes and confidence scores. A key idea is that it also uses low-confidence detections instead of discarding them immediately, which can help preserve tracks during short occlusions.

In this project, ByteTrack was used to assign and maintain `track_id` values for objects across frames.

#### SegFormer

**SegFormer** is a transformer-based semantic segmentation model. Unlike instance segmentation, semantic segmentation labels pixels by category, such as countertop, wall, floor, sink, or cabinet.

In this pipeline, SegFormer was used to detect background or scene regions, so foreground objects and background surfaces could be handled separately.

#### MediaPipe Hand Detection

**MediaPipe Hands** is a lightweight hand landmark detection system. It estimates hand keypoints, bounding boxes, and hand position.

Here, it was used to infer whether an object was close to or attached to the hand, which is important for persistence and recovery.

#### CLIP Region Classifier

**CLIP (Contrastive Language-Image Pretraining)** is a vision-language model that aligns images and text into a shared embedding space. It can classify an image region by comparing the region embedding with text prompt embeddings.

In this pipeline, CLIP was considered as an optional secondary classifier for uncovered regions.

### 2.5 Stage 2 Pipeline Overview

The Stage 2 pipeline can be summarized as follows:

```text
video frame
  ↓
frame preprocessing
  ↓
hand detection
  ↓
foreground detection with YOLO-World
  ↓
tracking with ByteTrack
  ↓
temporal label smoothing and track memory
  ↓
SAM2 foreground segmentation
  ↓
short-term memory recovery for missing objects
  ↓
uncovered-region redetection
  ↓
GroundingDINO rescue detection every N frames
  ↓
SegFormer scene/background segmentation
  ↓
scene takeover guard
  ↓
foreground conflict resolution
  ↓
visualization and JSON/COCO export
```

The full implementation included many defensive mechanisms:

- hand-triggered object persistence;
- short-term track memory;
- label smoothing over a temporal window;
- rescue detection by GroundingDINO;
- uncovered-region redetection;
- scene/background mask clipping;
- foreground protection against SegFormer takeover;
- contaminated mask rejection;
- COCO-style annotation export.

### 2.6 Problems Found in Stage 2

Even after adding multiple layers of protection, the pipeline still had serious problems.

#### Problem 1: Object Loss Over Time

The first GroundingDINO frame usually had the highest quality. However, as frames progressed, objects were still frequently lost or misclassified.

Even with ByteTrack, YOLO-World redetection, memory recovery, and rescue detection, the system could not consistently maintain every important object.

This is especially problematic because workflow monitoring depends on stable object continuity.

#### Problem 2: YOLO-World Was Not Reliable Enough Without Fine-Tuning

YOLO-World was fast, but without fine-tuning it struggled with:

- occluded objects;
- visually similar objects;
- small objects;
- objects partially covered by hands;
- kitchen-specific tools and containers.

This caused unstable detections, which then weakened every downstream stage.

#### Problem 3: SAM2 Was Too Slow for Real-Time Use

SAM2 improved mask quality, but it became the main speed bottleneck.

The total per-frame processing time was often around **300–400 ms**, which is still too slow for a real-time monitoring system.

If SAM2 was removed, the pipeline became faster, but it had to rely mostly on bounding boxes. That caused much higher object loss and produced outputs that were not informative enough for later reasoning or LLM-based interpretation.

In other words, this was not just an accuracy issue. If the visual output is unstable or incomplete, the next reasoning stage may fail to understand what happened at all.

#### Problem 4: Scene Segmentation Sometimes Took Over Foreground Objects

SegFormer sometimes labeled foreground objects as part of the background scene. For example, a countertop or sink mask could overlap with or cover a movable object.

This created a “scene takeover” problem, where the background segmentation became too dominant and suppressed the object that should have been tracked.

Later tests added some protection mechanisms, such as foreground mask clipping and protected foreground regions, but this issue shows that background modeling must be handled carefully.

### 2.7 Stage 2 Conclusion

Stage 2 showed that explicit detection, segmentation, and tracking are necessary for detailed workflow monitoring.

However, it also showed that a heavy open-vocabulary pipeline is not suitable as the final real-time solution on the current hardware.

The main conclusions were:

1. **Some level of fine-tuning is necessary.** If the base detector cannot maintain basic object accuracy, no amount of memory, tracking, or rule logic can fully rescue the pipeline.
2. **The final runtime model must use a lighter backbone.** Even if GroundingDINO + SAM2 produces good quality, it is too slow for real-time deployment.
3. **Mask quality matters.** Pure bounding boxes are often too weak for detailed hand-object monitoring.

These findings motivated Stage 3.

---

## 3. Stage 3: Fine-Tuning a Lightweight YOLO11s-Seg Model

### 3.1 Why YOLO11s-Seg Was Chosen

After Stage 2, the next step was to test whether a lighter integrated segmentation model could provide both speed and acceptable object stability.

**YOLO11s-Seg** is a lightweight YOLO-based instance segmentation model. Unlike the YOLO-World + SAM2 combination, it predicts both bounding boxes and segmentation masks in one forward pass.

This is important because:

- detection and segmentation are integrated;
- there is no separate SAM2 segmentation step;
- inference is much faster;
- it is more suitable for real-time use.

In the current tests, YOLO11s-Seg took roughly **20 ms per frame**, compared with roughly **300–400 ms per frame** for the Stage 2 combined pipeline.

That is close to a 20× speed improvement.

### 3.2 Limitation: YOLO11s-Seg Is Not Open-Vocabulary

The downside is that YOLO11s-Seg is not an open-vocabulary model.

This means it cannot freely detect arbitrary text-prompted objects like YOLO-World or GroundingDINO. It needs a fixed class set and training data.

Without fine-tuning, it performed poorly in our target workflow setting:

- low accuracy;
- unstable object detection;
- weak tracking continuity;
- even when using GroundingDINO to adjust the first frame and memory logic to correct later frames, it could lose around 80% of objects by the second frame.

This made the non-fine-tuned version extremely unreliable.

### 3.3 Fine-Tuning Dataset: Kitchen VISOR

To test whether mask fine-tuning could solve the stability problem, we used the **Kitchen VISOR** dataset.

**VISOR** is a dataset with dense object masks in egocentric kitchen videos. It provides segmentation annotations for objects involved in first-person hand-object interactions.

It is not a hangar dataset, but it is valuable because:

- it contains first-person workflow videos;
- it has high-quality object masks;
- it includes hand-near objects and occlusions;
- it allows training a segmentation model before the real partner data arrives.

### 3.4 Fine-Tuning Experiment 1: 57 Fine-Grained Object Classes

The first training experiment used 57 object categories under 8 broader kitchen object groups.

Result:

```text
YOLO11s-Seg, 57 classes
mAP50 ≈ 25
```

This result was not very high. One likely reason is that fine-grained object classification is difficult when many kitchen objects look similar or are partially occluded.

For example, the model may find the object region but confuse the exact class.

### 3.5 Fine-Tuning Experiment 2: 8 Coarse Object Classes

The second experiment simplified the label space from 57 fine-grained classes to 8 coarse categories.

Result:

```text
YOLO11s-Seg, 8 coarse classes
mAP50 ≈ 44
```

This result was significantly better.

The improvement suggests that coarse category detection is currently more stable than fine-grained classification, especially when the dataset does not consistently annotate every visible object.

### 3.6 Important Dataset Issue

A key issue with the VISOR dataset is that it mainly masks objects close to the hands. Objects farther from the hands may not always be annotated, even if they are visible.

This creates a contradiction during training:

- the model sees some visible objects without masks;
- it may learn that far-away objects should be ignored;
- this can make detection inconsistent.

Another issue is that the model often finds the object but misclassifies its class.

This is likely not a fundamental failure of YOLO11s-Seg. It may be improved by:

- a better dataset;
- cleaner annotation rules;
- a more suitable class hierarchy;
- post-processing with tracking and temporal smoothing.

### 3.7 Training Scale and Runtime

The fine-tuning experiments used more than **32,000 mask-annotated images**.

Both training runs converged within around **25 epochs**.

The training time was slightly more than **4 hours**, which is acceptable for our project workflow.

This suggests that if we can obtain or generate mask annotations for the real maintenance domain, fine-tuning a lightweight segmentation model is practical.

### 3.8 Main Finding from Stage 3

The most important result is that mask fine-tuning greatly improved object stability.

After training, YOLO11s-Seg could track objects much more consistently even without using memory or ByteTrack.

The remaining problem was mainly label switching during continuous object judgment. For example, the model may temporarily confuse similar categories across frames.

This is a much easier problem than losing the object completely.

Label switching can likely be improved with:

- ByteTrack;
- temporal label smoothing;
- coarse-to-fine label hierarchy;
- class-specific rules;
- better training annotations.

### 3.9 Stage 3 Conclusion

Stage 3 shows that proper mask fine-tuning is necessary and valuable.

The key conclusion is:

> A lightweight segmentation model fine-tuned with mask annotations is likely a better real-time foundation than a heavy open-vocabulary model stack.

The heavy models are still useful, but more as annotation or rescue tools rather than as the main runtime model.

---

## 4. Stage 4: Automatic Labeling for the Real Hangar Maintenance Domain

### 4.1 Why Stage 4 Is Needed

If we move from kitchen videos to hangar maintenance equipment, the biggest challenge is data annotation.

Unlike kitchen datasets, we probably will not have a ready-made dataset with masks for aircraft parts, tools, panels, and maintenance objects.

Therefore, the main Stage 4 question is:

> How can we create useful mask annotations for a new maintenance environment without manually labeling everything from scratch?

This is the current research direction.

### 4.2 Proposed Direction: Heavy Model Auto-Labeling + Human Review

The current plan is to use heavy models such as GroundingDINO and SAM2 to automatically generate initial masks and object candidates.

The idea is:

1. run strong detection/segmentation models offline on training frames;
2. generate masks for unknown or repeated objects;
3. cluster similar object masks/features;
4. assign labels to clusters instead of individual frames;
5. manually correct only ambiguous or high-value cases;
6. train a lightweight YOLO11s-Seg or similar model from the cleaned annotations.

This turns heavy models into an **offline annotation engine**, not the final real-time model.

### 4.3 Clustering Unknown Objects

**Clustering** is an unsupervised learning method that groups similar samples together. In this context, each segmented object crop or mask can be represented by visual features, and similar objects can be grouped into the same cluster.

The expected workflow is:

```text
raw maintenance video frames
  ↓
GroundingDINO / SAM2 segmentation
  ↓
object crops and masks
  ↓
feature extraction
  ↓
clustering of similar unknown objects
  ↓
human labels each cluster instead of every frame
  ↓
cleaned mask dataset
  ↓
fine-tune lightweight segmentation model
```

This may greatly reduce manual work. Instead of labeling tens of thousands of masks one by one, a human may only need to review a much smaller set of object clusters.

The target is to reduce the manual burden so that humans may need to handle fewer than about **1,000 label decisions**, depending on the actual data complexity.

### 4.4 Self-Supervised Learning Possibility

**Self-supervised learning** is a training approach where a model learns useful representations from unlabeled data by solving automatically generated tasks. For example, it may learn that two augmented views of the same object should have similar features.

In this project, self-supervised learning may help by:

- improving object embeddings without manual labels;
- grouping visually similar parts;
- learning environment-specific features from maintenance videos;
- reducing dependence on large labeled datasets.

This is still exploratory, but it fits the Stage 4 goal of lowering annotation cost.

### 4.5 Stage 4 Goal

The goal of Stage 4 is not to make a perfect model immediately.

The goal is to build a practical annotation pipeline:

- use heavy models to over-generate candidate masks;
- use clustering to organize unknown objects;
- let humans label groups instead of individual frames;
- train a fast model for real-time inference;
- repeat the process with active learning or correction.

---

## 5. Overall Technical Lessons

### 5.1 High-Level Video Classification Is Not Enough

CNN/action-recognition models are useful for understanding the general activity, but they do not provide enough detail for maintenance verification.

They can answer:

> What is the general action?

But they struggle to answer:

> Which exact part was touched, moved, rotated, installed, or missed?

For this project, the second type of question is more important.

### 5.2 Real-Time Monitoring Needs Explicit Visual Evidence

A practical system needs explicit visual evidence such as:

- object bounding boxes;
- instance masks;
- track IDs;
- hand-object proximity;
- object persistence;
- temporal label stability;
- foreground-background separation.

Without this, later reasoning modules or LLMs do not have reliable evidence to analyze.

### 5.3 Heavy Open-Vocabulary Models Are Good for Labeling, Not Runtime

GroundingDINO, SAM2, and similar models are valuable because they can detect and segment objects without much task-specific training.

However, they are too slow for real-time monitoring on the current RTX 4060 laptop setup.

Their best role is likely:

- offline annotation;
- rescue detection;
- initial dataset creation;
- human-in-the-loop labeling support.

They should not be the default real-time inference backbone unless stronger hardware changes the constraint.

### 5.4 Lightweight Fine-Tuned Segmentation Is the Most Promising Runtime Route

YOLO11s-Seg shows that a lightweight segmentation model can be fast enough for real-time use.

The problem is that it needs domain-specific mask training.

Once fine-tuned on mask data, object continuity improves significantly. This makes it a stronger candidate for the final runtime system than a heavy multi-model pipeline.

### 5.5 Dataset Quality Matters More Than Model Tricks

Stage 2 showed that complicated memory and rescue logic cannot fully compensate for weak base detections.

Stage 3 showed that even a small segmentation model becomes much more useful after mask fine-tuning.

Therefore, the most important bottleneck is likely not only model architecture, but also how to obtain high-quality mask annotations for the target maintenance domain.

---

## 6. Current Recommended System Direction

Based on the current experiments, the most practical future system should use a two-level design.

### 6.1 Offline Heavy Annotation Pipeline

Use strong but slow models to generate training data:

```text
GroundingDINO / SAM2 / VLM-assisted tools
  ↓
automatic mask proposals
  ↓
object clustering
  ↓
human correction and labeling
  ↓
clean maintenance-domain mask dataset
```

This stage prioritizes quality over speed.

### 6.2 Online Lightweight Runtime Pipeline

Train a fast segmentation model for real-time monitoring:

```text
fine-tuned YOLO11s-Seg or similar lightweight segmentation model
  ↓
ByteTrack / temporal smoothing
  ↓
hand-object interaction logic
  ↓
workflow state monitoring
  ↓
manual/SOP comparison or alert generation
```

This stage prioritizes speed, stability, and deployability.

---

## 7. Suggested Next Steps

### Step 1: Continue the Automatic Labeling Experiment

The immediate next step is to test whether GroundingDINO + SAM2 can produce useful masks for unknown objects and whether those masks can be clustered effectively.

The key evaluation question is:

> Can we reduce human labeling work from frame-by-frame annotation to cluster-level correction?

### Step 2: Define a Maintenance-Oriented Class Hierarchy

Before collecting or labeling real maintenance data, we should define a label hierarchy.

For example:

```text
coarse class: tool
  fine classes: wrench, screwdriver, drill, pliers

coarse class: fastener
  fine classes: bolt, nut, screw, washer

coarse class: panel/component
  fine classes: access panel, cable, connector, bracket
```

The VISOR experiment suggests that coarse labels are much easier to stabilize than fine-grained labels. A coarse-to-fine structure may be the best compromise.

### Step 3: Build a Small Proof-of-Concept Maintenance Dataset

Once partner data becomes available, we should not start by labeling everything.

A better first step is:

- sample a small number of representative clips;
- run heavy auto-labeling;
- cluster object masks;
- manually correct a small subset;
- train a lightweight segmentation model;
- evaluate whether tracking stability improves.

### Step 4: Add Temporal Tracking After the Base Model Is Stable

Tracking and memory should be added after the detector is reasonably accurate.

Otherwise, the tracker only propagates incorrect or unstable detections.

A good order is:

```text
1. train stable segmentation model
2. add ByteTrack
3. add temporal label smoothing
4. add hand-object interaction rules
5. add workflow-level reasoning
```

### Step 5: Use VLMs Later for Reasoning, Not Raw Real-Time Perception

VLMs may still be useful, but they should probably receive structured visual evidence rather than raw video frames.

For example, instead of asking a VLM to inspect every frame directly, we can provide:

```text
frame t:
- object A: wrench, track_id=3, near right hand, moving toward panel
- object B: screw, track_id=7, disappeared after contact with panel
- hand state: grabbing
- current workflow step candidate: fastening
```

This would make VLM reasoning cheaper, more interpretable, and less dependent on high frame-level inference speed.

---

## 8. Short Summary for Teammates

The project started with EGTEA Gaze+ because real hangar data is not available yet.

The first route used CNN/video action recognition models such as TSM, X3D-S, SlowFast, and causal GRU. This can understand the general action in a video clip, but it cannot reliably capture detailed hand-object interactions.

The second route tested explicit object detection, segmentation, and tracking using GroundingDINO, YOLO-World, SAM2, ByteTrack, SegFormer, and memory recovery. This produced better object-level evidence, but the heavy model combination was too slow and still unstable without fine-tuning.

The third route fine-tuned YOLO11s-Seg on Kitchen VISOR mask data. This was much faster, around 20 ms per frame, and after mask fine-tuning it showed much more stable object tracking. The 8-class coarse-label experiment performed better than the 57-class fine-grained experiment, suggesting that coarse-to-fine labeling is a better strategy.

The current fourth stage focuses on automatic labeling for the future hangar maintenance domain. Since we probably will not have ready-made mask annotations, the plan is to use heavy models such as GroundingDINO and SAM2 offline, cluster unknown object masks, label clusters manually, and then train a lightweight segmentation model for real-time use.

The main conclusion is:

> For real-time maintenance workflow monitoring, the best route is likely not to run a large VLM or heavy open-vocabulary model directly. Instead, we should use heavy models offline to create training labels, then fine-tune a lightweight segmentation model for fast and stable runtime perception.
