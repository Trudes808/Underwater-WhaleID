# 04_whale_fish_detection (WP5a): text-prompted models, no training

Truth: AI by-eye labels from WP3 (`experiments/02_reference_labels`), not ground truth. Development video
`2025-12-20 CG.m4v` (191 labelled frames at 2 per second, 31 mask frames). Everything in sections 1-5 was chosen on
the development video; section 6 applies it unchanged to the 2025-02-03 test recordings. One run each.

Units. A window is a fixed 1 s slot (`uwid.eval_presence`). Window truth: `whale` = a clear other whale on some
labelled frame of the window, `whale_free` = no clear and no faint whale on any labelled frame, `faint_only` = the
rest (reported apart, neither hit nor false alarm). Dev: 74 whale windows, 18 whale-free (12 underwater, 6 air or
crossing), 4 faint-only. Single-frame numbers use the 191 labelled frames themselves (145 whale, 37 whale-free, 9
faint-only).

## 1. Operating point (chosen on dev)

OWLv2-base (`google/owlv2-base-patch16-ensemble`, fp16, full 1920x1080 frame padded to square, 960 px input), prompt
**"whale"**; boxes lying in the burned-in clock or covering > 60 % of the frame are dropped; **2 frames per window**
(every 15th frame, 0.5 s apart); **a window has a whale when both frames have a whale box >= 0.25**.

Why: every rule and frames-per-window setting scores AUC 0.99-1.00 on dev (74 vs 18 windows cannot separate them);
the 2-frame "both" rule is the cheapest persistence rule. The threshold 0.25 sits between the highest whale-free
window (0.16) and the bulk of whale windows (all but 2 above 0.25). "humpback whale" ranks as well but squeezes the
scores (highest whale-free 0.095, its 0.25 point loses 7 windows), so "whale" leaves more margin.

| scoring at threshold 0.25 (same model, same boxes) | whale found | false alarms (whale-free) | faint-only fired |
|---|---|---|---|
| single frame (labelled frames) | 144 / 145 (99.3 %) | 8 / 37 (21.6 %) | 3 / 9 |
| window, any of its 2 frames | 74 / 74 (100 %) | 6 / 18 (33 %) | 3 / 4 |
| **window, both frames (operating point)** | **72 / 74 (97.3 %)** | **0 / 18 (0 %)** | 0 / 4 |

By situation (window rule): bubbles in view 4 / 6, clear view 68 / 68; one whale in view 2 / 4, two or more 70 / 70.
The two missed windows (87, 88 s) are one whale behind the bubble curtain at 87-88 s (score 0.19 and 0.07); sheet
`detect_dev_misses.jpg`.

Curve (window, both frames, prompt "whale"): AUC 0.995; recall 73 / 74 at no false alarm (threshold 0.19);
recall 74 / 74 needs threshold 0.075 (7 / 18 false alarms). Single frame: AUC 0.988, recall 135 / 145 at no false alarm
(threshold 0.55).

## 2. What the window step buys (same model, prompt "whale", dev)

| frames used per window | rule | AUC | recall at 0 false alarms (thr) | recall at <= 10 % false alarms (thr) |
|---|---|---|---|---|
| 1 (single labelled frame) | - | 0.988 | 0.931 (0.548) | 0.945 (0.476) |
| 2 | max | 0.990 | 0.959 (0.548) | 0.959 (0.517) |
| 2 | both frames | 0.995 | 0.986 (0.190) | 0.986 (0.160) |
| 2 | max of persistent boxes | 1.000 | 1.000 (0.112) | 1.000 (0.070) |
| 3 | max | 0.992 | 0.946 (0.531) | 0.946 (0.519) |
| 3 | 2 of 3 | 0.998 | 0.973 (0.419) | 1.000 (0.255) |
| 6 | max | 0.989 | 0.959 (0.621) | 0.959 (0.519) |
| 6 | 5 of 6 | 0.995 | 0.986 (0.221) | 0.986 (0.133) |
| 6 | max of persistent boxes | 0.989 | 0.973 (0.476) | 0.973 (0.426) |

"Persistent box" = a box backed by an overlapping box (IoU >= 0.3) in the previous or next processed frame. Pooling
whole neighbouring windows (tried first) is wrong for this question: whale-free stretches here are 1-5 s bubble
episodes between whale windows, and pooling copies the neighbours' whales into them (AUC 0.88).

What the step buys, in one line: at the same threshold, requiring the whale in both frames of the window removes
all 8 single-frame false alarms' windows (single frame 8 / 37, window 0 / 18) for 2 lost whale windows; taking the
max over the window instead makes false alarms worse (6 / 18). The 0-false-alarm curve point moves from 93 % to
99 % recall.

## 3. Prompt wording (OWLv2, dev; OWLv2 scores each prompt on its own, one pass)

| prompt | AUC frame | AUC window (both) | window recall at 0 FA (thr) | at 0.25: windows / FA | at 0.25: frames / FA |
|---|---|---|---|---|---|
| whale | 0.988 | 0.995 | 73 / 74 (0.19) | 72 / 74, 0 / 18 | 144 / 145, 8 / 37 |
| humpback whale | 0.995 | 0.995 | 73 / 74 (0.156) | 67 / 74, 0 / 18 | 137 / 145, 3 / 37 |
| a whale underwater | 0.986 | 0.994 | 72 / 74 (0.282) | 72 / 74, 0 / 18 | 144 / 145, 11 / 37 |
| whale tail | 0.976 | 0.988 | 69 / 74 (0.066) | 11 / 74, 0 / 18 | 47 / 145, 0 / 37 |
| whale fin | 0.967 | 0.991 | 66 / 74 (0.156) | 43 / 74, 0 / 18 | 103 / 145, 1 / 37 |
| whale back | 0.977 | 0.998 | 73 / 74 (0.161) | 50 / 74, 0 / 18 | 118 / 145, 1 / 37 |
| whale skin | 0.963 | 0.994 | 71 / 74 (0.118) | 9 / 74, 0 / 18 | 28 / 145, 0 / 37 |
| dolphin | 0.985 | 0.995 | 72 / 74 (0.163) | 69 / 74, 0 / 18 | 141 / 145, 8 / 37 |
| shark | 0.992 | 0.998 | 73 / 74 (0.156) | 69 / 74, 0 / 18 | 139 / 145, 5 / 37 |

Sensitivity: the RANKING hardly depends on the wording (AUC 0.96-1.00 for every animal prompt, including "dolphin"
and "shark": OWLv2 answers "large animal shape in water"), but the SCORE SCALE does, so a threshold belongs to one
prompt. Part prompts ("whale tail", "whale fin", "whale skin") score low on whole whales. "whale back" and "whale
skin" box OTHER whales, not the tagged whale's back (section 5).

## 4. Mask frames (31 frames, 85 reference whale instances), prompt "whale"

Box matched to an instance: IoU with the instance's bounding box >= 0.3, or the box holds >= 50 % of its pixels
(greedy, best score first). Mask IoU: SAM 2.1 hiera-large mask from the matched box against the reference mask.

| instances | n | found at 0.25 | found at 0.10 | median box IoU (0.25) | median SAM mask IoU (0.25) | mask IoU >= 0.5 |
|---|---|---|---|---|---|---|
| all | 85 | 71 (84 %) | 80 (94 %) | 0.82 | 0.83 | 63 / 71 |
| clear | 25 | 22 | 23 | 0.89 | 0.94 | 22 / 22 |
| hazy | 40 | 35 | 40 | 0.82 | 0.82 | 32 / 35 |
| faint | 20 | 14 | 17 | 0.65 | 0.64 | 9 / 14 |
| cut by the frame edge | 36 | 31 | 35 | 0.80 | 0.79 | 27 / 31 |
| behind bubbles | 7 | 4 | 5 | 0.66 | 0.59 | 3 / 4 |
| small (< 1 % of the frame) | 28 | 22 | 25 | 0.71 | 0.75 | 18 / 22 |

Boxes >= 0.25 on mask frames that matched no instance: 13 on open water (mostly loose boxes around a whale or far
whales not outlined), 9 on parts of whales already matched, 1 on bubbles, 2 whole-frame (dropped). None on the back.
Reference masks for hazy / faint whales are themselves SAM outlines (WP3), so mask IoU on those is SAM against SAM.

## 5. False alarms, the back, the clock (dev, prompt "whale", 0.25)

- Whale-free windows: 19 processed frames carry a box >= 0.25 (sheet `detect_dev_false_alarms.jpg`). By eye on the
  sheet: **14 on bubble clouds / entry froth / bubble streaks**, 1 at the surface edge, and **4 that show a real
  whale shape the labels do not have** (f530 / f535 in window 17: a flipper or fluke; f50, f145: whale shapes in the
  haze): the window truth comes from 2 labelled frames, so a whale seen only on other frames of the window is labelled
  whale-free. None on the back, the clock or tag parts. The both-frames rule removes all of them.
- Clock: 84 boxes lie in the clock region, all from "seafloor" (51), "air bubbles" (32) and "small fish" (1), all
  below 0.06; no whale or fish box there at >= 0.1. The filter stays, at no cost.
- Whole-frame boxes (> 60 % of the frame, mostly bubble-filled frames): 36 of the 60 boxes >= 0.25 in whale-free
  windows; dropped by rule.
- **The back as a whale: rare.** 22 "whale" boxes >= 0.25 lie in the bottom zone (touch the bottom edge, top below
  75 % of the height) on 21 of 572 frames (sheet `detect_dev_bottom_zone.jpg`); by eye about 17 of them are other
  whales swimming low in the frame (f1380-f1490, f1795-f1800, f2555-f2570), 1 is exhalation bubbles on the back
  (f1895), 3 are on the dark band just above the back (f200-f215, possibly a faint whale). On mask frames no box
  >= 0.25 lies >= 50 % on back pixels (best 0.23). A "drop boxes at the bottom" rule would therefore cost real
  whales and remove almost nothing; it changed no window and no curve. The "does not move" rule (same box +-2 s,
  IoU >= 0.6 in 60 % of frames) flagged 0 boxes.
- **The back as a class:** text prompts cannot find it. "whale back" and "whale skin" never put a box mostly on the
  back pixels at >= 0.25 (27 mask frames with a back mask; best 0.16 and 0.25), and SAM masks from those boxes
  overlap the back with IoU 0.

## 6. LLMDet-large (`iSEE-Laboratory/llmdet_large`, transformers 5.20.1 `MMGroundingDinoForObjectDetection`)

**Its boxes are wrong in this transformers version.** On the docs image (COCO 000000039769, two cats) it returns
cats at y 181-369 and 97-248 where they span about 52-472 and 24-374, a remote control 5 px tall: widths and x are
right, heights come out at about 0.4-0.5x. Same in fp32, on CPU, with custom kernels off, with square-resized or
square-padded input and with my own preprocessing; the checkpoint loads with 0 missing / unexpected keys. Not
debugged further (no fix found by reading the box decoding code). So LLMDet boxes are not used for masks, the back
or fish locations; only its scores are compared. Prompts in one pass compete, so each set is one run.

Same 286 dev frames (every 10th; 3 per window), 92 labelled frames, 74 whale / 18 whale-free windows:

| run : prompt | single frame AUC | single frame recall at 0 FA | window "2 of 3" AUC | window "2 of 3" recall at 0 FA |
|---|---|---|---|---|
| OWLv2 : whale | 0.984 | 67 / 72 | 0.998 | 72 / 74 |
| OWLv2 : humpback whale | 0.992 | 68 / 72 | 0.999 | 73 / 74 |
| LLMDet set A ("whale", "fish") : whale | 0.964 | 50 / 72 | 0.972 | 68 / 74 |
| LLMDet set B ("humpback whale", "fish", "air bubbles") : humpback whale | 0.995 | 71 / 72 | 0.995 | 73 / 74 |

Wording matters more for LLMDet than for OWLv2 (set A vs B). Fish on the dev clip (no fish): "fish" >= 0.3 on 94 /
286 frames (set A) and 46 / 286 (set B), >= 0.4 on 15 and 6 frames. Speed 1.8-2.0 frames/s including decoding,
4.1 GB peak at batch 1 (it took 9.2 GB at batch 2: stopped and re-run at batch 1). Verdict: no gain over OWLv2 for
presence, 4x slower, unusable boxes in this transformers version: not carried to the test recordings.

## 7. Mask2Former Swin-L (ADE20K semantic, COCO panoptic), 540 px short side, fp16

What it calls each reference class on the 31 mask frames (share of the class's pixels):

| reference class | ADE20K | COCO panoptic |
|---|---|---|
| open water | water 100 % | sea 100 % |
| other whales | water 62 %, person 22 %, rock 10 %, animal 3 % | sea 64 %, food 8 %, bird 7 %, bear 6 % |
| the back | water 82 %, earth 9 %, rock 6 %, floor 3 % (seafloor-like 18 %) | sea 78 %, sand 22 % |
| bubbles | water 99 % | sea 100 % |
| surface from below | water 96 %, waterfall 4 % | sea 100 % |
| air (above water) | sky 49 %, water 48 % | sky 71 %, sea 29 % |

Seafloor-like classes fixed beforehand: ADE `sand, rock, earth, floor, ground, mountain, hill, land, field, dirt
track`; COCO `sand, rock-merged, dirt-merged, gravel, mountain-merged, floor-other-merged`. **Back called seafloor**:
the dev clip has no seafloor, yet seafloor-like pixels cover > 5 % of the frame on 19 / 191 labelled frames (ADE)
and 29 / 191 (COCO), > 1 % on 67 and 88, almost all on the back at the bottom. It never names a whale "whale" (ADE
has no whale class; "animal" 3 %), so it is no whale detector, and it cannot tell the back from water. Load report:
the Swin final layernorm is "missing" (not used by Mask2Former's multi-scale features) and the relative-position
index buffers "unexpected" (recomputed): harmless. Speed 10 frames/s at 540 px.

## 8. The back as a class: SAM 2.1 prompted by position (no text, no training)

The back fills the bottom of the frame and does not move, so the cheapest "back" detector is SAM 2.1 with a fixed
prompt. Prompts fixed before looking: A = box over the bottom 25 % of the frame, B = three points along the bottom
edge (x 25 / 50 / 75 %, y 97 %), C = both. Against the reference back masks (27 of the 31 dev mask frames have one):

| prompt | median IoU with the back mask | IoU >= 0.5 | IoU >= 0.7 | area returned on the 4 frames without a back (median) |
|---|---|---|---|---|
| A bottom box | **0.81** | 25 / 27 | 21 / 27 | 11 % of the frame |
| B bottom points | 0.66 | 18 / 27 | 12 / 27 | 9 % |
| C box + points | 0.71 | 22 / 27 | 14 / 27 | 10 % |

It always returns something, so "is the back in view" still needs a check (on dev the back is labelled on 182 /
191 frames: present almost always under water). On the 2025-02-03 recordings WP3 notes the back is "a band at the
bottom of every underwater frame, often only faint" with a knobbed hump at bottom centre: a different look, not
tested here.

## 9. Cost (NVIDIA RTX 4000 Ada, 20 GB, shared; model time only, `cost.json`; 1920x1080 frames)

| model | frames / s | peak GPU memory | GPU hours per hour of video at 2 frames per 1 s window |
|---|---|---|---|
| OWLv2-base, 17 prompts, batch 4, fp16 | 7.4 | 0.7 GB | 0.27 |
| OWLv2-base, 1 prompt | 7.9 | 1.0 GB | 0.25 |
| SAM 2.1 hiera-large, 3 boxes on one frame, fp32 | 3.7 | 1.4 GB | 0.54 (only needed for masks, not for presence) |
| LLMDet-large, 2 prompts, batch 1, fp16 | 2.0 | 4.0 GB | 1.0 |
| Mask2Former Swin-L ADE / COCO, 540 px, fp16 | 8.7 / 13.8 | 2.0 GB | 0.23 / 0.15 |

End to end with decoding in a second thread, OWLv2 ran a 30 min test recording (4,386 frames: every 15th frame
plus 845 labelled frames) in 641 s = 6.8 frames/s, i.e. **0.36 h of GPU per hour of video** at 2 frames per window
with a whole decode of the H.264 file. 6 frames per window would triple it.

## 10. YOLOE-26l (ultralytics 8.4.0, `yoloe-26l-seg.pt` + MobileCLIP2 text encoder; licence AGPL-3.0)

Installed into this repo's `.venv` (`ultralytics==8.4.0` and ultralytics' `CLIP` fork; torch unchanged); weights in
`underwater_cache/detect/weights/`. Checked on the docs cats image first: boxes right. 1280 px input, fp16, batch 8.
Each box carries ONE class (the prompts compete), unlike OWLv2.

Dev, same 572 frames as OWLv2 (every 5th), 7 prompts in one run (whale, humpback whale, fish, school of fish,
seafloor, air bubbles, whale back):

| run : prompt | single frame AUC (182 frames) | recall at 0 FA | window AUC, max / 5 of 6 | window recall at 0 FA, max |
|---|---|---|---|---|
| OWLv2 : whale | 0.988 | 135 / 145 | 0.989 / 0.995 | 71 / 74 |
| YOLOE-26l : whale | 0.942 | 104 / 145 | 0.970 / 0.960 | 67 / 74 |
| YOLOE-26l : humpback whale | 0.888 | 67 / 145 | 0.854 / 0.831 | 27 / 74 |

A run with the single prompt "whale" and one with "whale" + "fish" give identical whale numbers (no competition
effect here). Fish on the fishless dev clip: "fish" >= 0.5 on 348 / 572 frames (84 / 96 windows), >= 0.3 on 482. Speed: **29.8 frames/s including
decoding** (604 frames in 20 s), i.e. about 4x OWLv2: 0.07 GPU hours per hour of video at 2 frames per window
(decoding a whole 30 min H.264 file then sets the pace).
Verdict on dev: the fast option is clearly weaker than OWLv2 for whales.

## 11. SAM 3 (`facebook/sam3`, transformers `Sam3Model`, fp16, SDPA attention)

Access: the user's Hugging Face login appeared during this work; the gated repo answered 403 ("awaiting review") at
20:00 and was readable at 20:35. Downloaded without the duplicate `sam3.pt` (3.4 GB). **Runs on this machine's torch
2.6.0** through transformers 5.20.1 (checked on the docs cats image on CPU: both cats and both remotes, boxes right).
The image is encoded once per frame and reused for every prompt. First dev run took 15 GB of GPU memory (allocator
cache; stopped at once): all runners now cap themselves at 8.5 GB (`UWID_GPU_GB`); with the cap, 4.2 GB peak
allocated. Speed 1.8 s per frame for 8 prompts (0.6 frames/s with decoding): about 0.9-1.0 GPU hours per hour of
video at 2 frames per window, 3-4x OWLv2.

Dev, the 191 frames every 15th (2 per window; = the labelled frames), prompts whale, humpback whale, fish, school of
fish, seafloor, sand, air bubbles, whale back:

| run : prompt | single frame AUC | recall at 0 FA | window AUC: max / both frames | window recall at 0 FA (max) |
|---|---|---|---|---|
| OWLv2 : whale (same frames) | 0.988 | 135 / 145 | 0.990 / 0.995 | 71 / 74 |
| SAM 3 : whale | 0.982 | 132 / 145 | 0.988 / 0.968 | 69 / 74 |
| SAM 3 : humpback whale | 0.996 | 138 / 145 | 0.999 / 0.983 | 73 / 74 |

Mask frames, SAM 3's own masks against the reference (instance found = mask IoU >= 0.3):

| SAM 3 "whale" | n | found >= 0.5 | found >= 0.3 | median mask IoU (0.3) | IoU >= 0.5 (0.3) |
|---|---|---|---|---|---|
| all | 85 | 58 | 71 | 0.89 | 66 |
| clear | 25 | 21 | 23 | 0.94 | 23 |
| hazy | 40 | 31 | 35 | 0.85 | 34 |
| faint | 20 | 6 | 13 | 0.66 | 9 |
| cut by frame edge | 36 | 26 | 30 | 0.90 | 29 |
| behind bubbles | 7 | 3 | 3 | 0.70 | 3 |
| small | 28 | 15 | 22 | 0.76 | 20 |

(OWLv2 boxes + SAM 2.1: 71 found at 0.25, median IoU 0.83, 63 at IoU >= 0.5.) SAM 3's masks are tighter; it finds
the same number of whales at 0.3.

Share of each reference class covered by the union of a prompt's masks (score >= 0.3; 0.5 in brackets):

| prompt | back | other whales | air | "unsure" (the former dev fish) | water | bubbles |
|---|---|---|---|---|---|---|
| seafloor | **0.75** (0.69) | 0.01 | 0 | 0 | 0.007 | 0.01 |
| fish | 0.15 | **0.47** (0.31) | 0.25 | 0.62 | 0.02 | 0.03 |
| whale | 0.15 | 0.86 (0.77) | 0.26 | 0.64 | 0.03 | 0.01 |
| whale back | 0 | 0.56 | 0 | 0 | 0.004 | 0 |
| sand | 0.10 | 0 | 0 | 0 | 0.001 | 0 |
| air bubbles | 0 | 0 | 0.03 | 0.006 | 0.002 | 0.09 |

So on a clip with no seafloor and no fish: **"seafloor" segments the tagged whale's back** (>= 0.5 on 112 / 191
frames, 64 / 96 windows; 75 % of back pixels, almost nothing else), **"fish" lands on whales** (>= 0.5 on 99 / 191
frames, 61 / 96 windows) and on the dark gear in air, and "whale back" segments OTHER whales. "air bubbles" barely
covers the bubbles (9 %).

## Rerun (every GPU command under `flock /var/tmp/sat3737-parts-cache/gpu_train.lock`; detections are cached
under `underwater_cache/detect/<video>/<run>/` and resume where they stopped)

```
D=experiments/04_whale_fish_detection; PY=.venv/bin/python; LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
V="/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
$LOCK $PY $D/run_detect.py --video "$V" --model owl --prompts owl_all --stride 5          # OWLv2, dev
$LOCK $PY $D/run_detect.py --video "$V" --model llmdet --prompts llmdet_A --stride 10     # LLMDet (also llmdet_B)
$LOCK $PY $D/run_detect.py --video "$V" --model sam3 --prompts sam3_all --stride 15       # SAM 3
$LOCK $PY $D/run_detect.py --video "$V" --model yoloe --prompts yoloe_all --stride 5      # YOLOE-26l
$LOCK $PY $D/analyze_masks.py --run owl_owl_all_s5 --sam                                 # SAM 2.1 masks from boxes
$LOCK $PY $D/m2f.py run --frames-dir <reference>/2025-12-20_CG/frames --tag 2025-12-20_CG
$LOCK $PY $D/back_sam.py ; $LOCK $PY $D/sam3_masks.py run ; $LOCK $PY $D/cost.py
cd $D   # CPU analysis (dev):
$PY analyze_presence.py --run owl_owl_all_s5 ; $PY operating_point.py ; $PY report_dev.py
$PY analyze_masks.py --run owl_owl_all_s5 --query whale --thr 0.25 ; $PY compare_models.py --runs owl_owl_all_s5:whale ...
$PY m2f.py dev ; $PY sam3_masks.py report --thr 0.3 ; $PY make_sheets.py dev
# test pieces 00004-00010 (2025-02-03, .m4v as labelled by WP3): experiments/04_whale_fish_detection/jobs/owl_test*.sh,
# sam3_test.sh, m2f_test.sh; then
$PY score_test.py --sheets --fish-sheets ; $PY test_masks.py ; $PY test_masks.py --sheet
$PY back_sam.py --ref <reference>/CC-11-96-20250203-*      # CPU is fine (CUDA_VISIBLE_DEVICES="")
```

## 12. TEST: 2025-02-03 pieces 00004-00008 (another day, another tagged whale; nothing tuned here)

Same OWLv2 run settings, `.m4v` (the form WP3 labelled), every 15th frame over the whole 30 min (2 per window) plus
WP3's 120 labelled frames per piece; 148 min of video. Window truth = the ONE labelled frame inside each window
(WP3 stepped through its 180 s windows for events only): 600 windows, 175 whale, 389 whale-free (374 underwater, 15
air: the drone flight), 36 faint-only. Almost all whale windows hold ONE other whale (170 / 175; the calf).

| scoring at the dev operating point ("whale" >= 0.25) | whale found | false alarms (whale-free) | of which underwater | faint-only fired |
|---|---|---|---|---|
| single frame (the labelled frame) | 165 / 175 (94.3 %) | 13 / 389 (3.3 %) | 6 / 374 | 10 / 36 |
| window, any of its 2 frames | 168 / 175 (96.0 %) | 15 / 389 (3.9 %) | | |
| **window, both frames (dev operating point)** | **157 / 175 (89.7 %)** | **6 / 389 (1.5 %)** | **3 / 374 (0.8 %)** | 7 / 36 |

Per piece (window, both frames): 00004 25 / 26 and 3 / 88 (all 3 on the drone, in air); 00005 25 / 27 and 0 / 87;
00006 58 / 64 and 2 / 42; 00007 30 / 36 and 0 / 81; 00008 19 / 22 and 1 / 91. AUC: window 0.994, single frame 0.994.
Pooled window curve: no false alarm at threshold 0.57 (111 / 175 found); <= 1 % at 0.41 (137 / 175); <= 2 % at 0.24
(159 / 175, 7 false); <= 5 % at 0.12 (169 / 175, 19 false). The dev threshold 0.25 lands at the 2 % point.

What the window step bought on test: half the false alarms (13 -> 6; underwater 6 -> 3) for 8 lost whale windows
(165 -> 157). On 00006 alone it was a loss (60 / 64, 1 false -> 58 / 64, 2 false). With one labelled frame per
window the "whale" truth comes from that frame only; a whale leaving the view in the window's other frame counts as a
miss of the rule.

Test false alarms (sheets `detect_test_<piece>_false_alarms.jpg`), by eye: 3 windows on the drone's own parts in air
before the drop (dark housing at the frame edge, the hanging tag: "whale" 0.27-0.45); underwater 1 on a school of
fish (00008, 934 s: several small "whale" boxes 0.3-0.5 on fish), 1 on fish in haze (00006, 1179 s), 1 on fish plus
the back's knobbed hump (00006, 837 s). No bubble, surface or clock false alarm (bubbles are rare on this day: 4 of
840 labelled frames).

Test misses (sheets `detect_test_<piece>_misses.jpg`), by eye: whales cut by the frame edge or the surface (top
edge, corners), small far whales in haze, a fluke behind the back's hump, and **very near whales filling the
frame** (00006 f5070, 00007 f4080: OWLv2 scores them 0.72-0.89, but the box covers > 60 % of the frame and the
dev-chosen "whole-frame box" filter drops it). Post-hoc check, not a re-tune: without that filter test windows go
157 -> 159 / 175 with the same 6 false alarms, while dev false alarms go 0 -> 3 / 18; the filter needs a score
condition (bubble-filled frames scored ~0.2 on dev, near whales 0.7-0.9 here).

Fish, test (OWLv2, any fish prompt, threshold 0.5 set from the dev false-alarm rate only; dev has no fish):
- Labelled frames: 43 with fish (WP3 stricter rule: body and tail) / 557 without: AUC 0.963; at 0.5 28 / 43 found,
  12 / 557 false; recall 0.81 at <= 5 % false alarms. Windows (both frames): 20 / 43, 10 / 557, AUC 0.934.
- WP3's 103 fish boxes: 59 found at >= 0.5, 97 at >= 0.3 (median fish 66 px long; on the 4 mask frames of 00004:
  <= 60 px 3 / 8, > 60 px 6 / 10 at 0.5).
- Whales called fish: 1,764 of 7,011 fish boxes >= 0.5 lie on a "whale" box >= 0.25 (IoU >= 0.5): about 1 in 4.
  On 00007 (no fish labelled) all 13 confident fish places are whale tail stocks / flukes ("small fish") or the sea
  surface seen from below ("school of fish"). Fish called whale: on 00004 f8938 three "whale" boxes 0.25-0.43 lie on
  jacks; the underwater whale false alarms above are all fish.
- Confident fish candidates over the WHOLE recordings (best box per window, top 96 per piece) for a human to judge:
  `detect_fish_test_0000{4..8}.jpg` (00004: 1,324 places >= 0.5 in 210 windows, mostly real jacks by eye).

Seafloor, test: WP3 found no seafloor in any of the 840 labelled frames (looked for on purpose), so only the false
side can be measured. Mask2Former seafloor-like pixels > 5 % of the frame on underwater labelled frames: ADE 8 / 346
and COCO 21 / 346 on pieces 00004-00006 (near whales and the back's hump), never > 20 %. SAM 3 "seafloor" >= 0.5 on
181 / 228 underwater labelled frames of 00004-00005 (the back). A real seafloor test needs footage with a bottom.

The back, test (14 WP3 mask frames, 11 with a back mask): SAM 2.1 with the dev-chosen bottom-25 % box: median IoU
0.69, >= 0.5 on 10 / 11, >= 0.7 on 5 / 11 (dev 0.81, 25 / 27, 21 / 27); the back here is a thin band.
OWLv2 "whale" boxes on the back: 1 of the 6 underwater false alarms involves the hump; on the 14 mask frames none.

Test mask frames (11 of WP3's 14 lie in 00004-00008; sheet `detect_test_maskframes.jpg`): whales found 8 / 8 at
0.25; fish 17 / 37 at 0.5 and 36 / 37 at 0.3 (median 75 px; <= 60 px 6 / 13, > 60 px 11 / 24 at 0.5). **Fish called
whale is common at close range**: 14 of the 33 "whale" boxes >= 0.25 on these frames lie on fish (fish pixels >= 25 %
of the box, whale pixels < 10 %; by eye on the sheet: single jacks boxed "whale" 0.3-0.55 on 00006 f13991, f41576 and
00008 f28007, often the same fish that also has a "fish" box). No "fish" box >= 0.5 lies mostly on whale pixels on
these frames (the whale-called-fish cases on the whole recordings are tail stocks and flukes, see above).

## 13. SAM 3 on test labelled frames (single frames, prompts whale / fish / seafloor; pieces 00004-00005 only so
far; `jobs/sam3_test.sh` continues with 00006-00010 when the GPU lock frees, then `score_test.py` reports them)

| SAM 3 prompt | frames: positive / negative | AUC | >= 0.3 | >= 0.5 | >= 0.7 |
|---|---|---|---|---|---|
| whale (vs whale-free) | 53 / 175 | 0.961 | 48 found, 18 false | 41, 12 | 30, 2 |
| fish (vs no fish) | 17 / 223 | 0.954 | 17, 104 false | 17, 67 | 17, 45 |
| seafloor (none labelled) | 0 / 228 underwater | - | 197 false | 181 | 105 |

OWLv2 on the same 228 frames: whale AUC 0.995, 50 / 53 found with 8 / 175 false at 0.25. SAM 3 is weaker on
whales here, calls far more frames "fish", and calls the back "seafloor" on most frames.
