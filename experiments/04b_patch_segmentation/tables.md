# 04b_patch_segmentation (WP5b): a linear classifier on frozen DINOv3 patch features

Run 2026-11-20. Truth everywhere = AI by-eye labels (WP3), not ground truth. One run per row unless stated.

## What was fitted on what

| model | fitted on | scored on |
|---|---|---|
| fold classifiers (4) | dev clip `2025-12-20 CG.m4v`, mask frames of 3 time blocks (frames within 3.3 s of the held-out block dropped) | the held-out block's mask frames (31 frames over the 4 folds) and, for windows, every dev frame of that block |
| `clf_w1280_h0_all.pt` | all 31 dev mask frames (58,171 water / 7,675 whale / 5,450 back / 20,413 bubbles / 8,673 surface / 6,946 air patches; no fish, no seafloor, no gear) | the 7 test recordings 2025-02-03 00004-00010 (840 labelled frames, 14 mask frames) |
| input size, head, presence thresholds, unknown threshold | chosen on the dev clip's held-out blocks only | |
| `test_adapt` A, `clf_lomo_<stem>.pt` | **FITTED ON THE TEST DEPLOYMENT**: dev 31 + mask frames of one (A) or all other (leave one recording out) 2025-02-03 recordings | the remaining recordings' mask frames / windows |

Setup: DINOv3 ViT-B/16 (LVD-1689M weights), frozen, last block's normed patch tokens, frame resized to 1280 x 720
(80 x 45 patches of 24 x 24 px of the 1080p frame); linear classifier (standardised features, AdamW, 60 epochs,
class weights 1/sqrt(frequency)); patch label = majority class when it covers >= 75 % of the patch's labelled pixels
and <= 25 % of the patch is unsure (fish: >= 30 %), otherwise ignored. Class map = bilinear upsampling of the patch
probabilities. Scoring: pixel IoU with a 12 px band around region borders ignored (strict IoU also listed),
unsure pixels ignored. The dev clip's 13 small "fish" were not fish (user, 2026-11-20): fish is ignored in the dev
masks (`common.DEV`, `ignore_codes`); no classifier here was ever trained with them (at 1280 px they never reached
patch purity anyway). The "can this route see 15 px fish" experiment was dropped on that correction.

Window = 1 s grid slot (`uwid.eval_presence`); 6 frames classified per window (every 5th frame) plus the labelled
frame. Presence score of a frame = largest 8-connected region of patches with p(whale) > 0.5. Window rules: `kth2` /
`kth3` = the 2nd / 3rd largest frame score (the region must persist in 2 / 3 of the ~7 frames); `meanmap` = largest
region on the window's averaged class map; `max` = largest frame score. On the test recordings each window holds one
labelled frame, so frame and window units are the same 840 moments.

## Plain answer

- Whales carry across deployments: on 14 test mask frames 94 % of whale pixels are called whale (8 / 8 instances
  found, 2 clear, 6 hazy), whale IoU 0.70 (dev held-out 0.71). Presence over 791 test windows (221 with a clear
  whale, 570 with none; 50 faint-only left out): AUC 0.98 for frame and window scores alike.
- False alarms are the cost of moving to a new deployment: the dev clip is a crowded competitive group and too easy to
  set a threshold on (18 whale-free windows, mostly air / bubbles). Its thresholds give 220 / 221 found with 250 / 570
  false alarms per single frame, 187 / 221 with 16 / 570 for `kth3`. On a fixed grid of region sizes the window
  average halves the false alarms at equal recall (>= 32 patches: frame 202 / 221 with 34 / 570, window mean map
  200 / 221 with 18 / 570; >= 64: 176 / 221 with 4 / 570).
- Back: 99.4 % of the tagged whale's back pixels are called back on the new tag position; back is almost never
  called whale (0.1 %) nor whale back (0.1 %). But 8.7 % of open water is called back (sun glow, horizon haze), so
  back IoU is 0.21.
- The new deployment's bright blue open water is called `air` on 30 % of water pixels (53-63 % of the frame in five
  of seven recordings): the dev clip has two air mask frames and darker water. Underwater `air` should be read as
  water until test-like water is in the training set.
- Seafloor: none of the 840 test frames shows seafloor (WP3 looked for it), so seafloor recall cannot be measured
  and the fitted-on-test seafloor experiment could not be run. Mask2Former ADE's sand / rock / earth / floor /
  mountain fire on 68 / 840 test frames (> 1 % of the frame) and 9 / 31 dev frames, all false: on near whales, the
  back's horizon, the sea seen from air. Not usable as a seafloor source without a check.
- Unknown (cosine distance to the nearest training patch above the dev 99th percentile, 0.297): flags drone gear
  (99.9 %) and entry froth (54 %), i.e. truly new things, but also 9 % of whale pixels and 0 % of the water called
  air. It does not catch the domain shift that matters.
- More labels: 4 mask frames of one test recording raise water IoU 0.38 -> 0.71 and back 0.17 -> 0.37 on the other
  six recordings and teach fish (25 / 25 fish found, with 39 extra fish blobs). With 9-13 test frames (leave one
  recording out): water 0.54 -> 0.95, back 0.21 -> 0.72, whale 0.70 -> 0.76, fish pixels found 72 %; presence false
  alarms at >= 8 patches drop from 93 to 28 / 570 per frame and from 69 to 17 / 570 per window mean map, at
  216 / 212 of 221 found. More dev frames do not do this: on the test masks back IoU goes 0.46 / 0.28 / 0.53 with
  5 dev frames to 0.21 with 31.

## How to add labelled frames, deployments or classes

A label folder is `uwid.patchseg.LabelFolder(name, masks_dir, images, ignore_codes=...)` (images = a folder of
`<frame>.jpg` or the video). `common.DEV` and `common.test_folder(stem)` are the two kinds in use. Features are
cached per folder and input size (`labels_<name>_<config>.npz`, features only; targets are rebuilt from the masks at
every load, so repainted masks take effect). Training data = `C.load_folder(...)` for each folder, then
`ps.training_set` or `C.xy_from`; a class gets an output as soon as any folder has patches of it (classes without
patches are never predicted). Class codes are `ps.CLASSES` (gear = 8 was added for the test masks; unknown = 9).

## Scripts (run from the repo root with `.venv/bin/python`, GPU steps under the lock)

`extract_dev.py` (features), `cv_dev.py compare|curve|final`, `unknown_calib.py`, `predict_video.py dev|test`,
`m2f_seafloor.py`, `eval_dev.py`, `eval_test.py [--tag _lomo]`, `test_masks.py features|curve|adapt|lomo_save`,
`cost.py`, `make_sheets.py`, `write_tables.py` (the tables below); `run_gpu.sh`, `run_gpu2.sh`, `run_gpu3.sh` chain
the GPU steps. Results JSON in `results/`. Cache: `underwater_cache/patchseg/` (features, classifiers `clf_*.pt`,
class maps `pred_*.npz`, Mask2Former maps `m2f_*.npz`). Sheets: `review_sheets/underwater/patchseg_*.jpg`.

# Tables (generated by write_tables.py)

## Within the dev clip: input size and head (leave one time block out, single frames)

IoU with a 12 px band around region borders ignored. Recall = share of the whale's pixels called whale; found = instances with >= 25 % of pixels called whale.

| run | frames | water | whale | back | bubbles | surface | air | seafloor | mIoU (6) | whale recall clear (found/n) | hazy | faint | back->whale | whale->back | bubbles->whale | surface->whale | faint->water | water->whale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w960_h0 | 31 | 0.85 | 0.72 | 0.78 | 0.76 | 0.48 | 0.95 | - | 0.76 | 0.90 (24/25) | 0.78 (37/40) | 0.44 (17/20) | 0.002 | 0.021 | 0.008 | 0.010 | 0.484 | 0.034 |
| w1280_h0 | 31 | 0.85 | 0.71 | 0.79 | 0.77 | 0.45 | 0.98 | - | 0.76 | 0.92 (24/25) | 0.79 (39/40) | 0.39 (15/20) | 0.002 | 0.011 | 0.005 | 0.017 | 0.525 | 0.034 |
| w1280_h256 | 31 | 0.86 | 0.71 | 0.75 | 0.78 | 0.45 | 0.98 | - | 0.75 | 0.88 (23/25) | 0.72 (35/40) | 0.41 (16/20) | 0.032 | 0.022 | 0.003 | 0.006 | 0.468 | 0.023 |
| w1920_h0 | 31 | 0.82 | 0.64 | 0.75 | 0.78 | 0.42 | 0.86 | - | 0.71 | 0.91 (24/25) | 0.69 (36/40) | 0.37 (15/20) | 0.005 | 0.012 | 0.012 | 0.032 | 0.536 | 0.036 |
| w1920_h256 | 31 | 0.86 | 0.68 | 0.77 | 0.79 | 0.43 | 0.94 | - | 0.75 | 0.89 (24/25) | 0.69 (36/40) | 0.37 (14/20) | 0.010 | 0.010 | 0.014 | 0.027 | 0.516 | 0.022 |
| w1280_L4_h0 | 31 | 0.85 | 0.71 | 0.78 | 0.78 | 0.45 | 0.95 | - | 0.75 | 0.90 (24/25) | 0.78 (37/40) | 0.36 (14/20) | 0.021 | 0.015 | 0.004 | 0.010 | 0.534 | 0.028 |
| w1920_t2_h0 | 31 | 0.85 | 0.62 | 0.74 | 0.80 | 0.56 | 0.76 | - | 0.72 | 0.86 (23/25) | 0.79 (38/40) | 0.40 (16/20) | 0.003 | 0.017 | 0.016 | 0.024 | 0.507 | 0.040 |
| w1920_t2_h256 | 31 | 0.87 | 0.66 | 0.76 | 0.80 | 0.53 | 0.88 | - | 0.75 | 0.86 (23/25) | 0.75 (37/40) | 0.40 (17/20) | 0.021 | 0.010 | 0.013 | 0.024 | 0.512 | 0.023 |

Strict IoU (no border band), same runs:

| run | water | whale | back | bubbles | surface | air |
|---|---|---|---|---|---|---|
| w960_h0 | 0.81 | 0.64 | 0.71 | 0.74 | 0.46 | 0.95 |
| w1280_h0 | 0.81 | 0.65 | 0.73 | 0.74 | 0.44 | 0.98 |
| w1280_h256 | 0.83 | 0.65 | 0.70 | 0.75 | 0.43 | 0.98 |
| w1920_h0 | 0.79 | 0.60 | 0.71 | 0.76 | 0.41 | 0.86 |
| w1920_h256 | 0.83 | 0.64 | 0.73 | 0.76 | 0.42 | 0.94 |
| w1280_L4_h0 | 0.82 | 0.65 | 0.72 | 0.75 | 0.44 | 0.95 |
| w1920_t2_h0 | 0.82 | 0.59 | 0.70 | 0.77 | 0.54 | 0.75 |
| w1920_t2_h256 | 0.84 | 0.63 | 0.72 | 0.77 | 0.50 | 0.88 |

## Within the dev clip: learning curve (w1280 linear; training = a contiguous block of n mask frames of the training pool, 3 starting points; held-out block as above)

| n frames | mIoU (6) | whale IoU | back IoU | bubbles IoU | surface IoU | whale recall hazy | faint |
|---|---|---|---|---|---|---|---|
| 5 | 0.60 / 0.44 / 0.62 | 0.46 / 0.47 / 0.56 | 0.64 / 0.64 / 0.74 | 0.65 / 0.37 / 0.72 | 0.49 / 0.33 / 0.43 | 0.74 / 0.86 / 0.77 | 0.45 / 0.54 / 0.39 |
| 10 | 0.59 / 0.51 / 0.65 | 0.45 / 0.54 / 0.62 | 0.68 / 0.66 / 0.76 | 0.60 / 0.60 / 0.72 | 0.47 / 0.43 / 0.45 | 0.78 / 0.84 / 0.77 | 0.50 / 0.55 / 0.43 |
| 20 | 0.66 / 0.65 / 0.76 | 0.67 / 0.65 / 0.71 | 0.78 / 0.77 / 0.80 | 0.67 / 0.68 / 0.77 | 0.46 / 0.46 / 0.45 | 0.77 / 0.77 / 0.79 | 0.45 / 0.44 / 0.43 |
| all | 0.76 | 0.71 | 0.79 | 0.77 | 0.45 | 0.79 | 0.39 |

## Within the dev clip: single frame against the 1 s window (held-out fold classifiers)

| run | frames | water | whale | back | bubbles | surface | air | seafloor | mIoU (6) | whale recall clear (found/n) | hazy | faint | back->whale | whale->back | bubbles->whale | surface->whale | faint->water | water->whale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mask frame alone | 31 | 0.85 | 0.71 | 0.79 | 0.77 | 0.45 | 0.98 | - | 0.76 | 0.92 (24/25) | 0.79 (39/40) | 0.39 (15/20) | 0.002 | 0.011 | 0.005 | 0.017 | 0.525 | 0.034 |
| mean of 7 frames (-15..+15) | 31 | 0.83 | 0.61 | 0.78 | 0.74 | 0.48 | 0.99 | - | 0.74 | 0.75 (23/25) | 0.54 (30/40) | 0.21 (7/20) | 0.001 | 0.015 | 0.002 | 0.016 | 0.702 | 0.026 |

Whale presence (score = largest whale region in patches; whale = a clear whale on a labelled frame, free = no clear and no faint whale; faint-only units left out):

| unit / rule | AUC | whale units | whale-free units | faint-only | threshold at FA <= 10 % | found | false alarms |
|---|---|---|---|---|---|---|---|
| frame | 0.993 | 145 | 37 | 9 | 1 | 143/145 | 1/37 |
| win_max | 0.900 | 74 | 18 | 4 | 282 | 12/74 | 1/18 |
| win_kth2 | 0.958 | 74 | 18 | 4 | 66 | 73/74 | 1/18 |
| win_kth3 | 0.993 | 74 | 18 | 4 | 58 | 73/74 | 0/18 |
| win_meanmap | 0.993 | 74 | 18 | 4 | 1 | 73/74 | 0/18 |

## Unknown threshold (dev clip, held-out blocks)

Cosine distance to the nearest training patch: median 0.132, 95th pct 0.245, 99th pct 0.297 (used), 99.5th 0.314; 111600 patches.

| class | median distance | share > p99 |
|---|---|---|
| water | 0.123 | 0.000 |
| whale | 0.214 | 0.067 |
| back | 0.086 | 0.000 |
| bubbles | 0.150 | 0.025 |
| surface | 0.108 | 0.000 |
| air | 0.176 | 0.000 |

## Test recordings pooled (7 recordings, 840 labelled frames; classifier fitted on the 31 dev frames only)

Whale presence, one labelled frame per window (thresholds in patches of 24 x 24 px; 'dev thr' = chosen on the dev clip at false alarms <= 10 %; the sweep is a fixed grid, not tuned):

| unit / rule | AUC | whale | free | at dev thr: found | false alarms | thr | sweep 4: found / FA | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frame | 0.981 | 221 | 570 | 220/221 | 250/570 | 1 | 220 / 141 | 217 / 93 | 214 / 56 | 202 / 34 | 175 / 10 |
| win_max | 0.969 | 221 | 570 | 45/221 | 0/570 | 282 | 221 / 246 | 221 / 185 | 216 / 138 | 207 / 90 | 191 / 33 |
| win_kth2 | 0.978 | 221 | 570 | 185/221 | 18/570 | 66 | 221 / 209 | 219 / 145 | 215 / 94 | 207 / 54 | 186 / 18 |
| win_kth3 | 0.981 | 221 | 570 | 187/221 | 16/570 | 58 | 220 / 171 | 217 / 120 | 213 / 72 | 205 / 34 | 181 / 12 |
| win_meanmap | 0.981 | 221 | 570 | 218/221 | 201/570 | 1 | 217 / 117 | 216 / 69 | 211 / 39 | 200 / 18 | 176 / 4 |

Test mask frames pooled (14 frames):

| run | frames | water | whale | back | bubbles | surface | air | seafloor | mIoU (6) | whale recall clear (found/n) | hazy | faint | back->whale | whale->back | bubbles->whale | surface->whale | faint->water | water->whale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 14 | 0.54 | 0.70 | 0.21 | 0.00 | 0.12 | 0.30 | - | 0.31 | 0.99 (2/2) | 0.93 (6/6) | - | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 0.018 |
| window | 14 | 0.57 | 0.63 | 0.25 | 0.01 | 0.11 | 0.30 | - | 0.31 | 0.99 (2/2) | 0.86 (6/6) | - | 0.003 | 0.001 | 0.000 | 0.000 | 0.000 | 0.025 |
| single_unknown | 14 | 0.54 | 0.67 | 0.22 | 0.00 | 0.12 | 0.31 | - | 0.31 | 0.86 (1/2) | 0.86 (6/6) | - | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 0.018 |

Rows = truth, columns = prediction (share of the truth class's pixels), single frame:

| truth | pixels | water | whale | back | fish | bubbles | surface | air | seafloor | gear | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| water (single) | 20733534 | 0.534 | 0.018 | 0.087 | 0.000 | 0.022 | 0.038 | 0.301 | 0.000 | 0.000 | 0.000 |
| whale (single) | 1052352 | 0.042 | 0.944 | 0.001 | 0.000 | 0.012 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 |
| back (single) | 815627 | 0.005 | 0.001 | 0.994 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fish (single) | 87131 | 0.688 | 0.310 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 |
| bubbles (single) | 2073600 | 0.000 | 0.000 | 0.122 | 0.000 | 0.000 | 0.000 | 0.878 | 0.000 | 0.000 | 0.000 |
| surface (single) | 120956 | 0.015 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.000 | 0.000 | 0.000 | 0.000 |
| air (single) | 3646497 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.999 | 0.000 | 0.000 | 0.000 |
| gear (single) | 500703 | 0.000 | 0.085 | 0.000 | 0.000 | 0.015 | 0.000 | 0.900 | 0.000 | 0.000 | 0.000 |
| water (single_unknown) | 20733534 | 0.534 | 0.018 | 0.087 | 0.000 | 0.022 | 0.038 | 0.301 | 0.000 | 0.000 | 0.000 |
| whale (single_unknown) | 1052352 | 0.042 | 0.864 | 0.001 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.093 |
| back (single_unknown) | 815627 | 0.005 | 0.001 | 0.994 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fish (single_unknown) | 87131 | 0.667 | 0.279 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.053 |
| bubbles (single_unknown) | 2073600 | 0.000 | 0.000 | 0.106 | 0.000 | 0.000 | 0.000 | 0.357 | 0.000 | 0.000 | 0.537 |
| surface (single_unknown) | 120956 | 0.015 | 0.000 | 0.000 | 0.000 | 0.000 | 0.985 | 0.000 | 0.000 | 0.000 | 0.000 |
| air (single_unknown) | 3646497 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.892 | 0.000 | 0.000 | 0.108 |
| gear (single_unknown) | 500703 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.999 |

## Test recordings pooled (7 recordings, 840 labelled frames; FITTED ON THE TEST DEPLOYMENT: leave one recording out, dev 31 + the mask frames of the other 2025-02-03 recordings (9-13 frames))

Whale presence, one labelled frame per window (thresholds in patches of 24 x 24 px; 'dev thr' = chosen on the dev clip at false alarms <= 10 %; the sweep is a fixed grid, not tuned):

| unit / rule | AUC | whale | free | at dev thr: found | false alarms | thr | sweep 4: found / FA | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frame | 0.982 | 221 | 570 | 218/221 | 65/570 | 1 | 218 / 37 | 216 / 28 | 211 / 21 | 196 / 10 | 156 / 7 |
| win_max | 0.981 | 221 | 570 | 39/221 | 4/570 | 282 | 220 / 89 | 218 / 60 | 215 / 40 | 204 / 23 | 172 / 13 |
| win_kth2 | 0.985 | 221 | 570 | 163/221 | 12/570 | 66 | 219 / 56 | 218 / 41 | 213 / 29 | 202 / 16 | 166 / 12 |
| win_kth3 | 0.985 | 221 | 570 | 168/221 | 9/570 | 58 | 217 / 40 | 214 / 30 | 210 / 20 | 199 / 12 | 162 / 8 |
| win_meanmap | 0.985 | 221 | 570 | 218/221 | 41/570 | 1 | 215 / 22 | 212 / 17 | 205 / 14 | 194 / 6 | 159 / 4 |

Test mask frames pooled (14 frames):

| run | frames | water | whale | back | bubbles | surface | air | seafloor | mIoU (6) | whale recall clear (found/n) | hazy | faint | back->whale | whale->back | bubbles->whale | surface->whale | faint->water | water->whale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | 14 | 0.95 | 0.76 | 0.72 | 0.00 | 0.31 | 0.59 | - | 0.56 | 0.99 (2/2) | 0.81 (6/6) | - | 0.002 | 0.001 | 0.000 | 0.000 | 0.000 | 0.007 |
| window | 14 | 0.96 | 0.69 | 0.81 | 0.00 | 0.39 | 0.57 | - | 0.57 | 0.99 (2/2) | 0.67 (4/6) | - | 0.004 | 0.001 | 0.000 | 0.000 | 0.000 | 0.011 |
| single_unknown | 14 | 0.95 | 0.74 | 0.72 | 0.00 | 0.32 | 0.71 | - | 0.57 | 0.86 (1/2) | 0.73 (6/6) | - | 0.002 | 0.001 | 0.000 | 0.000 | 0.000 | 0.007 |

Rows = truth, columns = prediction (share of the truth class's pixels), single frame:

| truth | pixels | water | whale | back | fish | bubbles | surface | air | seafloor | gear | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| water (single) | 20733534 | 0.941 | 0.007 | 0.015 | 0.004 | 0.021 | 0.010 | 0.002 | 0.000 | 0.000 | 0.000 |
| whale (single) | 1052352 | 0.138 | 0.858 | 0.001 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| back (single) | 815627 | 0.029 | 0.002 | 0.969 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fish (single) | 87131 | 0.245 | 0.037 | 0.000 | 0.718 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bubbles (single) | 2073600 | 0.000 | 0.000 | 0.034 | 0.000 | 0.000 | 0.000 | 0.966 | 0.000 | 0.000 | 0.000 |
| surface (single) | 120956 | 0.107 | 0.000 | 0.000 | 0.000 | 0.000 | 0.893 | 0.000 | 0.000 | 0.000 | 0.000 |
| air (single) | 3646497 | 0.000 | 0.001 | 0.000 | 0.000 | 0.014 | 0.001 | 0.984 | 0.000 | 0.000 | 0.000 |
| gear (single) | 500703 | 0.000 | 0.143 | 0.000 | 0.000 | 0.019 | 0.018 | 0.820 | 0.000 | 0.000 | 0.000 |
| water (single_unknown) | 20733534 | 0.941 | 0.007 | 0.015 | 0.004 | 0.021 | 0.010 | 0.002 | 0.000 | 0.000 | 0.000 |
| whale (single_unknown) | 1052352 | 0.137 | 0.768 | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.093 |
| back (single_unknown) | 815627 | 0.029 | 0.002 | 0.969 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fish (single_unknown) | 87131 | 0.245 | 0.037 | 0.000 | 0.665 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.053 |
| bubbles (single_unknown) | 2073600 | 0.000 | 0.000 | 0.034 | 0.000 | 0.000 | 0.000 | 0.429 | 0.000 | 0.000 | 0.537 |
| surface (single_unknown) | 120956 | 0.107 | 0.000 | 0.000 | 0.000 | 0.000 | 0.893 | 0.000 | 0.000 | 0.000 | 0.000 |
| air (single_unknown) | 3646497 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.000 | 0.878 | 0.000 | 0.000 | 0.108 |
| gear (single_unknown) | 500703 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.999 |

## Across deployments: 2025-02-03 recordings (classifier fitted on all 31 dev frames, thresholds from the dev clip)

| recording | unit / rule | AUC | whale | free | faint-only | found at dev threshold | false alarms at dev threshold | threshold |
|---|---|---|---|---|---|---|---|---|
| 00004 | frame | 0.971 | 26 | 88 | 6 | 26/26 | 40/88 | 1 |
| 00004 | win_max | 0.948 | 26 | 88 | 6 | 8/26 | 0/88 | 282 |
| 00004 | win_kth2 | 0.962 | 26 | 88 | 6 | 23/26 | 6/88 | 66 |
| 00004 | win_kth3 | 0.964 | 26 | 88 | 6 | 23/26 | 6/88 | 58 |
| 00004 | win_meanmap | 0.976 | 26 | 88 | 6 | 26/26 | 41/88 | 1 |
| 00005 | frame | 0.976 | 27 | 87 | 6 | 27/27 | 59/87 | 1 |
| 00005 | win_max | 0.971 | 27 | 87 | 6 | 8/27 | 0/87 | 282 |
| 00005 | win_kth2 | 0.977 | 27 | 87 | 6 | 26/27 | 8/87 | 66 |
| 00005 | win_kth3 | 0.980 | 27 | 87 | 6 | 25/27 | 7/87 | 58 |
| 00005 | win_meanmap | 0.996 | 27 | 87 | 6 | 27/27 | 51/87 | 1 |
| 00006 | frame | 0.993 | 64 | 42 | 14 | 64/64 | 18/42 | 1 |
| 00006 | win_max | 0.975 | 64 | 42 | 14 | 14/64 | 0/42 | 282 |
| 00006 | win_kth2 | 0.980 | 64 | 42 | 14 | 54/64 | 1/42 | 66 |
| 00006 | win_kth3 | 0.994 | 64 | 42 | 14 | 54/64 | 0/42 | 58 |
| 00006 | win_meanmap | 0.983 | 64 | 42 | 14 | 63/64 | 13/42 | 1 |
| 00007 | frame | 1.000 | 36 | 81 | 3 | 36/36 | 12/81 | 1 |
| 00007 | win_max | 1.000 | 36 | 81 | 3 | 9/36 | 0/81 | 282 |
| 00007 | win_kth2 | 1.000 | 36 | 81 | 3 | 32/36 | 0/81 | 66 |
| 00007 | win_kth3 | 1.000 | 36 | 81 | 3 | 33/36 | 0/81 | 58 |
| 00007 | win_meanmap | 1.000 | 36 | 81 | 3 | 36/36 | 8/81 | 1 |
| 00008 | frame | 0.996 | 22 | 91 | 7 | 22/22 | 18/91 | 1 |
| 00008 | win_max | 0.990 | 22 | 91 | 7 | 2/22 | 0/91 | 282 |
| 00008 | win_kth2 | 0.996 | 22 | 91 | 7 | 17/22 | 0/91 | 66 |
| 00008 | win_kth3 | 0.996 | 22 | 91 | 7 | 18/22 | 0/91 | 58 |
| 00008 | win_meanmap | 0.997 | 22 | 91 | 7 | 22/22 | 12/91 | 1 |
| 00009 | frame | 0.956 | 33 | 81 | 6 | 32/33 | 37/81 | 1 |
| 00009 | win_max | 0.943 | 33 | 81 | 6 | 2/33 | 0/81 | 282 |
| 00009 | win_kth2 | 0.959 | 33 | 81 | 6 | 22/33 | 1/81 | 66 |
| 00009 | win_kth3 | 0.963 | 33 | 81 | 6 | 24/33 | 2/81 | 58 |
| 00009 | win_meanmap | 0.950 | 33 | 81 | 6 | 31/33 | 29/81 | 1 |
| 00010 | frame | 0.990 | 13 | 100 | 7 | 13/13 | 66/100 | 1 |
| 00010 | win_max | 0.972 | 13 | 100 | 7 | 2/13 | 0/100 | 282 |
| 00010 | win_kth2 | 0.988 | 13 | 100 | 7 | 11/13 | 2/100 | 66 |
| 00010 | win_kth3 | 0.990 | 13 | 100 | 7 | 10/13 | 1/100 | 58 |
| 00010 | win_meanmap | 0.989 | 13 | 100 | 7 | 13/13 | 47/100 | 1 |

Back, fish, mean class shares on underwater labelled frames:

| recording | frames with back | mean back share (with / without back) | AUC back share | fish frames with a fish patch | no-fish frames with a fish patch | mean shares water / whale / back / bubbles / surface / air |
|---|---|---|---|---|---|---|
| 00004 | 112 | 0.105 / - | - | 0/13 | 0/107 | 0.282 / 0.027 / 0.105 / 0.021 / 0.031 / 0.534 |
| 00005 | 115 | 0.103 / 0.000 | 1.000 | 0/4 | 0/116 | 0.177 / 0.033 / 0.102 / 0.038 / 0.030 / 0.621 |
| 00006 | 117 | 0.107 / 0.039 | 0.983 | 0/25 | 0/95 | 0.512 / 0.040 / 0.106 / 0.023 / 0.075 / 0.245 |
| 00007 | 119 | 0.098 / 0.102 | 0.256 | 0/0 | 0/120 | 0.218 / 0.021 / 0.098 / 0.020 / 0.045 / 0.598 |
| 00008 | 117 | 0.084 / - | - | 0/1 | 0/119 | 0.186 / 0.011 / 0.084 / 0.033 / 0.058 / 0.629 |
| 00009 | 118 | 0.102 / 0.124 | 0.161 | 0/2 | 0/118 | 0.184 / 0.023 / 0.102 / 0.037 / 0.047 / 0.606 |
| 00010 | 118 | 0.103 / 0.060 | 0.966 | 0/0 | 0/120 | 0.082 / 0.030 / 0.102 / 0.052 / 0.142 / 0.593 |

What the classifier calls the pixels Mask2Former calls sand / rock / earth / floor / mountain:

| recording | frame label seafloor | frames | frames with M2F seafloor | water | whale | back | bubbles | surface | air | unknown |
|---|---|---|---|---|---|---|---|---|---|---|
| 00004 | none | 120 | 21 | 0.013 | 0.248 | 0.124 | 0.005 | 0.000 | 0.611 | 0.153 |
| 00005 | none | 120 | 14 | 0.007 | 0.038 | 0.084 | 0.068 | 0.000 | 0.803 | 0.207 |
| 00006 | none | 120 | 27 | 0.007 | 0.525 | 0.259 | 0.023 | 0.002 | 0.183 | 0.423 |
| 00007 | none | 120 | 14 | 0.017 | 0.532 | 0.444 | 0.000 | 0.000 | 0.007 | 0.000 |
| 00008 | none | 120 | 16 | 0.004 | 0.104 | 0.115 | 0.002 | 0.000 | 0.775 | 0.128 |
| 00009 | none | 120 | 14 | 0.005 | 0.259 | 0.590 | 0.000 | 0.000 | 0.146 | 0.134 |
| 00010 | none | 120 | 12 | 0.000 | 0.181 | 0.582 | 0.002 | 0.049 | 0.186 | 0.034 |

Test mask frames (WP3 AI by-eye masks):

| run | frames | water | whale | back | bubbles | surface | air | seafloor | mIoU (6) | whale recall clear (found/n) | hazy | faint | back->whale | whale->back | bubbles->whale | surface->whale | faint->water | water->whale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00004 single | 4 | 0.89 | 0.83 | 0.57 | - | - | 0.81 | - | 0.78 | 1.00 (1/1) | 0.91 (3/3) | - | 0.000 | 0.000 | - | - | 0.000 | 0.020 |
| 00004 window mean | 4 | 0.87 | 0.84 | 0.56 | - | - | 0.80 | - | 0.77 | 1.00 (1/1) | 0.84 (3/3) | - | 0.000 | 0.001 | - | - | 0.000 | 0.023 |
| 00004 single + unknown | 4 | 0.89 | 0.86 | 0.57 | - | - | 0.80 | - | 0.78 | 0.92 (1/1) | 0.91 (3/3) | - | 0.000 | 0.000 | - | - | 0.000 | 0.020 |
| 00005 single | 2 | 0.79 | 0.63 | 0.33 | - | 0.30 | - | - | 0.51 | - | 0.99 (1/1) | - | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.030 |
| 00005 window mean | 2 | 0.79 | 0.50 | 0.33 | - | 0.30 | - | - | 0.48 | - | 0.90 (1/1) | - | 0.000 | 0.001 | - | 0.000 | 0.000 | 0.040 |
| 00005 single + unknown | 2 | 0.79 | 0.63 | 0.33 | - | 0.30 | - | - | 0.51 | - | 0.99 (1/1) | - | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.030 |
| 00006 single | 2 | 0.20 | 0.74 | 0.17 | - | - | - | - | 0.37 | - | 0.87 (2/2) | - | 0.004 | 0.000 | - | - | 0.000 | 0.010 |
| 00006 window mean | 2 | 0.31 | 0.53 | 0.20 | - | - | - | - | 0.35 | - | 0.92 (2/2) | - | 0.010 | 0.000 | - | - | 0.000 | 0.025 |
| 00006 single + unknown | 2 | 0.20 | 0.30 | 0.17 | - | - | - | - | 0.22 | - | 0.47 (2/2) | - | 0.004 | 0.000 | - | - | 0.000 | 0.010 |
| 00007 single | 1 | 0.00 | - | 0.09 | - | - | - | - | 0.05 | - | - | - | 0.000 | - | - | - | 0.000 | 0.000 |
| 00007 window mean | 1 | 0.00 | - | 0.08 | - | - | - | - | 0.04 | - | - | - | 0.000 | - | - | - | 0.000 | 0.000 |
| 00007 single + unknown | 1 | 0.00 | - | 0.09 | - | - | - | - | 0.05 | - | - | - | 0.000 | - | - | - | 0.000 | 0.000 |
| 00008 single | 2 | 0.56 | 1.00 | 0.31 | - | - | - | - | 0.62 | 0.96 (1/1) | - | - | 0.000 | 0.028 | - | - | 0.000 | 0.001 |
| 00008 window mean | 2 | 0.58 | 0.53 | 0.29 | - | - | - | - | 0.47 | 0.95 (1/1) | - | - | 0.000 | 0.042 | - | - | 0.000 | 0.003 |
| 00008 single + unknown | 2 | 0.56 | 0.00 | 0.31 | - | - | - | - | 0.29 | 0.04 (0/1) | - | - | 0.000 | 0.028 | - | - | 0.000 | 0.001 |
| 00009 single | 2 | 0.29 | - | 0.10 | 0.00 | - | - | - | 0.13 | - | - | - | 0.000 | - | 0.000 | - | 0.000 | 0.060 |
| 00009 window mean | 2 | 0.38 | - | 0.18 | 0.01 | - | - | - | 0.19 | - | - | - | 0.000 | - | 0.000 | - | 0.000 | 0.071 |
| 00009 single + unknown | 2 | 0.29 | - | 0.10 | 0.00 | - | - | - | 0.13 | - | - | - | 0.000 | - | 0.000 | - | 0.000 | 0.060 |
| 00010 single | 1 | - | - | - | - | - | 1.00 | - | 1.00 | - | - | - | - | - | - | - | 0.000 | - |
| 00010 window mean | 1 | - | - | - | - | - | 0.99 | - | 0.99 | - | - | - | - | - | - | - | 0.000 | - |
| 00010 single + unknown | 1 | - | - | - | - | - | 0.99 | - | 0.99 | - | - | - | - | - | - | - | 0.000 | - |

Seafloor pixels in the test masks, predicted as (single frame):

| recording | seafloor px | water | whale | back | fish | bubbles | surface | air | seafloor | unknown |
|---|---|---|---|---|---|---|---|---|---|---|

## Learning curve on the test masks (dev frames only; contiguous block of n dev mask frames)

| run | frames | water | whale | back | bubbles | surface | air | seafloor | mIoU (6) | whale recall clear (found/n) | hazy | faint | back->whale | whale->back | bubbles->whale | surface->whale | faint->water | water->whale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 dev frames (start 0) | 14 | 0.83 | 0.65 | 0.45 | 0.00 | 0.04 | 0.51 | - | 0.42 | 0.96 (2/2) | 0.95 (6/6) | - | 0.000 | 0.010 | 0.000 | 0.159 | 0.000 | 0.014 |
| 5 dev frames (start 1) | 14 | 0.80 | 0.20 | 0.28 | 0.00 | 0.03 | 0.00 | - | 0.22 | 1.00 (2/2) | 0.99 (6/6) | - | 0.023 | 0.000 | 0.780 | 0.000 | 0.000 | 0.028 |
| 5 dev frames (start 2) | 14 | 0.80 | 0.43 | 0.53 | 0.00 | 0.03 | 0.60 | - | 0.40 | 0.94 (1/2) | 0.91 (6/6) | - | 0.113 | 0.022 | 0.000 | 0.000 | 0.000 | 0.033 |
| 10 dev frames (start 0) | 14 | 0.81 | 0.68 | 0.48 | 0.00 | 0.04 | 0.45 | - | 0.41 | 0.99 (2/2) | 0.94 (6/6) | - | 0.001 | 0.002 | 0.000 | 0.354 | 0.000 | 0.013 |
| 10 dev frames (start 1) | 14 | 0.67 | 0.15 | 0.27 | 0.00 | 0.02 | 0.00 | - | 0.19 | 1.00 (2/2) | 1.00 (6/6) | - | 0.066 | 0.000 | 0.825 | 0.000 | 0.000 | 0.054 |
| 10 dev frames (start 2) | 14 | 0.79 | 0.74 | 0.37 | 0.00 | 0.04 | 0.55 | - | 0.41 | 0.98 (2/2) | 0.88 (6/6) | - | 0.021 | 0.002 | 0.000 | 0.000 | 0.000 | 0.012 |
| 20 dev frames (start 0) | 14 | 0.61 | 0.63 | 0.32 | 0.00 | 0.09 | 0.32 | - | 0.33 | 0.99 (2/2) | 0.99 (6/6) | - | 0.002 | 0.000 | 0.000 | 0.053 | 0.000 | 0.023 |
| 20 dev frames (start 1) | 14 | 0.54 | 0.44 | 0.20 | 0.25 | 0.02 | 0.00 | - | 0.24 | 1.00 (2/2) | 0.83 (5/6) | - | 0.026 | 0.002 | 0.000 | 0.000 | 0.000 | 0.043 |
| 20 dev frames (start 2) | 14 | 0.41 | 0.76 | 0.53 | 0.03 | 0.05 | 0.24 | - | 0.34 | 1.00 (2/2) | 0.92 (6/6) | - | 0.023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 |
| 31 dev frames (start 0) | 14 | 0.51 | 0.72 | 0.21 | 0.00 | 0.10 | 0.28 | - | 0.30 | 0.99 (2/2) | 0.93 (6/6) | - | 0.002 | 0.001 | 0.000 | 0.083 | 0.000 | 0.017 |

## FITTED ON THE TEST DEPLOYMENT: dev 31 + k mask frames of one recording, scored on the other recordings' mask frames

| run | frames | water | whale | back | bubbles | surface | air | seafloor | mIoU (6) | whale recall clear (found/n) | hazy | faint | back->whale | whale->back | bubbles->whale | surface->whale | faint->water | water->whale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| +0 of 00004 -> 00005,00006,00007,00008,00009,00010 | 10 | 0.38 | 0.51 | 0.17 | 0.00 | 0.18 | 0.19 | - | 0.24 | 0.93 (1/1) | 0.95 (3/3) | - | 0.003 | 0.002 | 0.000 | 0.083 | 0.000 | 0.016 |
| +1 of 00004 -> 00005,00006,00007,00008,00009,00010 | 10 | 0.34 | 0.60 | 0.21 | 0.00 | 0.27 | 0.21 | - | 0.27 | 0.94 (1/1) | 0.95 (3/3) | - | 0.003 | 0.002 | 0.000 | 0.099 | 0.000 | 0.011 |
| +2 of 00004 -> 00005,00006,00007,00008,00009,00010 | 10 | 0.47 | 0.65 | 0.30 | 0.00 | 0.31 | 0.23 | - | 0.33 | 0.93 (1/1) | 0.95 (3/3) | - | 0.002 | 0.002 | 0.000 | 0.110 | 0.000 | 0.009 |
| +4 of 00004 -> 00005,00006,00007,00008,00009,00010 | 10 | 0.71 | 0.67 | 0.37 | 0.00 | 0.29 | 0.39 | - | 0.41 | 0.91 (1/1) | 0.98 (3/3) | - | 0.013 | 0.003 | 0.000 | 0.026 | 0.000 | 0.010 |

## Cost (first 120 s of recording 00005; RTX 4000 Ada shared)

| input | stride | frames decoded | frames classified | wall s | decode s | GPU part s | video s | GPU h per video h | peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|
| m4v | 5 | 3591 | 719 | 43.3 | 7.8 | 35.4 | 120.0 | 0.361 | 1.07 |
| m4v | 1 | 3594 | 3594 | 165.4 | 6.3 | 159.1 | 120.0 | 1.378 | 1.08 |
| mov | 5 | 3591 | 719 | 33.9 | 2.0 | 31.9 | 120.0 | 0.282 | 1.08 |
| mov | 1 | 3594 | 3594 | 165.3 | 6.2 | 159.2 | 120.0 | 1.378 | 1.08 |

