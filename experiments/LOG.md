# Experiment log

Newest entry first. One entry per finished piece of work:

    ## YYYY-MM-DD HH:MM  <experiment folder>: <one-line result>
    What was run, on which data (videos, number of frames), the numbers, the verdict, the caveats, output paths.

Rules of evidence are in `CLAUDE.md` (split by video, AI by-eye labels are named as such, one run is one run).

## 2026-11-20 23:45  02_reference_labels (WP3b part A): size and quality of the largest other whale on all 366 whale frames: on the test deployment 118/221 whale frames have the whale at <= 0.2 of the frame width, 11/221 are `hq`
AI by-eye labels by Claude (first proposal; the user's eye defines "high quality"). Added to every `frames.csv`
(existing columns unchanged): `whale_size` (longest extent of the largest other whale / frame width, 0.1 steps),
`whale_frame_share` (rough, 0.05 steps), `whale_cut`, `quality` (hq / ok / low), `quality_note`; empty on frames
without a clear other whale. Graded on 640 px sheets with 0.1 tick marks (`quality.py view`).
- Dev clip (145 frames): hq 15, ok 94, low 36; sizes 0.2-0.7, mostly 0.3-0.4.
- Test 2025-02-03 pieces 00004-00010 (221 frames): hq 11, ok 82, low 128; size 0.1: 40, 0.2: 78, 0.3: 49, 0.4: 24,
  0.5: 17, 0.6: 9, 0.7: 2, 0.9: 2; cut by the edge 127. Per piece in `02_reference_labels/tables.md`.
- Rule: hq = size >= 0.5, sharp, surface detail visible, lit (not a dark silhouette), a whole part in view, little
  bubble cover; ok = clear whale >= 0.3 failing an hq condition; low = <= 0.2, or hazy / sliver / in bubbles.
- Size check against 31 mask frames: mean |eye - mask| 0.046, 26/31 within 0.1. Area share runs high vs masks
  (coarse only).
- Main open point for the user: large backlit silhouettes from below (27 frames `ok` at size >= 0.5, 20 marked
  `near hq`). If those count as hq, hq roughly doubles. 26 close calls marked.
Outputs: `experiments/02_reference_labels/labels/*/frames.csv`, `quality.py`, README section "Size and quality",
sheets `/home/sat3737/NOAA_whales/review_sheets/underwater/reference_quality_examples_01..10.jpg` (39 frames).

## 2026-11-20 23:00  04b_patch_segmentation (WP5b): linear probe on DINOv3 patches from 31 dev frames: whales carry to the new deployment (94 % of whale pixels, presence AUC 0.98), water and back do not (water called air 30 %, back IoU 0.21); 10-13 frames of the new deployment fix most of it
Frozen DINOv3 ViT-B/16 at 1280 x 720 (patches of 24 px), linear classifier per patch (`src/uwid/patchseg.py`). Truth:
WP3 AI by-eye labels. Fitted on the 31 dev mask frames of `2025-12-20 CG.m4v` only, unless marked; input size,
head, presence and unknown thresholds chosen on the dev clip's held-out time blocks. The dev "fish" are ignored
(user correction: not fish); no classifier was trained with them; the 15 px fish experiment was dropped.
- Dev clip, leave one of 4 time blocks out (optimistic: one clip): IoU water 0.85, whale 0.71, back 0.79, bubbles
  0.77, surface 0.45 (coarse polygons), air 0.98 (2 air frames). Whale pixel recall clear 0.92 (24/25 found), hazy
  0.79 (39/40), faint 0.39 (15/20; 52 % of faint pixels called water). Back->whale 0.2 %, whale->back 1.1 %,
  bubbles->whale 0.5 %, surface->whale 1.7 %. 960 and 1280 px equal, 1920 px and 2x tiles worse, two-layer head and
  4-layer features no gain. Learning curve (contiguous blocks): mIoU 0.44-0.62 with 5 frames, 0.51-0.65 with 10,
  0.65-0.76 with 20, 0.76 with all ~23 of the pool.
- Test, 7 recordings 2025-02-03 00004-00010 (840 labelled frames; 221 windows with a clear whale, 570 without, 50
  faint-only left out; one labelled frame per window): presence AUC 0.98 single frame and window alike. At the dev
  thresholds: single frame 220/221 found with 250/570 false alarms (the dev clip is too crowded to set a threshold:
  18 whale-free windows); window `kth3` (region in 3 of ~7 frames) 187/221 with 16/570. Fixed grid, not tuned:
  region >= 32 patches frame 202/221 with 34/570, window mean map 200/221 with 18/570; >= 64: 176/221 with 4/570.
  WP5a (OWLv2 + window rule) is better at the same false alarms (192/221 at 7/570).
- Test masks (14 frames, 9 whales): whale IoU 0.70, 94 % of whale pixels found (8/8 instances), back pixels called
  back 99.4 % (new tag position), but 8.7 % of water called back (back IoU 0.21) and 30 % of water called air
  (bright blue water; water IoU 0.54); fish pixels -> water 69 % / whale 31 % (no fish class); entry froth -> air;
  drone gear -> air 90 %. Window averaging of class maps lowers mask IoU of moving things (dev whale 0.71 -> 0.61,
  faint recall 0.39 -> 0.21; test whale 0.70 -> 0.63): average for presence, not for masks.
- Seafloor: none in any test frame (WP3), so not measurable; the fitted-on-test seafloor run could not be done.
  Mask2Former ADE sand / rock / earth / floor / mountain fire on 68/840 test frames and 9/31 dev frames, all false
  (near whales, the back's horizon, sea seen from air). "Unknown" (nearest-training-patch distance > dev 99th pct)
  flags drone gear 99.9 % and entry froth 54 % but also 9 % of whale pixels and none of the water called air.
- FITTED ON THE TEST DEPLOYMENT: +4 mask frames of 00004 -> other recordings: water IoU 0.38 -> 0.71, back 0.17 ->
  0.37, fish 25/25 found (+39 false fish blobs). Leave one recording out (dev + 9-13 frames of the other
  recordings): water 0.54 -> 0.95, back 0.21 -> 0.72, whale 0.70 -> 0.76, 72 % of fish pixels found; presence false
  alarms at >= 8 patches 93 -> 28/570 (frame) and 69 -> 17/570 (window mean map) at 216/212 of 221 found. More dev
  frames do not do this (test back IoU 0.46 / 0.28 / 0.53 with 5 dev frames, 0.21 with 31).
- Cost (shared RTX 4000 Ada, first 120 s of 00005): 6 frames per 1 s window: 0.28 GPU-h per video-h from `.mov`,
  0.36 from `.m4v` (decode in order + DINOv3 + classifier + unknown distance), 21 classified frames/s, 1.1 GB GPU;
  every frame: 1.38 GPU-h per video-h.
Caveats: 14 test mask frames (2 clear whales); one test day, one tag; test JPEGs (q92) used for the mask-frame
features, decoded frames for windows. Outputs: `experiments/04b_patch_segmentation/` (scripts, `tables.md`,
`results/*.json`), cache `underwater_cache/patchseg/` (features, `clf_w1280_h0_{all,fold*}.pt`, `clf_lomo_*.pt`,
`pred_*.npz`, `m2f_*.npz`, `test_frames_*.csv`), sheets `review_sheets/underwater/patchseg_*.jpg`.

## 2026-11-20 22:20  04_whale_fish_detection (WP5a, update): all seven test pieces scored: 192/221 whale windows found, 7/570 false; SAM 3 text prompts weaker on the test deployment
The agent's queued runs (OWLv2 on pieces 00009-00010, SAM 3 on 00006-00010) finished after its report; the
orchestrator re-ran `score_test.py --sheets --fish-sheets` with no change of settings. Same truth as the entry below
(AI by-eye labels, one labelled frame per window), now 2025-02-03 pieces 00004-00010 (840 labelled frames).
- OWLv2 "whale", both frames of the window >= 0.25: 192/221 whale windows found (86.9 %), 7/570 whale-free windows
  fired (1.2 %); single frame 203/221 and 14/570; window AUC 0.987. Weakest piece 00009: 23/33 found, 1/81 false.
  Curve: 80.5 % found at 5/570 false, 97.3 % at 28/570.
- OWLv2 fish, labelled frames: 29/45 fish frames found, 14/795 fired, AUC 0.963; 60 of 106 fish boxes found at 0.5,
  99 at 0.3; 1,910 of 7,160 confident fish boxes sit on a whale box.
- SAM 3, text prompts, all seven pieces: whale frame AUC 0.921 (at 0.5: 167/221 found, 55/570 fired; OWLv2 single
  frame: 203/221, 14/570); fish at 0.5: 45/45 found but 322/795 fired; "seafloor" >= 0.5 on 617 of 821 underwater
  frames, where no seafloor exists (it segments the back).
- Output: `experiments/04_whale_fish_detection/test_results.json`, sheets `detect_test_0000{4..10}_*.jpg`.

## 2026-11-20 22:05  04_whale_fish_detection (WP5a): OWLv2 "whale" + 2-frame window rule finds 157/175 whale windows at 6/389 false on the new deployment; fish found (AUC 0.96) but fish and whales get swapped; back = SAM 2.1 bottom box; seafloor untestable (none in any label)
Text-prompted models, no training. Truth: AI by-eye labels (WP3). Everything chosen on the dev clip `2025-12-20 CG.m4v`
(191 labelled frames = 74 whale / 18 whale-free 1 s windows; 31 mask frames, 85 whale instances); then applied once,
unchanged, to the 2025-02-03 test pieces 00004-00008 (`.m4v`, 148 min; 600 labelled frames = windows: 175 whale,
389 whale-free, 36 faint-only; 11 mask frames). One run each. Full tables: `experiments/04_whale_fish_detection/tables.md`.
- Operating point (dev): OWLv2-base, prompt "whale", boxes in the clock or > 60 % of the frame dropped, 2 frames per
  window (0.5 s apart), window = whale if both frames have a box >= 0.25. Dev: 72/74 found, 0/18 false (single frames
  at 0.25: 144/145, 8/37 false). **Test: 157/175 (89.7 %) found, 6/389 (1.5 %) false, 3/374 underwater** (single
  frames: 165/175, 13/389). Window AUC 0.995 dev, 0.994 test. Test misses: whales cut by the edge or the surface, small far
  whales, a fluke behind the back's hump, and very near whales filling the frame (scored 0.72-0.89 but removed by the
  dev-chosen whole-frame filter; post-hoc without it 159/175, same false alarms, dev false 0 -> 3/18).
- Window step: dev, same threshold, single frame 8/37 false -> window 0/18 for 2 lost whale windows; any-frame max is
  worse (6/18). Test: false alarms halved (13 -> 6) for 8 lost whale windows (00006 alone got slightly worse).
- False alarms: dev = bubble clouds / froth / streaks (14 of 19 boxes by eye; 4 more are real whales the 2 labelled
  frames miss); test = the drone's own parts in air (3) and fish (3 underwater). Clock: 0 whale or fish boxes >= 0.1.
  The back: rarely a "whale" (dev: ~1-4 of 22 bottom-zone boxes; the other ~17 are real whales swimming low, so a
  "drop boxes at the bottom" rule would cost whales; the "does not move" rule flagged none).
- Prompt wording (OWLv2): ranking barely moves (AUC 0.96-1.00 for whale / humpback whale / dolphin / shark / ...);
  the score scale does, so a threshold belongs to one prompt. LLMDet is more wording-sensitive.
- Back as a class: text prompts fail ("whale back" boxes OTHER whales); SAM 2.1 with a fixed bottom-25 % box: median
  IoU 0.81 on dev (25/27 >= 0.5), 0.69 on test (10/11 >= 0.5). SAM 3 "seafloor" also segments the back (75 % of it).
- Seafloor: none in any dev or test label (WP3 looked on purpose), so only false alarms: Mask2Former seafloor-like
  > 5 % of the frame on 19 (ADE) / 29 (COCO) of 191 dev frames and 8 / 21 of 346 test frames, on the back / near
  whales; SAM 3 "seafloor" >= 0.5 on 112/191 dev and 181/228 test frames (the back). A real test needs footage with
  a bottom. Mask2Former calls whales "water" / "person" and never "whale": no whale detector.
- Fish (dev has none: the 13 dev "fish" were withdrawn by the user): dev false alarms per minute ("fish" both frames
  >= 0.5) ~1.9, on bubble streaks; threshold 0.5 set from that only. Test, labelled frames 43 fish / 557 none: AUC
  0.963, 28/43 found, 12/557 false at 0.5; WP3's 103 fish boxes 59 found at 0.5, 97 at 0.3. Swaps: ~1 in 4 confident
  fish boxes sits on a whale box (tail stocks, flukes, the surface as "school of fish"); 14 of 33 "whale" boxes on the
  test mask frames are on single jacks at close range; all 3 underwater whale false alarms on test are fish. One
  deployment with fish (mostly 00004 / 00006): anecdotal for the rate, not for the confusion.
- Other models (dev): LLMDet-large boxes are broken in transformers 5.20.1 (heights ~0.45x on the docs cats image;
  weights load clean); presence no better than OWLv2 and 4x slower; dropped. SAM 3 (access arrived during the run;
  runs on torch 2.6 via transformers): whale presence like OWLv2 on dev ("humpback whale" frame AUC 0.996), tighter
  masks (median IoU 0.89 vs 0.83 OWLv2 + SAM 2.1), but on test weaker (frame AUC 0.961, 2 pieces) and "fish" lands on
  whales (47 % of dev whale pixels). YOLOE-26l (AGPL-3.0, installed in `.venv`): 30 frames/s, weaker (frame AUC 0.94),
  fish on 348/572 fishless dev frames.
- Cost (RTX 4000 Ada): OWLv2 7.4-7.9 frames/s, 0.7-1.0 GB; 0.25-0.36 GPU h per video h at 2 frames per window
  (measured end to end 0.36 incl. decoding + the labelled frames). SAM 2.1 3.7 frames/s (masks only); SAM 3 1.8 s per
  frame for 8 prompts, 4.2 GB; LLMDet 2.0 frames/s, 4.0 GB; Mask2Former 9-14 frames/s, 2 GB; YOLOE 30 frames/s.
- Not done / caveats: test pieces 00009-00010 and SAM 3 on 00006-00010 are queued detached (`jobs/owl_test2.sh`,
  `jobs/sam3_test.sh`), then `score_test.py` includes them. Test window truth is one labelled frame per window. The
  dev clip is a large competitive group (many whales per frame); test windows hold one whale (the calf) 170/175 times.
  My first LLMDet and SAM 3 runs took 9.2 / 15 GB of GPU memory (stopped at once; all runners now cap at 8.5 GB).
- Code: `src/uwid/detect.py` (OWLv2, LLMDet, SAM 3, YOLOE, SAM 2.1, Mask2Former runners; common box format;
  resumable cache), `src/uwid/eval_presence.py` (window truth + scorers, for WP5b too; `tests/test_eval_presence.py`),
  scripts in `experiments/04_whale_fish_detection/`. Cache `underwater_cache/detect/` (1.2 GB incl. YOLOE weights).
  Sheets `review_sheets/underwater/detect_dev_{false_alarms,misses,bottom_zone,successes}.jpg`, `detect_fish_dev.jpg`,
  `detect_test_<piece>_{false_alarms,misses}.jpg`, `detect_test_maskframes.jpg`, `detect_fish_test_<piece>.jpg`.

## 2026-11-20 21:10  03_water_state (WP4): picture alone: near perfect on unseen pieces of the same tag; on a new deployment DINOv3 + 1 s window step gets 94 % of windows right (1112/1182) and 30 (0.5 s) to 38 (1 s) of 40 events, with false events
Per 1 s window (user's unit), frames at 5/s. Fitted on 2025-02-03 `.mov` pieces 00004-00006 (sensor truth; C by
leave-one-piece-out, window step tuned on those LOPO scores), scored on 00007-00009 (same tag, 5,342 s, 54 events),
the dev clip `2025-12-20 CG.m4v` (95 s, AI by-eye labels, 4 events) and a public CATS deployment (Zenodo 4902325,
videos 00011 + 00013, Antarctic, 1,184 s, depth truth, 40 events; 1.2 GB fetched, time base corrected by eye on 6
events: lens out 1.15 s / back under 0.65 s after the depth crossing, +-0.15 s). Nothing fitted on any scored video.
- Same tag, unseen pieces (2-state chain): colour numbers, CLIP probe and DINOv3 all get 5228-5231/5232 underwater,
  53-54/55 air, 48-51/54 crossing windows; events 52-53/54 within 0.5 s, 0-1 false in 1.5 h; per frame air recall
  100 %, precision 98-100 %. CLIP zero-shot: 31/54 events, 61 false (calls whales against blue water air).
- New deployment, Zenodo (DINOv3, v2 chain): underwater windows 350/356, air 731/786 (precision 731/734), crossing
  31/40; events 30/40 within 0.5 s, 38/40 within 1 s, 58 false (176/h); with the stronger v1 chain 21/40 events,
  5 false, air 785/786. CLIP probe 26/40 events (213/356 underwater windows), colour 1/40, saturation 0/40, CLIP
  zero-shot 17/40. About 32 of DINOv3's 55 disagreeing "air" windows show the lens under water (waves over a logging
  whale; the depth port cannot see them; AI by-eye, one frame each): part of its false events are real.
- New deployment, dev clip (4 events, 2 air windows): every method gets the 2 air windows; CLIP probe + hysteresis
  4/4 events, 0 false; DINOv3 chain 2/4 (3/4 within 1 s), 4 false (exhalation bubble clouds near the surface).
- Window step: removes isolated flips; per-frame thresholds give 263 false events on Zenodo (DINOv3), 18 on the dev
  clip, 422 on check (CLIP zero-shot); the chain cuts these to 58 / 4 / 61.
- With a log (WP3's 41 by-eye events, 00004-00009): depth + fixed lags and the picture methods are equally exact
  (median |error| 0.09-0.15 s, all < 0.45 s); the picture adds little to timing when the lag is known, but recovers an
  unknown video-to-sensor offset (Zenodo: picture minus depth +0.99 / +0.43 s vs by-eye +1.15 / +0.65 s) and sees
  wave wash the depth threshold misses.
- Cost (5 frames/s, shared RTX 4000 Ada): per hour of video decode 61 s, colour 45 s CPU, CLIP-L 81 s, DINOv3 73 s GPU.
- Two runs: v1 picked the best step on the fit pieces; v2 (default) picks the least smoothing among near-ties,
  decided after v1's dev-clip result (v1 erased a 1.8 s exposure there), so v2's dev numbers are not clean; Zenodo
  was first scored after v2 was fixed. Shipped: `uwid.waterstate.predict_video` (DINOv3 + logistic + chain,
  parameters `src/uwid/waterstate_params.json`, 49 KB, no model weights); `method="color"` CPU-only, same-tag only.
- Caveats: one fitting deployment; dev clip tiny; Zenodo truth is depth with a by-eye time fix and misses wave wash;
  no floating-tag, failed-drop, deck or night footage scored (not on disk / not in these pieces); one drone drop.
- Outputs: `experiments/03_water_state/` (scripts, `tables.md`, `prompts.json`), `src/uwid/{eval_state,waterstate}.py`,
  `src/uwid/waterstate_params.json`; cache `underwater_cache/waterstate/` (features, predictions, results*.json,
  zenodo_truth/), downloads `underwater_cache/public/zenodo_4902325/`; sheets
  `review_sheets/underwater/waterstate_{false_air,missed_air,crossing_errors,crossings_check,crossings_dev,crossings_zenodo,drop,zenodo_dino_disagreements}.jpg`,
  `waterstate_scores_{00004,dev,Z00011,Z00013}.png`.

## 2026-11-20 20:55  02_reference_labels (WP3 part 2): 7 test pieces of 2025-02-03 labelled, 840 frames, drone drop + 26 breaths, 14 mask frames; fish in 45 frames, no seafloor
AI by-eye labels on TEST videos (scoring only, nothing fitted). Pieces CC-11-96-...-00004 to -00010 (the seven fully
copied `.m4v` at sampling time; 00011 onward still copying, not labelled). Per piece 120 frames at even spacing with a
seeded offset (seed 20250203 + piece) and two random 180 s windows (seed 1000 + piece) stepped through for events.
00004 labelled by me (incl. the drone drop); 00005-00010 by helper agents from `LABELLING_GUIDE_PART2.md`; all fish
calls re-checked under the user's stricter rule (body and tail visible) before this entry.
- Frames (840): underwater 821, air 19 (7 on the drone), crossing 0; view clear 836, bubbles 4; another whale in 221
  (5 with two), faint shapes in 52; back on 825; seafloor none on all 840 (looked for on purpose).
- Fish: 45 frames (44 few, 1 school: 00006 f13991), 110 fish boxed in `fish_calls.json`; many tail-less specks moved
  to `unsure` in `other`. Every call is on `review_sheets/underwater/reference_fish_candidates_01..15.jpg`.
- Drone drop (00004): on drone f0-2722, release f2723 (90.92 s), hit f2747 (91.72 s), entry f2796 (93.35 s), settled
  about f2830; a hit, not a miss. Breaths stepped through in windows: 26 (each 2.3-3.3 s in air); list in tables.md.
- Masks: 14 frames, SAM 2.1 (on the CPU) from my boxes / points and the fish boxes: 8 whales (2 clear, 6 hazy),
  39 fish, back, drone parts (`gear`, new class 8), air, entry froth.
- Uncertain: whale parts at the back's horizon (own fin vs calf; rule: count only if it moves relative to the back),
  fish of 45-60 px with blurred tails, crossing starts (about 3-5 frames), 00009's exits moved 6-11 frames under the
  half-out rule. Helper labels were not re-labelled by me; spot-check them on the sheets.
- Class findings: no seafloor anywhere so far; entry froth looks like sand; Snell's window just under the surface
  shows the back's brown reflection (still underwater); objects fixed on the back (scar, tag part) are `back`.
- Outputs: `experiments/02_reference_labels/labels/CC-11-96-20250203-*/`, README part 2, tables.md; masks in
  `underwater_cache/reference/<stem>/masks/`; sheets `reference_test_<piece>_frames_NN.jpg`,
  `reference_test_<piece>_masks_01.jpg`, `reference_fish_candidates_NN.jpg`; DONE files per piece.

## 2026-11-20 19:53  WP0 scaffold: env + uwid package; clock read on 100 % of frames; tag .mov is the time-true input
Built `.venv` (torch 2.6.0+cu124, transformers 5.20.1, PyAV 19.0.2, imageio-ffmpeg 0.8.1 with ffmpeg 9.0.2, CUDA
visible) and `src/uwid/` (video, audio, timecode, sheets, inventory, clips, cli `uwid`); 25 pytest tests on synthetic
videos pass. Checks on the real footage (no content labelled or analysed):
- Inventory (`uwid inventory`): dev clip + 5 tag recordings x 2 forms on disk at 19:44; 00009 still copying, skipped.
  Every frame of every file decodes, 0 decode errors. Dev clip: 2,857 shown of 3,020 packets; its edit list hides 161
  frames at the START (clock 11:29:31.3-36.5) and 2 at the end, not the end only. Tag `.mov`: HEVC, 1,779 fragments,
  keyframe every 1.00 s, timestamps jitter 31-36 ms (variable), AAC 32 kHz mono, no creation time. `.m4v`: H.264
  (VLC conversion dated 2025-02-05), constant 29.95-29.97 frames/s, keyframes 3-8 s apart, MP3 44.1 kHz stereo.
- `.mov` vs `.m4v`: same frames in the same order (00004: frames 8,992 / 26,976 / 44,960 match one to one, NCC
  0.999, shift < 0.04 px, no crop), but the `.m4v` lacks 1-43 frames at the end (00004: 41, 00005: 43, 00006: 1,
  00007: 19, 00008: 1), its constant-rate timeline drifts from the clock by up to 1.48 s per 30 min, and its levels
  are squeezed (m4v ~ 0.83 x mov + 17; PSNR 50 dB after undoing that, 30-34 dB raw). Both carry the burned-in clock
  (it is in the tag's own video). Default input: the `.mov`.
- Clock reader (glyphs learned on the dev clip only): dev clip all 2,857 frames read, monotonic, slope 1.000014,
  largest residual 0.051 s (the 0.1 s display step). Tag 00004 at 2 frames/s: 3,563 / 3,563 read in both forms,
  monotonic; `.mov` slope 0.9999999, largest residual 0.017 s; `.m4v` slope 0.99922. First clock 10:12:38.100 is
  2.1 s after the file name's 10:12:36 (00005-00008: +2.04, +1.10, +2.11, +2.08 s). Consecutive pieces leave about
  25 s unrecorded (24.8, 25.6, 25.7, 25.6 s from one piece's last clock to the next one's first).
- Clips from the dev clip: copy 20-30 s started on the keyframe at 17.28 s (frame 518, 384 frames); re-encode
  40-45 s gave exactly frames 1,199-1,348 (150). Decoded clips match the source frames the index names at both ends
  (clock text equal, NCC 1.0 copy / 0.998-0.9998 re-encode). Copy from a tag `.mov` copied the sound from the file's
  start (a fragmented-file quirk), so copy mode now re-encodes the sound; after that sound and picture agree (sound
  correlation 0.975 with the source, offset = the clip's 0.032 s video start).
Caveats: one sound-sync check, one picture comparison (00004); month letters other than "Dec" come from the file
name. Outputs: `underwater_cache/inventory/inventory.{json,md}`, `underwater_cache/wp0_checks/` (check scripts and
JSON), `underwater_cache/clips/wp0_test*/`, `underwater_cache/timecode/glyphs.json`,
`review_sheets/underwater/inventory_*.jpg`, `review_sheets/underwater/wp0_mov_vs_m4v_crops.jpg`.

## 2026-11-20 20:05  01_metadata (WP1): depth answers "surfaced" to ~0.3 s; 3 drone drops, 80 surfacings; burned clock = sensor + 2.03 s; sound track empty
Data: one deployment, `2025-2-3 Mini-mother 11-96` (tag CC-11-96): the whole 3.1 GB sensor CSV (6.74 h, 400 rows/s,
no gaps) and the `.mov` / `.m4v` pieces 00004-00008 (the ones fully copied by 19:41; 14 pieces were recorded). No
fitting on video except: digit templates for the clock reader (from 00004 frames) and one sound threshold chosen on
00004 and tested on 00005-00008. Picture-side checks are AI by-eye readings of frame strips, named as such.
- No wet/dry signal: SWS is listed as available but not activated in the `.txt`, and is not in the CSV. Depth (10 Hz,
  0.031 m steps, air reading -0.062 m, sd 0.048 on deck, 27 glitch readings of 1529.5 m) is the wet/dry sensor:
  on the whale its histogram has a separate air peak, threshold air + 0.10 m.
- Timeline from sensors alone: deck; 3 drone flights (motor vibration 1.8-9.4 m/s2); drop 1 09:14:58.15 and drop 2
  09:35:41.59 failed (tag floated 2.5 / 2.2 min, recovered); a 2.2 min float test 09:59:30; drop 3 free fall
  10:14:06.948-10:14:07.785 (0.84 s, ~3.4 m), depth > 1 m 2.4 s after impact; on the whale 4 h 24.6 min; released at
  21.3 m at 14:38:44 (gyro jump), floated 25.5 min, recovered 15:04:48. Free fall in the picture 0.80 s.
- 80 surfacings (port out of water) in 72 bouts, 3.0-4.4 s each; 86 dives > 1 m, max 177 m. `surfacings.csv`
  documented in `tables.md`; against the picture (AI by-eye, 9 exits + 9 entries) the lens is out ~0.3 s after
  `start_t` and back under ~0.4 s before `end_t`, +-0.3 s.
- Time link (.mov): burned-in clock (present in both .mov and .m4v) - 2.03 s = sensor time (drone release + impact
  frames vs accelerometer, +-0.05 s); clock vs pts drift < 3 ppm; 48 surfacings in 5 files confirm it with no trend
  (automatic saturation cue, entry MAD 0.10 s). `Camera time` gives each file's first frame to 0.1 s from the log
  alone (resumes 6.87 s after it); file names are off by up to 0.9 s. `.m4v` pts drift up to -807 ppm (1.4 s per
  file): use their own clock fit. 2 of 80 surfacings fall in the 24 s gaps between files.
- Sound: the `.mov` track is a constant -77.4 dBFS noise floor in air and water; threshold test balanced accuracy
  0.51-0.60 on 00005-00008: sound cannot tell anything in this deployment. CSV `Hydrophone` column: no information.
- Not on a whale: 33.6 % of the recorded video (7,931 of 23,605 s: deck 4,802, flight 1,175, floating 1,954);
  pieces 1-3 and 13-14 contain no whale time. Floating is distinct (|gyro| median 780 vs 23 on the whale at depth,
  159 at the surface; depth -0.03..0.12 m; float orientation); successful vs failed drop decided by depth > 1 m
  within seconds; n = 3 drops, so the rule is plausible, not measured.
- Caveats: one tag, one day; meaning of Flags `T`, GPS 3/4, Camera codes guessed from the data; depth after recovery
  reads down to -3.4 m (unexplained); no GPS positions (only raw RXM-RAWX in the .ubx).
- Outputs: `src/uwid/tagdata.py`; `experiments/01_metadata/{build_cache,analyze_sensors,probe_videos,clock_ocr,
  sync_video,summarize_sync,sound_check,video_states,make_plots}.py`, `tables.md`; cache
  `underwater_cache/tagdata/2025-2-3 Mini-mother 11-96/` (sensors_50hz.parquet, surfacings.csv, dives.csv,
  states.csv, drops.csv, camera_log.csv, clock_fits_{mov,m4v}.csv, sync_*.csv, video_state_*.csv, *.json);
  plots `review_sheets/underwater/metadata_{deployment,zoom_1h,surfacing_1..6}.png`.

## 2026-11-20 19:50  02_reference_labels (WP3 part 1): dev clip labelled: 191 frames, 4 events, 31 mask frames; fish found once
AI by-eye labels (Claude), not ground truth. Data: `2025-12-20 CG.m4v` only (2,857 frames decode; every 15th frame =
191 frames at 2 per second, each looked at 960 px wide or larger, full-size crops where needed).
- Frames: underwater 180, air 8, crossing 3; view clear 149 / bubbles 42; other whales clearly visible 0-5 per frame
  (0 on 46 frames), faint shapes on 103 frames; `back` (tagged whale's body) on 182; seafloor none on all 191 (the
  clip shows no seafloor; the textured surface at the bottom is the back).
- Fish: one sighting, a group of 6-8 small dark fish (15-20 px) at f1480-f1515 (49.4-50.5 s), followed over frames at
  full size; `few` on 3 sampled frames. Everything else fish-like checked was wave facets, bubbles or foam.
- Events (rule: first frame whose centre shows the other side of the surface): entry f24 (0.80 s), exit f437
  (14.58 s, crossing f417-436), entry f492 (16.41 s), exit f2806 (93.61 s, crossing f2796-2805).
- Masks: 31 frames, SAM 2.1 hiera-large from my boxes / points, hand polygons where SAM failed; 85 whale instances
  (clear 25, hazy 40, faint 20), 13 fish, back / bubbles / surface / air regions; 6 frames without other whales.
- Uncertain: whale totals on frames with faint shapes, about 9 bubbles-vs-clear borderline frames, the owner of a
  huge pectoral fin at f2175-2190, f240 pale streak, foam-phase start of the two exits (about 3 frames), SAM speckle
  on 4 hazy whales. Details and rebuild commands: `experiments/02_reference_labels/README.md`, counts `tables.md`.
- Outputs: `experiments/02_reference_labels/labels/2025-12-20_CG/{frames.csv,events.json,prompts.json}`, masks in
  `underwater_cache/reference/2025-12-20_CG/masks/`, sheets `review_sheets/underwater/reference_dev_frames_01..32.jpg`
  and `reference_dev_masks_01..08.jpg`.
- Correction 2026-11-20 20:20: the user judged the 15-20 px shapes at f1480-f1515 not fish. The dev clip has no fish:
  f1485 / f1500 / f1515 set to `fish` none (`unsure` in `other`), the 13 fish mask instances repainted `unsure`,
  masks f1485 / f1500 and all dev check sheets rebuilt, README and tables corrected. Rule since: specks are `unsure`.

## 2026-11-20 19:35  docs/research (WP2): model shortlist, fish / seafloor data, a public tag deployment
Web research only, nothing run or measured here; write-up `docs/research/models_and_datasets.md` (every claim tagged
read / search / recall / inference). Entry written by the orchestrator from the agent's report.
- Test order: OWLv2-base boxes + SAM 2.1 masks (both cached); LLMDet-large (open, 1.4 GB) + SAM 2.1; SAM 3 (gated,
  manual approval; needs the user's Hugging Face login); YOLOE-26 (AGPL) as the fast option for whole folders. Fish:
  Community Fish Detector (RF-DETR). Seafloor: cached Mask2Former ADE / COCO-panoptic have `sand`, `sea`, `rock`,
  `water`; a coral-reef SegFormer has `sand` and `fish`.
- Data: UIIS10K (0.76 GB, masks) and Brackish (0.54 GB, boxes) for fish; Open Images V7 val + test for whale against
  fish (199 whale, 1,328 fish images, mostly above water, 24 with both); SUIM / UIIS for seafloor. No public
  underwater set has whales and fish together in quantity.
- Zenodo record 4902325 (CC0, Gough and Cade): a public camera-tag deployment with video, sensor CSV and the
  toolbox's video timing file. Record checked by the orchestrator through the Zenodo API (open, cc-zero, 96 files).
- Sync, from the tag toolbox's documentation: video start time comes from the file name; the burned-in clock "may be
  1-2 s delayed"; sensor, sound and camera clocks are separate; the recommended alignment matches the camera's sound
  against the hydrophone on impulsive sounds such as the tag breaking the surface.
- Prior work: no published method that detects other whales in video from a camera on a whale, and no published
  number for these models on whales underwater. A 2026 turtle-camera study classified head above / below the surface
  per frame at recall 99.1 %, precision 63.3 %, and needed smoothing over time.
- Not verified: SAM 3 memory at 1080p in video mode, SAM 3 through transformers on torch 2.6, whether the fish
  detector's training data labels marine mammals as fish.

