# 09_seafloor_sand (WP10 part 2): seafloor, sand and dark per 1 s window

> **WITHDRAWN 2026-11-23 (user review of `seafloor_01..05`): the seafloor tables below (setup, seafloor per window,
> per pixel, the back confusion) measure agreement with wrong labels.** The labeller's `visible` spans (00017 842-1390 s,
> 00023 0-306 s) show the tagged whale's back with haze above it, and the classifier learned that band. The user:
> "Seafloor_01 is a whale back, so is 02, 03 is just hazy water, 04 is a mixture of hazy water, a crowded whale scene,
> and reflection off surface from below, 05 is also just a whale back." The dark rule and the cost / decode notes
> stand. The relabelling (WP12) is in the section "WP12" at the end of this file.

Run 2026-11-22. Truth everywhere = AI by-eye labels (WP3 / WP3b), not ground truth: the 2025-01-02 masks (29 frames,
seafloor painted on 15 as a coarse band with an `unsure` band above it), `seafloor_spans.csv` and
`blocked_spans.csv` (spans set at 10 s, then 1-2 s steps; ends good to 2-10 s for seafloor, about 5 s for the
sand / dark split). **Seafloor exists on one deployment only, in two recordings (00017 and 00023): the same day, the
same tag, the same sandy bottom seen from a whale resting on it.** Every seafloor number below is a
leave-one-recording-out number inside that one deployment; sand is a within-recording number.

## Setup

- Frozen DINOv3 ViT-B/16, 1280 x 720 input, a linear classifier per patch (`uwid.patchseg`, as WP5b), classes water,
  whale, back, fish, bubbles, surface, air, seafloor, gear. Patch label rules as WP5b (75 % purity).
- Windows: the 1 s grid; two frames per window at 0.25 s and 0.75 s (the scan's frames) decoded at 1280 x 720 from
  the `.mov` (all six on-whale recordings of 2025-01-02, 5,656 windows, and 2025-02-03 00006, 1,782 windows); the
  2025-12-20 clip's labelled frames (2 per window, 91 underwater windows) and the 821 underwater labelled frames of
  2025-02-03 (1 per window). Window seafloor share = mean over its frames of the share of patches called seafloor.
- Air breaths (`blocked_spans.csv` kind air) are left out of the 2025-01-02 windows (the scan knows them from depth).

| head | fitted on (mask frames) | scored on |
|---|---|---|
| test17 | dev 31 + 2025-02-03 14 + 2025-01-02 00023 (16) + 00024 (4) | 2025-01-02 00017 (masks and windows) |
| test23 | dev 31 + 2025-02-03 14 + 2025-01-02 00017 (9) + 00024 (4) | 2025-01-02 00023 |
| test_d1 | dev 31 + 2025-01-02 29 | 2025-02-03 (no seafloor anywhere) |
| test_dev | 2025-02-03 14 + 2025-01-02 29 | the 2025-12-20 clip (no seafloor) |
| all (delivered, `underwater_cache/seafloor/heads/head_all.pt`) | all 74 mask frames | nothing |

Window threshold on the seafloor share: chosen on the TRAINING recording's windows only (best balanced accuracy on a
fixed grid): test17 0.15 (from 00023), test23 0.075 (from 00017), test_d1 / test_dev / all 0.15 (from 00017 + 00023).
A fixed 0.05 is shown next to it.

## Seafloor per window

| recording (head) | threshold | seafloor visible: found | no seafloor: called | `unsure` spans: called |
|---|---|---|---|---|
| 2025-01-02 00017 (test17) | 0.15 chosen | 541 / 548 | 0 / 282 | 49 / 546 |
| 2025-01-02 00017 (test17) | 0.05 fixed | 547 / 548 | 43 / 282 | 127 / 546 |
| 2025-01-02 00023 (test23) | 0.075 chosen | 305 / 306 | 25 / 981 | 0 / 10 |
| 2025-01-02 00023 (test23) | 0.05 fixed | 306 / 306 | 34 / 981 | 0 / 10 |

False seafloor on recordings and sources with no seafloor at all:

| windows | head (threshold) | called seafloor |
|---|---|---|
| 2025-01-02 00010 (616) | test17 (0.15) / test23 (0.075) | 0 / 1 |
| 2025-01-02 00011 (1,499; dives to 50 m, open water) | test17 / test23 | 3 / 4 |
| 2025-01-02 00012 (220) | test17 / test23 | 0 / 0 |
| 2025-01-02 00024 (386; its 4 mask frames were in both heads' training) | test17 / test23 | 0 / 4 |
| 2025-02-03 00006 dense (1,736 underwater on-whale windows) | test_d1 (0.15) | 22 (1.3 %); 68 at 0.05 |
| 2025-02-03 labelled frames (821 windows) | test_d1 (0.15) | 6 (0.7 %); 249 at 0.05 |
| 2025-12-20 clip (91 windows) | test_dev (0.15) | 1; 4 at 0.05 |

What the false alarms are (sheets `seafloor_03..05`): on 2025-02-03 a dark haze band just above the back in green,
turbid water (the class map paints a seafloor band right where the training masks had one: a band above the back);
on 2025-01-02 00023 the resting view after the bottom has faded (the same geometry with the bottom gone); on 00010 a
pale, sand-coloured close-up at 406 s labelled `none` (it may be the bottom or a water sheet; not checked further);
on the dev clip entry froth with a whale.

## Seafloor per pixel and the back confusion (held-out mask frames; share of the truth class's pixels)

| head on | frames | seafloor found | seafloor IoU (12 px border ignored) | seafloor called back | back called seafloor | seafloor called water | water called seafloor | bubbles called seafloor |
|---|---|---|---|---|---|---|---|---|
| test17 on 00017 | 9 | 96.9 % | 0.87 (0.90) | 2.2 % | 0.4 % | 0.9 % | 5.5 % | 0 |
| test23 on 00023 | 16 | 79.5 % | 0.78 (0.83) | 1.7 % | 0.4 % | 17.5 % | 0.1 % | 0 |
| test_d1 on 2025-02-03 | 14 | (none) | | | 0.2 % | | 2.1 % | 0 |
| test_dev on the dev clip | 31 | (none) | | | 0.03 % | | 1.8 % | 1.8 % |

Back called seafloor stays below half a percent of back pixels on every source, including 2025-01-02 where the back
fills the lower half of the frame; seafloor called back 1.7-2.2 %. In windows, none of the 2025-01-02 false seafloor
windows is the back itself (sheets); the 2025-02-03 false windows put the seafloor band on the haze ABOVE the back,
not on it. The froth / brown-sheet look-alikes the labeller marked do not fire (bubbles called seafloor 0 % on the
held-out 00023 masks, which include the froth frame f33916; the 00024 froth frame was in both heads' training, so it
is not a test).

## Sand (stirred-up sediment hiding the view; `blocked_spans.csv` kind sand, only in 00017)

Frame embedding = mean DINOv3 patch feature of the frame and of its four quarters (3,840 numbers); window = mean of its
two frames; logistic regression (C 0.001, balanced). Dark windows (rule below) and air breaths left out.

| test | fitted on | sand windows found | other windows called sand |
|---|---|---|---|
| 00017, leave one 100 s block out (5 s buffer); WITHIN ONE RECORDING | the other 00017 blocks + all windows of 00010, 00011, 00012, 00023, 00024 as negatives | 164 / 190 | 35 / 956 |
| 2025-01-02 00010 / 00011 / 00012 / 00023 / 00024, each left out | 00017 + the other four | | 0 / 616, 0 / 1,499, 0 / 220, 3 / 1,297, **103 / 386** |
| 2025-02-03 00006 dense | 2025-01-02 only | | **728 / 1,736** |
| 2025-02-03 labelled frames | 2025-01-02 only | | **581 / 821** |
| 2025-12-20 clip | 2025-01-02 only | | 3 / 91 |

A first run with 00017 negatives only: 172 / 203 found, 36 false inside 00017, but 104 / 386 on 00024, 1,283 /
1,736 on 2025-02-03 00006 and 37 / 91 on the dev clip. A low-dimensional model (grey levels + mean class
probabilities) did worse (177 / 190 found, 63 / 956 false inside 00017; 85 / 1,297 on 00023, 111 / 386 on 00024,
903 / 1,782 on 2025-02-03). **Verdict: sand is found inside the one recording that has it, and the model does not
carry to another deployment** (it calls most of 2025-02-03's green, hazy water sand). The delivered model
(`underwater_cache/seafloor/sand_all.npz`) adds the 2025-02-03 and dev windows as negatives; its false-alarm rate is
therefore unmeasured. The scan should not keep windows on `sand` alone until a second recording with sand is
labelled.

## Dark (black or covered lens; no model)

Rule: every frame of the window has mean grey (0-255, 320 px copy) below 20. Truth: `blocked_spans.csv` kind dark in
00017 (215 windows). Caveat: the labeller used "mean brightness below 20" to place the dark ends inside 00017, so the
check there is partly circular; the false-alarm counts elsewhere are not.

| threshold | 00017 dark found | 00017 other windows called dark (inside sand spans) |
|---|---|---|
| 10 | 189 / 215 | 14 (12) |
| 20 (delivered) | 193 / 215 | 15 (13) |
| 30 | 195 / 215 | 18 (15) |
| 40 | 198 / 215 | 21 (18) |

Everywhere else: 0 windows dark at 20 or 30 (2025-01-02 00010, 00011, 00012, 00023, 00024: 4,018 windows, darkest
window mean grey 57; 2025-02-03 00006 dense 1,736, darkest 55; 2025-02-03 labelled 821, darkest 50; dev 91,
darkest 62). The 22 dark windows missed are at span ends (spans are good to about 5 s) or dark-green murk above 20.

## Cost (from the timings of the runs above; the separate cost run was not repeated, see below)

| step | measured | per video hour |
|---|---|---|
| patch classes (DINOv3 1280 x 720 + five linear heads; one head costs the same within noise), shared RTX 4000 Ada | 2,780 frames of 00017 in 158 s model time = 57 ms per frame; 1.1 GB GPU | 7,200 frames at 2 per window = about 410 GPU s = 0.11 GPU-h |
| decoding the 4K `.mov` to 1280 px (PyAV, 8 threads, CPU) | 00017: 1,390 s of video in 406 s; 00011: 1,539 s in 326 s | about 0.2-0.3 CPU-h (8 threads) |
| decoding a 1080p `.mov` | 2025-02-03 00006: 1,800 s in 68 s | about 0.04 CPU-h |
| dark rule | grey mean of a 320 px copy | negligible |
| sand | one dot product per frame on the embedding | negligible |

The scan already decodes the same two frames per window for the detector (at 1920 px): calling
`patchseg.frame_classes` on those frames (resized to 1280 px) adds only the 0.11 GPU-h.

What went wrong with `cost.py` (2026-11-22/23): its first launch had no `__main__` guard; the spawned decode child
failed at start-up, and the parent (`uwid.video.iter_frames_isolated`) waited on the pipe for ever while holding the
GPU lock (1 h 43 min, ended by the orchestrator, a queued second copy too). Changes: `window_classes` now decodes
through `patchseg._iter_decoded`, which reads with a timeout, checks that the child is alive and raises (test
`test_decode_child_failure_raises_not_hangs`); `window_classes` logs before loading, before decoding and after
classifying; every locked command from now on is `flock <lock> timeout 900 nice -n 10 ...`, one copy at a time.

## Files

Scripts: `decode.py` (window frames to a temporary cache folder, deleted afterwards with `--clean`), `gpu.py heads
frames`, `score.py`, `sheets.py`, `cost.py` (not run). Results: `results/seafloor.json`, `pixels.json`, `sand.json`,
`dark.json`, `rules.json`. Cache: `underwater_cache/seafloor/` (heads, per-frame maps and embeddings
`frames/*.npz`, `d2_windows.parquet`, `windows_*.parquet`, `sand_all.npz`). Sheets:
`review_sheets/underwater/seafloor_01..05.jpg` (hits, misses and false alarms on all three sources).

## WP12 (2026-11-23): where the real seafloor is (search only; nothing fitted yet)

AI by-eye look by the WP10 agent; the user has not yet seen it. Candidate moments from the sensors and the picture
(`candidates.py`), not from the old labels: 60 s around every dark window (mean grey < 20; only 00017 has any), every
stretch within 20 % of the recording's maximum depth that is flat (20 s rolling std < 0.3 m) with 30 s margins, and in
a second pass every stretch deeper than 70 % of the maximum in 00011 and 00023. The 1 Hz sensor table has no
accelerometer, so contact from motion was not used. Frames at 1 s steps, 1280 x 720, looked at in 2 s steps (4 s in
the second pass) and at 1 s around every boundary; black frames skipped.

| recording | looked at (s) | seafloor visible | sediment | unsure | verdict |
|---|---|---|---|---|---|
| 00010 | 430-538 (deepest, 34 m) | 0 | 0 | 0 | none |
| 00011 | 84-215, 602-695, 1155-1349 (to 51 m) | 0 | 0 | 0 | none: open water above the back on every deep dive |
| 00012 | 81-204 (5 m) | 0 | 0 | 0 | none |
| 00017 | 268-1390 | **94 s in 5 episodes**: 292-327, 405-422, 465-487, 510-524, 527-533 | 128 s | 323 s (mostly black lens and green murk) | the only recording with a recognisable bottom |
| 00023 | 0-328, 897-958, 1167-1238 | 0 | 0 | 25 s (286-301, 1180-1190: a dark speckled band beyond the back) | no clear bottom; the old `visible` 0-306 s is the back with haze |
| 00024 | no candidate (no dark window, no flat deep stretch; max 30 m) | | | | not looked at |

The bottom looks like what the orchestrator saw: a rough, mottled surface at the SIDE of the frame (upper right
292-327 and 510-533, a wall on the right 405-422, a wall on the left 465-487) while the whale is rolled and scrapes
along it; sediment clouds rise from it right after (327-336, 422-440, 533-536, 540-557, 610-622, 736-765), with black
or covered-lens stretches in between. None of it is in the old `visible` spans. 94 s of visible bottom in one
recording is very little: enough for a sheet and a few dozen mask frames, not for a trustworthy pixel class.

Sheets for the user (before anything is fitted): `review_sheets/underwater/seafloor_candidates_best_01.jpg` (the 24
clearest, all 00017), `seafloor_candidates_00017_01..06.jpg`, `seafloor_candidates_00023_01.jpg`,
`seafloor_candidates_00010_01.jpg`, `_00011_01`, `_00012_01` (frame at 640 px, a CLAHE-stretched copy beside it,
time and call). Calls: `candidate_calls.csv` (75 frames: 37 seafloor, 21 sediment, 9 unsure, 8 none).

Prepared, not used: new `seafloor_spans.csv` for 00017 and 00023 at 1 s (kinds visible / sediment / unsure / none;
`unsure` also covers black frames), the old rows in `seafloor_spans_withdrawn.csv` beside them.

Sand labels against this look (`blocked_spans.csv` kind sand): the clouds at 332-336, 535-565, 615-625 and
735-785 are sediment as labelled; but 405-422 is the bottom itself (labelled sand), 645-675 is the back in haze with
no cloud (labelled sand), and clouds at 422-440, 449-459, 510-523 and 610-614 are partly or not labelled. The sand
labels hold for the big clouds and are wrong at the bottom-contact moments; the sand result above is withdrawn with
the seafloor result until they are redone from the WP12 spans.

## WP12 step A (2026-11-23): masks on user-confirmed bottom, a re-fitted head (v2), measured inside one recording

**What it rests on.** The user confirmed `seafloor_candidates_best_01.jpg` ("all of those seafloor candidates now
show both the back of the whale in foreground, and the seafloor in background"; a combined back + seafloor class is
an accepted fallback). Visible bottom exists in ONE recording (2025-01-02 00017), on one day, one bottom, in 4
episodes: E1 292-327, E2 405-422, E3 465-487, E4 510-524 + 527-533 s, 94 s in all. Neighbouring windows of an
episode are near copies.

**Labels.** 22 mask frames (E1 4, E2 5, E3 6, E4 7; 298 and 304 s too faint to outline), AI masks by the WP10 agent
on user-confirmed frames (`wp12_prompts.json`): seafloor as polygons by eye, the back by SAM 2.1 (box + points), the
edge of the bottom in haze, sediment clouds and the dark band between the back and the bottom = ignored. Old
2025-01-02 masks: the withdrawn seafloor paint is ignored (never a target). Sheets `seafloor_masks_01..02`.

**Heads** (`underwater_cache/seafloor/heads/head_v2_*.pt`; `head_all.pt` and `patchseg.SEAFLOOR_CLF` unchanged):
linear head on DINOv3 1280 px patches (as WP5b), fitted on dev 31 + 2025-02-03 14 + old 2025-01-02 29 (seafloor
ignored) + the 22 new frames. Fold k (`v2_fold1..4`) leaves out every mask frame within 5 s of episode k (new and
old); `v2_test_d1` leaves out 2025-02-03, `v2_test_dev` the dev clip; `v2_all` sees all (9,159 seafloor patches).
Window = 2 frames (0.25 / 0.75 s), seafloor share = mean share of patches called seafloor; threshold per fold
chosen on 00017 windows outside the held-out episode +- 5 s only (other episodes vs `none` spans): fold1 0.15,
fold2 0.03, fold3 0.05, fold4 0.05; v2_test_d1 / v2_test_dev / v2_all 0.05 (all 00017 episodes vs none).

### Pixels on held-out episodes (each fold head on the mask frames of its own held-out episode)

| episode (frames) | seafloor found | seafloor IoU (12 px border ignored) | seafloor called back | seafloor called water | back called seafloor | water called seafloor |
|---|---|---|---|---|---|---|
| E1 (4) | 86.1 % | 0.86 | 0 | 13.9 % | 0.04 % | 0.0 % |
| E2 (5) | 93.2 % | 0.91 | 0 | 6.2 % | 0.0 % | 1.3 % |
| E3 (6) | 100.0 % | 0.94 | 547 px | 0.0 % | 0.14 % | 2.8 % |
| E4 (7) | 98.7 % | 0.84 | 0 | 1.2 % | 0.28 % | 5.3 % |
| all 22 | 96.7 % (5,227,026 / 5,407,044 px) | 0.89 (0.91) | 547 / 5,407,044 px | 3.1 % | 0.13 % (23,729 / 18,651,738 px) | 2.7 % (426,046 / 15,749,739 px) |

Combined class (back + seafloor = background), same frames: background found 99.0 % (23,823,533 / 24,058,782 px),
IoU 0.96 (0.98 with the border ignored), background called water 0.9 %, water called background 4.4 % (693,789 /
15,749,739 px), background called whale 7,953 px (0.03 %, nearly all in E2); no other whale in these frames.

### Whale safety: whale pixels called seafloor or back (reference masks)

| whale masks | new head | whale called seafloor | whale called back | old head | whale called seafloor | whale called back |
|---|---|---|---|---|---|---|
| 2025-02-03, 14 frames (1,052,352 px; heads not fitted on 2025-02-03) | v2_test_d1 | 0 | 0.21 % | test_d1 | 0 | 0.34 % |
| dev clip, 31 frames (4,987,442 px; heads not fitted on it) | v2_test_dev | **2.9 % (144,322 px)** | 0.43 % | test_dev | 0.40 % (19,962 px) | 0.78 % |
| 2025-01-02 (3,806,963 px; IN-SAMPLE, both heads saw these masks) | v2_all | 14 px | 0.02 % | all | 0.34 % | 0.03 % |

### Windows

Held-out episodes (the fold head never saw a mask or a window within 5 s): E1 0 / 35 (threshold 0.15 chosen by
fold 1 on the large episodes; E1's bottom is small and faint, median share 0.064: post hoc 20 / 35 at 0.05, not a
result), E2 17 / 17, E3 22 / 22, E4 20 / 20.

False seafloor windows (fold 1 / 2 / 3 / 4, windows within 5 s of the fold's held-out episode left out):

| windows | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|
| 00017 `none` 0-292, 644-673 s | 8 / 302 | 36 / 307 | 34 / 307 | 32 / 307 |
| **00017 HARD NEGATIVE: withdrawn `visible` 842-1390 s** | 0 / 524 | 3 / 524 | 3 / 524 | 1 / 524 |
| 00017 sediment (clouds right after the bottom) | 61 / 123 | 84 / 123 | 93 / 128 | 77 / 122 |
| 00017 unsure, black (grey < 20) | 0 / 208 | 1 / 207 | 0 / 204 | 0 / 206 |
| 00017 unsure, not black (murk) | 9 / 115 | 61 / 111 | 43 / 109 | 32 / 110 |
| 00010 | 0 / 616 | 9 / 616 | 1 / 616 | 4 / 616 |
| 00011 (deep dives, open water) | 0 / 1,499 | 41 / 1,499 | 27 / 1,499 | 38 / 1,499 |
| 00012 | 0 / 220 | 0 / 220 | 0 / 220 | 0 / 220 |
| **00023 HARD NEGATIVE: withdrawn `visible` 0-306 s without 286-301** | 0 / 291 | **103 / 291** | **62 / 291** | **67 / 291** |
| 00023 WEAK POSITIVES 286-301, 1180-1190 s (not fitted on) | 0 / 25 | 13 / 25 | 9 / 25 | 1 / 25 |
| 00023 rest | 1 / 981 | 9 / 981 | 11 / 981 | 9 / 981 |
| 00024 | 0 / 386 | 3 / 386 | 7 / 386 | 4 / 386 |

Sources not in the head's training: 2025-02-03 00006 dense 4 / 1,736 (v2_test_d1), 2025-02-03 labelled frames
2 / 821, dev clip 8 / 91 (v2_test_dev; the old head 1 / 91). The old head's false band on 2025-02-03 (22 / 1,736) is
down to 4. On the sheets (`seafloor_false_01`): the worst false windows are a sediment cloud (00017), a near whale's
dark underside (2025-02-03 00006: a whale-safety case), a bubble curtain (dev), the resting back with haze (00023).

### Verdict

- **Separate seafloor class: not usable as a scan label yet.** It finds the large, clear bottom of a held-out
  episode (59 / 59 windows of E2-E4) but not the faint one (E1 0 / 35 at its fold's threshold); at the thresholds
  folds 2-4 chose it calls 21-35 % of 00023's resting back-with-haze windows seafloor (the user's counterexample)
  and 2-3 % of 00011's open-water dives; on the dev clip it calls 2.9 % of whale pixels seafloor (old head 0.4 %).
- **Combined back + seafloor ("background"): usable as a pixel class for masks**, not as a window label: IoU 0.96 on
  held-out episodes, 0.03 % into whale on those frames, whale pixels called back or seafloor 0.21 % (2025-02-03,
  better than the old 0.34 %) but 3.3 % on the crowded, hazy dev clip (old 1.2 %).
- **A clean test needs** visible bottom in a second recording and day (another bottom, not a scraping roll), a few
  hundred seconds of it, and masks there; 94 s in one recording supports a sheet and a within-recording check only.
- **Bottom contact as a window flag** (depth near its local maximum and flat + dark windows + sediment) would mark
  00017 296-842 s without any picture model; it is not measured here. For the user's purpose (keep the stretch,
  label it, keep it out of 3D / ID) it is the cheaper and safer flag; the pixel class is for masks.

### Which stages read which head; what switching costs

- The scan's picture-class stage (`pipeline.py`, picture classes on scanned window frames) loads
  `patchseg.SEAFLOOR_CLF` = `heads/head_all.pt` and caches per-frame class shares under
  `scan/<deployment>/cache/pclass/`; its `seafloor` column is empty since the withdrawal, `dark` uses the grey level
  and `sand` the frame embedding (neither depends on the head). Switching to `head_v2_all.pt` means recomputing the
  cached frame classes: 31,546 frames (2025-02-03) + 10,736 (2025-01-02) = 42,282 frames, about 40 GPU minutes at
  ~55 ms per 1280 px frame plus decoding (the 4K recordings at 3.5-5x real time on 8 threads).
- The tracker's patch pass (`trackstage.patch_pass`) reads `patchseg/clf_w1280_h0_all.pt` (WP5b's head without a
  seafloor class), not `head_all.pt`: a switch does not touch it.

### Sand

Not fitted. `sand_candidates_best_01.jpg` (22 sediment frames of 00017 + 2 frames where the old sand label is wrong:
411 s is the bottom, 660 s is the back in haze) for the user to confirm; frames whose cloud the old label missed
(327-331, 422-440, 449-459, 510-523, 610-614 s) are marked in the caption.

Files: `wp12.py` (frames, grids, heads, frames2), `wp12_score.py`, `wp12_sheets.py`, `wp12_prompts.json`,
`results/wp12_pixels.json`, `results/wp12_windows.json`; cache `underwater_cache/seafloor/wp12/` (frames, masks,
mask_preds.npz, windows_00017.parquet), `frames_v2/`, `heads/head_v2_*.pt`. Sheets: `seafloor_masks_01..02`,
`seafloor_heldout_found_01..02`, `seafloor_heldout_missed_01` (E1), `seafloor_false_01`, `sand_candidates_best_01`.

## WP12 step B (2026-11-23): sand ("view blocked by stirred-up sediment") refitted on user-confirmed sediment; sensors at the bottom

**What it rests on.** The user confirmed the 22 sediment frames of `sand_candidates_best_01.jpg` ("Yes those are the
sediment frames of dragging along seafloor."); the two frames marked there as old-label errors (411 s bottom, 660 s
back in haze) count as not-sediment. All in ONE recording (2025-01-02 00017), one day, one bottom; windows next to a
training episode are near copies. Episodes (WP12 `sediment` spans holding confirmed frames): S1 327-336 (9 windows),
S2 422-443 + 449-459 (31), S3 522-527 + 533-536 + 540-557 (25), S4 610-622 s (12): 77 sediment windows. Not-sediment:
`none` + `visible` (user-confirmed seafloor) windows of 00017: 920. Left out: `unsure` (black / murk), dark windows,
breaths, 519-522 s, and the AI-only sediment spans 736-765 and 874-898 s (scored apart).

**Model** (`sand_redo.py`, CPU only, cached embeddings): frame embedding (mean DINOv3 patch feature of the frame and
its quarters), window mean; logistic regression C 0.001, balanced. Leave one episode out (5 s buffer); threshold per
fold from an inner leave-one-episode-out on its training episodes (chosen: A 0.05 in every fold; B 0.05-0.15).
Variant A fits on 00017 only; variant B adds every on-whale window of 00010, 00011, 00012, 00023, 00024 as
not-sediment, and a recording's false calls come from a fit without it. 2025-02-03 and the dev clip are never in a
fit. Saved, not switched: `underwater_cache/seafloor/sand_v2_A.npz`, `sand_v2_B.npz` (`patchseg.SAND_CLF` unchanged).

### Held-out episodes and false calls inside 00017 (folds 1-4)

| | variant A | variant B |
|---|---|---|
| S1 found | 4 / 9 | 5 / 9 |
| S2 found | 19 / 31 | 19 / 31 |
| S3 found | 25 / 25 | 25 / 25 |
| S4 found | 11 / 12 | 9 / 12 |
| all held-out sediment | 59 / 77 | 58 / 77 |
| 00017 none + visible called sand | 77-107 of 912-920 | 70-96 of 912-920 |
| of which user-confirmed seafloor windows called sand | **49-64 of 81-89** | **50-66 of 81-89** |
| AI-only sediment 736-765, 874-898 (not fitted) | 29-42 of 53 | 29-41 of 53 |
| unsure murk, not black | 102-114 of 104-114 | 101-112 of 104-114 |

The label does not separate sediment from the visible bottom: both happen in the same seconds of the scraping roll.
It does call the murk sand (that murk is green-turbid water inside the cloud).

### False calls elsewhere (fit on all episodes; variant B: a fit without the recording scored)

| windows (underwater, not dark) | variant A | variant B | old sand model (`sand_all.npz`; it had 2025-02-03 and dev as negatives: in-sample there) |
|---|---|---|---|
| 2025-01-02 00010 (616) | 17 | 0 | 0 |
| 2025-01-02 00011 (1,499) | 67 | 1 (an air frame the blocked-spans list misses) | 0 |
| 2025-01-02 00012 (220) | 2 | 0 | 0 |
| 2025-01-02 00023 (1,297) | 91 | 3 | 0 |
| 00023 back with haze 0-306 s (291) | 5 | 0 | 0 |
| 2025-01-02 00024 (386) | 108 | 102 (1 of 272 on-whale; 101 after the tag came off: floating / in-air states) | 1 |
| deep > 30 m: 00010 / 00011 / 00023 (95 / 519 / 480) | 1 / 29 / 34 | 0 / 0 / 3 | 0 |
| within 10 s after a lens-out: 00010 / 00011 / 00012 / 00023 / 00024 (47 / 76 / 10 / 126 / 52) | 4 / 9 / 0 / 2 / 0 | 0 | 0 |
| **2025-02-03 00006 dense (1,736)** | **1,373** | **333** | 0 |
| 00006 deep > 30 m (947) | 908 | 235 | 0 |
| 00006 within 10 s after a lens-out (120) | 56 | 5 | 0 |
| **2025-02-03 labelled frames (821)** | **739** | **508** | 0 |
| dev clip (91) | 20 | 0 | 0 |

Sheet `review_sheets/underwater/sand_calls_01.jpg` (16 calls of variant B, plain frame | decision): 00024's calls are
empty blue water around the floating tag, 00023's a hazy band beyond the back, 2025-02-03's empty green-blue deep
water. No find: every call outside 00017 looks like a false alarm to me (the user decides).

Old sand label (`blocked_spans.csv` kind sand) against the new truth in 00017: 42 of 77 sediment windows labelled sand,
45 of 920 not-sediment windows labelled sand.

### Sensors in 00017 (50 Hz tag data, scan sensor time per window; medians)

| windows | n | depth (m) | below recording max (51.2 m) | depth sd over 21 s (m) | roll (deg; normal -170) | pitch (deg) | ODBA (m/s2) | light |
|---|---|---|---|---|---|---|---|---|
| sediment, user-confirmed | 77 | 49.1 | 2.1 | 0.44 | -3.7 | -55.0 | 0.25 | 846 |
| seafloor visible, user-confirmed | 92 | 48.8 | 2.4 | 0.43 | -5.6 | -45.9 | 0.24 | 3,890 |
| black / murk (unsure) | 208 | 49.8 | 1.5 | 0.27 | 1.6 | -0.2 | 0.21 | 0 |
| unsure, not black | 115 | 49.1 | 2.1 | 0.39 | -8.8 | 11.4 | 0.26 | 193 |
| sediment, AI only | 53 | 48.9 | 2.3 | 0.17 | -45.7 | 2.1 | 0.08 | 69 |
| resting, back with haze (842-1390) | 524 | 47.5 | 3.8 | 0.04 | -174.6 | -5.2 | 0.02 | 15,792 |
| swimming (0-292, 644-673) | 307 | 10.4 | 40.8 | 0.84 | -123.5 | -0.6 | 0.12 | 37,177 |
| breath (air) | 14 | -0.1 | 51.3 | 0.92 | -166.5 | -1.1 | 0.47 | 37,177 |

The scraping episodes (seafloor, sediment, black lens, murk) share: deep (within about 2.5 m of the recording's
maximum), the tag rolled about 165 deg from its normal attitude (the whale rolled over), head-down pitch while the
bottom is visible, active (ODBA 0.2-0.26 against 0.02 when resting), dark. Resting on the bottom is deep but flat,
not rolled and still. A trial flag, set by me after seeing this table (exploratory, not a result): roll more than 90
deg from the recording's median AND within 5 m of its maximum depth AND deeper than 10 m: in 00017 it marks 53 / 77
sediment, 55 / 92 seafloor, 190 / 208 black, 95 / 115 murk windows, against 7 / 524 resting and 29 / 307 swimming
(644-673 s, inside the bottom period); elsewhere it fires on 0 of 3,886 on-whale underwater windows of the other five
2025-01-02 recordings and 0 of 15,362 of 2025-02-03 (pieces 00004-00012).

### Verdict

- **The refitted picture `sand` label is not usable as a scan label**: inside 00017 it finds 58-59 of 77 held-out
  sediment windows but calls 55-75 % of the confirmed seafloor windows sand (it learns "bottom contact", not
  "sediment"), and it calls 333-1,373 of 1,736 windows of 2025-02-03 00006 and 508-739 of 821 labelled 2025-02-03
  windows sand, all empty deep water. Adding the other 2025-01-02 recordings as negatives cleans that deployment, not
  the other one.
- **The sensors carry the event**: deep + rolled over + active separates the scraping episodes from resting and
  swimming in 00017 and fires nowhere else on either deployment. A per-window `bottom contact` flag from depth and
  attitude (picture classes only as a description inside it) would serve the user's purpose (keep, label, keep out of
  3D / ID) better than a picture class. It rests on one event in one recording; the thresholds are mine, set after
  looking.
- **A clean test needs** a second bottom-contact event on another day or tag (another bottom, sediment seen without
  the scraping roll), and a deployment with green water and sediment, to see whether colour alone is what the picture
  model learnt.

Files: `sand_redo.py`, `sand_sheet.py`, `results/sand_redo.json`; cache `underwater_cache/seafloor/sand_redo_*.parquet`,
`sand_v2_{A,B}.npz`; sheet `sand_calls_01.jpg`. No GPU used.
