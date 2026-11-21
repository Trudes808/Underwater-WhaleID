# 03_water_state (WP4): underwater / crossing / air from the picture, per 1 s window

Question: can the video alone tell that the camera is under water, and find when it surfaces, on a deployment it has
not seen; and what does the picture add when the tag's depth log exists? Unit of decision: 1 s windows of video
time (window k = [k, k+1) s), using the frames of the window and its neighbours (user decision, 2026-11-20).
All numbers are one run. Run on 2026-11-20 by an AI agent (WP4).

## Data, truth and protocol

| set | videos | seconds | truth | events | used for |
|---|---|---|---|---|---|
| fit | 2025-02-03 `.mov` pieces 00004, 00005, 00006 | 5,344 | sensor | 57 | fitting classifiers; tuning the window step on leave-one-piece-out (LOPO) scores |
| check | pieces 00007, 00008, 00009 (same deployment, unseen pieces) | 5,342 | sensor | 54 | scoring only |
| dev clip | `2025-12-20 CG.m4v` (other deployment, Apple export, no log) | 95 | AI by-eye labels (WP3: 191 frames, 4 events) | 4 | scoring only |
| Zenodo | public CATS deployment mn200312-58 (Gough and Cade, CC0), videos 00011 and 00013 (Antarctic, 2020) | 1,184 | depth, by-eye-corrected time base (below) | 40 | scoring only |
| WP3 events | AI by-eye exits / entries in two 180 s windows per piece of 00004-00009 | 2,161 | by eye, to 1 frame | 17 fit + 24 check | independent check of the sensor truth and of event timing |

Sensor truth (`uwid.eval_state.sensor_timeline`): deck and drone flight = air; on the whale the lens is in air from
`start_t + 0.3 s` to `end_t - 0.4 s` of each surfacing (WP1's by-eye lags); after the attaching drop the lens stays
in air until the depth first passes the surfacing threshold (minus 0.4 s; the tag lands on a whale at the surface:
checked by eye, lens under at 93.2-93.4 s vs truth 93.05 s). `floating` (tag loose in the sea) would be its own state,
excluded from the scores (the lens bobs at the waterline); none of the pieces used here contains floating time.
Window truth: `crossing` if an exit or entry falls in the window, else the state that fills it. Windows with an event
within 0.3 s of an edge are flagged ambiguous and also scored with them left out. Frame truth: frames within 0.3 s of
an event are `crossing`, reported apart. Events: matched one-to-one, same kind, within 0.5 s (and 1.0 s).

The sensor truth against WP3's by-eye events (section 3): all 41 by-eye events in the labelled windows are within
0.41 s of a sensor event, median |difference| 0.09-0.14 s. The truth is good to about 0.1 s typical, 0.4 s worst.

Zenodo time base: the toolbox's `movieTimes.mat` puts a frame at `vidDN + pts / 86400` with `vidDN` = the file-name
time; depth `p` (10 Hz, calibrated) on `DN`. Port out of the water = `p < 0.10 m` (air peak centred at -0.01 m),
re-immersions under 1 s merged, exposures under 1 s dropped. By eye on 6 events (3 exits, 3 entries, frame strips at
0.1 s, AI by-eye): the lens is fully out 1.1-1.3 s after the depth crossing and back under 0.5-0.7 s after it, the same
for both videos. That is a common offset of about +0.9 s (video late against the sensor time; the toolbox notes the
camera clock can be 1-2 s off) plus the lens/port geometry. The truth uses `out_lag = 1.15 s`, `in_lead = -0.65 s`.
Confidence: the 6 events agree to about +-0.15 s, so the alignment is good to about 0.2 s for these two videos; it
was set by eye on the test videos themselves (truth construction, no method was tuned on it). The two videos are
dominated by a whale logging at the surface: 786 of 1,182 windows are air by depth, with waves washing over the lens
(see failures: the depth threshold cannot see wave wash, so part of the "air" truth is wrong).

Frames: decoded at 5 per second, 448x256, the bottom 24 rows (burned-in clock) left out of the colour numbers and
blacked out for the models. Methods (frame score = logit of air):
- (a0) mean HSV saturation alone, logistic on 1 number (WP1's cue);
- (a) 18 colour / brightness numbers (`uwid.waterstate.color_features`) + logistic regression;
- (b) CLIP ViT-L/14 zero-shot, 6 air and 6 underwater prompts (`prompts.json`, written before any score), score =
  logsumexp(air) - logsumexp(underwater) at CLIP's logit scale; nothing fitted except the window step;
- (b2) CLIP image embedding + logistic regression (a linear probe; not asked for, free once (b) ran);
- (c) DINOv3 ViT-B/16 CLS + mean patch token (1,536 numbers) + logistic regression.
Logistic regressions: standardised features, class weights balanced, C from {0.01, 0.1, 1} by LOPO average precision
on the fit pieces, frames within 0.3 s of an event left out of training, final model refitted on all three pieces.
Window step (`uwid.waterstate.binarize`), each tuned on the fit pieces' LOPO scores by window recall + precision +
event F1: `none` = threshold per frame; `pool` = mean over k = 3 / 5 / 9 frames (0.6-1.8 s, centred) then threshold;
`hysteresis` = pool, then two thresholds; `2-state chain` = pool, then the best air / underwater sequence with a
penalty per switch (Viterbi). Events = flips of the per-frame state (time = midpoint of the two frames); a window
holding an event is `crossing`.

**Two runs, both reported.** v1 took the best-scoring step setting on the fit pieces. Many settings tie there
(everything is near perfect within the deployment), and v1's chain for (b2) and (c) used penalty 30, which on the
dev clip erased a 1.8 s air spell. v2 (the default, and the shipped parameters) takes the least smoothing within 0.02
of the best objective. v2 was decided after seeing v1 on the dev clip, so v2's dev-clip numbers are not a clean
test. The Zenodo features did not exist yet when v2 was fixed; both v1 and v2 were then run on it once.

## 1. Answer in one table (window step = 2-state chain, v2; events within 0.5 s / 1.0 s)

| set | method | underwater windows R / P | crossing R / P | air R / P | events 0.5 s | 1.0 s | false events / h |
|---|---|---|---|---|---|---|---|
| check (same deployment) | (a) colour | 5229/5232 / 5229/5230 | 50/54 / 50/54 | 54/55 / 54/57 | 53/54 | 54/54 | 0.7 |
| check | (b2) CLIP probe | 5231/5232 / 5231/5234 | 51/54 / 51/52 | 54/55 / 54/55 | 52/54 | 52/54 | 0.0 |
| check | (c) DINOv3 | 5228/5232 / 5228/5230 | 48/54 / 48/54 | 53/55 / 53/57 | 53/54 | 54/54 | 0.7 |
| check | (b) CLIP zero-shot | 5090/5232 / 5090/5118 | 23/54 / 23/92 | 31/55 / 31/131 | 31/54 | 46/54 | 41.1 |
| dev clip (95 s) | (a) colour | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 2/4 | 4/4 | 2 false |
| dev clip | (b2) CLIP probe | 88/89 / 88/89 | 2/4 / 2/3 | 2/2 / 2/3 | 1/4 | 3/4 | 2 false |
| dev clip | (c) DINOv3 | 84/89 / 84/84 | 2/4 / 2/5 | 2/2 / 2/6 | 2/4 | 3/4 | 4 false |
| dev clip | (b) CLIP zero-shot | 89/89 / 89/89 | 4/4 / 4/4 | 2/2 / 2/2 | 3/4 | 4/4 | 1 false |
| Zenodo (1,184 s) | (a) colour | 229/356 / 229/981 | 17/40 / 17/79 | 15/786 / 15/122 | 1/40 | 4/40 | 246 |
| Zenodo | (b2) CLIP probe | 213/356 / 213/227 | 22/40 / 22/40 | 768/786 / 768/915 | 26/40 | 30/40 | 43 |
| Zenodo | (c) DINOv3 | 350/356 / 350/366 | 31/40 / 31/82 | 731/786 / 731/734 | 30/40 | 38/40 | 176 |
| Zenodo | (c) DINOv3, v1 chain (penalty 30) | 346/356 / 346/347 | 24/40 / 24/26 | 785/786 / 785/809 | 21/40 | 26/40 | 15 |
| Zenodo | (b) CLIP zero-shot | 345/356 / 345/906 | 17/40 / 17/68 | 197/786 / 197/208 | 17/40 | 26/40 | 155 |

Best on the dev clip with v2: (b2) with hysteresis: 89/89, 4/4, 2/2 windows, 4/4 events within 0.5 s, 0 false.
Saturation alone (a0) calls every Zenodo window air (Antarctic water is grey-green: 0/40 events).

## 2. What the window step buys

Same frame scores, four steps (full tables in appendix A). False events (and underwater-window recall) go:

| set | method | per frame (none) | pool 1 s | hysteresis | 2-state chain |
|---|---|---|---|---|---|
| check | (c) DINOv3 | 9 false | 1 | 2 | 1 |
| check | (b) CLIP zero-shot | 422 false | 94 | 63 | 61 |
| check | (a0) saturation | 32 false | 18 | 4 | 7 |
| dev clip | (c) DINOv3 | 18 false, U 76/89 | 6, 83/89 | 4, 84/89 | 4, 84/89 |
| dev clip | (b) CLIP zero-shot | 29 false, U 76/89 | 3, 88/89 | 1, 88/89 | 1, 89/89 |
| Zenodo | (c) DINOv3 | 263 false, U 332/356, A 694/786 | 111, 348, 712 | 87, 351, 708 | 58, 350, 731 (v1 chain: 5, 346, 785) |
| Zenodo | (b2) CLIP probe | 218 false | 115 | 74 | 14 |

Per frame the good classifiers are already near perfect inside the deployment (DINOv3 on check: air recall 305/305,
precision 305/310); the window step mainly removes isolated flips, i.e. false events, which per frame run at
hundreds per hour on a new deployment. The turtle-camera study's pattern (high recall, low precision per frame,
smoothing needed) shows up here for CLIP zero-shot (per-frame air precision 20-25 % on check) and on the new
deployments. Within the deployment the four steps differ by at most 2-3 events; across deployments the strength of
the chain is a trade: penalty 1 (v2) finds more events on Zenodo (30/40 vs 21/40 within 0.5 s) but leaves 58 false
events; penalty 30 (v1) gives cleaner windows (air precision 785/809, 5 false events) and misses short exposures.
Crossing windows: an event found within 0.5 s almost always lands in the right window; crossing recall tracks event
recall (check 48-51/54 for the trained methods).

## 3. Independent check against WP3's AI by-eye events (2025-02-03, two 180 s windows per piece)

Events inside the labelled windows only (fit pieces: 17 events, LOPO picture scores; check pieces: 24 events).
`minus_byeye` = source time minus by-eye time; median [min, max] over the matches within 1.0 s.

| set | source | tol_s | found | false | exit: median [min, max] (s) | entry: median [min, max] (s) | median abs (s) | labelled s (all pieces) |
|---|---|---|---|---|---|---|---|---|
| fit (LOPO scores) | sensor truth (depth + lags) | 0.5 | 17/17 | 0 | +0.09 [-0.18, +0.29] | -0.03 [-0.23, +0.41] | 0.14 | 1081 |
| fit (LOPO scores) | raw depth crossing | 0.5 | 13/17 | 4 | -0.21 [-0.48, -0.01] | +0.37 [+0.17, +0.81] | 0.30 | 1081 |
| fit (LOPO scores) | picture: sat | 0.5 | 17/17 | 0 | -0.16 [-0.33, -0.03] | +0.07 [-0.20, +0.20] | 0.13 | 1081 |
| fit (LOPO scores) | picture: color | 0.5 | 17/17 | 1 | +0.08 [-0.30, +0.20] | +0.07 [-0.20, +0.20] | 0.13 | 1081 |
| fit (LOPO scores) | picture: clip0 | 0.5 | 10/17 | 15 | +0.37 [-0.53, +0.90] | -0.30 [-0.60, +0.47] | 0.40 | 1081 |
| fit (LOPO scores) | picture: clipLR | 0.5 | 17/17 | 0 | +0.07 [-0.03, +0.17] | +0.07 [-0.20, +0.20] | 0.07 | 1081 |
| fit (LOPO scores) | picture: dino | 0.5 | 17/17 | 0 | +0.07 [-0.13, +0.17] | +0.00 [-0.20, +0.20] | 0.10 | 1081 |
| check | sensor truth (depth + lags) | 0.5 | 24/24 | 0 | +0.07 [-0.11, +0.20] | +0.12 [-0.02, +0.34] | 0.09 | 1080 |
| check | raw depth crossing | 0.5 | 17/24 | 7 | -0.23 [-0.41, -0.10] | +0.52 [+0.38, +0.74] | 0.39 | 1080 |
| check | picture: sat | 0.5 | 22/24 | 2 | -0.28 [-0.77, -0.17] | +0.25 [-0.10, +0.40] | 0.27 | 1080 |
| check | picture: color | 0.5 | 24/24 | 0 | +0.02 [-0.23, +0.17] | +0.15 [-0.30, +0.33] | 0.15 | 1080 |
| check | picture: clip0 | 0.5 | 13/24 | 15 | +0.17 [-0.77, +0.80] | -0.07 [-0.67, +0.73] | 0.30 | 1080 |
| check | picture: clipLR | 0.5 | 24/24 | 0 | -0.02 [-0.20, +0.13] | +0.15 [+0.07, +0.40] | 0.13 | 1080 |
| check | picture: dino | 0.5 | 24/24 | 0 | +0.03 [-0.03, +0.27] | +0.22 [-0.03, +0.43] | 0.12 | 1080 |

Reading: the sensor truth (depth + fixed lags) and the trained picture methods ((a), (b2), (c)) are equally exact
against the by-eye lens moment: median |error| 0.07-0.15 s, all within 0.45 s. Raw depth crossings without the lags
are 0.2 s early at exits and 0.4-0.5 s late at entries. The picture methods put entries 0.15-0.25 s late on the check
pieces (they wait for the froth to clear).

## 4. Picture events against the raw depth crossings (what the picture adds when a log exists)

Picture event (chain, v2) minus the nearest raw depth crossing within 1.5 s, no lag applied. On Zenodo the depth
crossings use the file-name time base with no correction.

| set | method | kind | n | median | q10 | q90 | IQR |
|---|---|---|---|---|---|---|---|
| Zenodo | clip0 | entry | 9 | 1.13 | 0.82 | 1.3 | 0.29 |
| Zenodo | clip0 | exit | 11 | 0.99 | 0.77 | 1.26 | 0.27 |
| Zenodo | clipLR | entry | 16 | 0.9 | -0.43 | 1.1 | 0.75 |
| Zenodo | clipLR | exit | 15 | 1.09 | 0.81 | 1.4 | 0.34 |
| Zenodo | color | exit | 5 | 0.54 | -0.14 | 0.67 | 0.27 |
| Zenodo | dino | entry | 20 | 0.43 | -0.25 | 0.66 | 0.34 |
| Zenodo | dino | exit | 15 | 0.99 | 0.06 | 1.16 | 0.5 |
| check 2025-02-03 | clip0 | entry | 21 | -0.85 | -1.24 | 0.15 | 0.69 |
| check 2025-02-03 | clip0 | exit | 25 | 0.45 | -0.01 | 0.85 | 0.32 |
| check 2025-02-03 | clipLR | entry | 26 | -0.36 | -0.48 | -0.26 | 0.17 |
| check 2025-02-03 | clipLR | exit | 26 | 0.23 | 0.02 | 0.34 | 0.12 |
| check 2025-02-03 | color | entry | 27 | -0.38 | -0.59 | -0.25 | 0.23 |
| check 2025-02-03 | color | exit | 27 | 0.24 | -0.08 | 0.39 | 0.18 |
| check 2025-02-03 | dino | entry | 27 | -0.36 | -0.5 | -0.12 | 0.2 |
| check 2025-02-03 | dino | exit | 27 | 0.24 | 0.12 | 0.45 | 0.25 |

- 2025-02-03 (check, 27 surfacings): DINOv3 exits +0.24 s (IQR 0.25), entries -0.36 s (IQR 0.20), the same as WP1's
  by-eye lags (+0.3 / -0.4). With a log and a known lag, the picture moves the event by 0.1-0.2 s at most: for timing
  it adds little that the depth + lag does not give.
- Zenodo: DINOv3 exits +0.99 s (n = 15), entries +0.43 s (n = 20); the by-eye check gave +1.15 / +0.65 s. The picture
  alone recovers the video-to-sensor offset of this deployment (about +0.9 s, which the file-name time base gets
  wrong) to about 0.2 s, without anyone looking at frames. That is the clearest use of the picture when a log exists:
  checking or fixing the sync per file.
- The picture sees what the depth threshold cannot: on Zenodo, of the 55 truth-air windows where DINOv3 disagrees,
  the middle frame shows the lens UNDER water (tan back, blue water: a wave washing over the logging whale) in about
  32, clearly in air in about 17, unclear in 6 (AI by-eye, one frame per window,
  `waterstate_zenodo_dino_disagreements.jpg`). The depth port at < 0.10 m does not register waves that cover the
  lens for 0.5-2 s. So part of DINOv3's 58 "false" Zenodo events are real lens submersions, and its Zenodo air
  recall (731/786) is a lower bound. Fixing that truth would need by-eye labels of the Zenodo videos (not made).

## 5. Failures (review sheets in `/home/sat3737/NOAA_whales/review_sheets/underwater/`)

- Called air while under water (`waterstate_false_air.jpg`): the bright surface seen from just below (Snell window,
  sun, the brown total reflection of the back: 00007 352 s, 00008 1234 / 1257 / 1715 s; DINOv3 and colour call these
  crossing); exhalation bubble clouds at the surface on the dev clip (11-13 s: DINOv3 air, colour and CLIP probe
  crossing); on Zenodo the near-surface frames with white foam above the tan back, and dark deep-water frames (the
  CLIP probe calls black frames air: Z00011 570-583 s, Z00013 492-586 s). CLIP zero-shot calls clear underwater
  frames with whale silhouettes against blue "air" (00007 1218-1772 s): 95 windows, its main failure.
- Called underwater while in air (`waterstate_missed_air.jpg`): on 2025-02-03 almost nothing (1-2 windows per method;
  00008 1236 s air through a water film). On Zenodo the long logging bouts: the grey back in air with waves breaking
  over it, and windows where the lens is really under water (see 6). Colour and CLIP zero-shot fail here wholesale
  (colour 15/786 air windows): the Antarctic water and the diatom-tan back have nothing like the Hawaiian colours.
- Drone flight and drop (`waterstate_drop.jpg`, `waterstate_scores_00004.png`, LOPO scores): all methods call the
  whole 91 s flight air with large margins (DINOv3 logit +10, colour +6, CLIP zero-shot +2); the drop splash at
  92.0-93.0 s is still air, the first dark frame after impact (93.5 s) underwater for every method; entry found at
  93.1-93.5 s against the truth 93.05 s. Only one drop in the data: n = 1.
- Crossing windows: per frame, 90-104 of 171 fit-piece crossing frames (within 0.3 s of an event) are called air by
  the trained methods, saturation alone 138 / 171 (it fires early on the bright surface from below, as WP1 saw).
  Missed crossing windows on check are events found 0.5-1.0 s off or two events in adjacent windows; on Zenodo they
  are mostly wave-wash flips. `waterstate_crossings_{check,dev,zenodo}.jpg` show frames every 0.4 s around events
  with each method's call; `waterstate_crossing_errors.jpg` the crossing windows each method got wrong.
- Short air spells: v1's strong chain erased the dev clip's 1.8 s exposure (CLIP probe 0/4 events); v2 does not.

## 6. Cost (RTX 4000 Ada shared with other jobs, 8 CPU threads, 5 frames/s sampled)

| stage | frames/s (sampled frames) | per hour of video (18,000 frames) |
|---|---|---|
| decode 1080p HEVC `.mov` (all frames decoded, 5/s kept, scaled to 448x256), CPU | 297 | 61 s |
| colour numbers, CPU, 1 thread | 404 | 45 s |
| CLIP ViT-L/14 at 224x224, fp16, batch 48, GPU (~1 GB) | 221 | 81 s |
| DINOv3 ViT-B/16 at 448x256, fp16, batch 48, GPU (< 1 GB) | 247 | 73 s |
| logistic + window step, CPU | > 100,000 | < 1 s |

Measured on the six 2025-02-03 pieces (53,431 frames); Zenodo MPEG-4 decodes about twice as fast. One hour of
video: colour method about 2 min CPU; DINOv3 method about 2.3 min (decode + GPU), so about 0.02 GPU hours per hour of
video. The GPU lock, not the compute, set the wall time today (waits of 5-30 min per job).

## 7. Shipped predictor

`uwid.waterstate.predict_video(path, params=None, device=None, method=None, step=None)` decodes at 5 frames/s and
returns `(windows, events, frames)`: one row per 1 s window (state underwater / crossing / air, score = mean
probability of air, events inside), the exit / entry list, and per-frame scores. Default model `dino`
(DINOv3 + logistic, v2 chain k = 3, bias 0.5, penalty 1); `method="color"` runs on the CPU only but did not transfer
to other deployments; `step="step_v1_chain"` gives the stronger chain (fewer false events, fewer events found).
Parameters: `src/uwid/waterstate_params.json` (49 KB: scaler, 1,536 + 18 logistic weights, steps; no model
weights; the DINOv3 weights are read from `/home/sat3737/dinov3/weights/`). Checked on the dev clip against run.py:
per-frame scores equal to 0.0035 logit (correlation 0.99999997), the same per-frame calls; 95 s clip in 8.0 s
(DINOv3, incl. model load) and 3.7 s (colour). On the dev clip the default DINOv3 model gives entry 1.10 s (truth
0.80), a false exit / entry pair at 1.70 / 2.10 s (entry froth), exit 11.91 s (truth 14.58: the bubble cloud at
11-13 s), entry 16.71 s (truth 16.41), exit 93.11 s (truth 93.61); colour gives 0.90 / 13.71 / 16.71 / 92.71 s.

## 8. What did not work, what was not done

- CLIP zero-shot: the worst method everywhere it was not tuned (check: 31/54 events, 41 false / h; per-frame air
  precision 20 % at its natural threshold). It was the only method that did well on the dev clip without being
  trained on it, and failed on Zenodo (air recall 197/786).
- Colour statistics: near perfect within the deployment, useless on Zenodo (1/40 events). Saturation alone likewise.
- The CLIP probe is as good as DINOv3 within the deployment and best on the dev clip, but loses most Zenodo
  underwater windows (213/356) and calls black frames air.
- Not done: floating-tag windows (none in pieces 00004-00009; pieces 00012-00013 hold 26 min of it), failed drops
  and deck scenes (pieces 00001-00003 not on disk when this ran), night / LED, by-eye labels for the Zenodo videos
  (needed to separate wave wash from method error), a combined model (e.g. DINOv3 + CLIP probe) since choosing it
  would use test results, any re-fit including test videos.
- Caveats: one deployment for fitting (one tag, one day, Hawaii); the dev clip has 4 events and 2 air windows;
  Zenodo is 2 videos of one Antarctic deployment dominated by surface logging, with its time base corrected by eye
  on 6 of its own events.

## Files

Scripts (`experiments/03_water_state/`): `extract.py` (features, GPU), `clip_text.py`, `prompts.json`,
`zenodo_truth.py` (run with `python -I`), `run.py` (fit, tune, score; `WS_TIE_RULE=argmax WS_RUN_TAG=_v1` for v1),
`make_tables.py`, `wp3_check.py`, `picture_vs_depth.py`, `sheets.py` (run with `python -I`). Library:
`src/uwid/eval_state.py` (truth, scorers), `src/uwid/waterstate.py` (features, window step, predictor),
`src/uwid/waterstate_params.json`. Cache: `underwater_cache/waterstate/` (feats/*.npz 0.27 GB, pred/, results.json,
results_v1.json, linear_params*.json, window_errors.csv, wp3_event_check.csv, picture_vs_depth.csv,
zenodo_truth/), downloads `underwater_cache/public/zenodo_4902325/` (1.2 GB: 2 videos, prh, movieTimes, Info, txt).
Rebuild: `extract.py VIDEO` per video under the GPU lock, then `run.py`, `make_tables.py`, `wp3_check.py`,
`picture_vs_depth.py`, `sheets.py`.

## Appendix A. All tables, run v2 (default)


### A. fit LOPO (00004-00006) (5344 s of video)

Window states: `correct/truth (recall) / correct/predicted (precision)`; events matched within 0.5 s (and 1.0 s), false events per hour, median |timing error| of the 0.5 s matches.

| method | window step | underwater R / P | crossing R / P | air R / P | events found 0.5 s | 1.0 s | false events | median abs err (s) |
|---|---|---|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | none (frame threshold) | 5115/5140 / 5115/5115 | 35/57 / 35/59 | 146/146 / 146/169 | 47/57 | 56/57 | 17 (11.5/h) | 0.21 |
|  | pool 1 s | 5134/5140 / 5134/5149 | 36/57 / 36/57 | 131/146 / 131/137 | 47/57 | 57/57 | 10 (6.7/h) | 0.24 |
|  | pool + hysteresis | 5128/5140 / 5128/5129 | 45/57 / 45/58 | 145/146 / 145/156 | 54/57 | 56/57 | 4 (2.7/h) | 0.16 |
|  | pool + 2-state chain | 5120/5140 / 5120/5120 | 38/57 / 38/58 | 146/146 / 146/165 | 50/57 | 57/57 | 8 (5.4/h) | 0.22 |
| (a) colour stats, 18 numbers | none (frame threshold) | 5131/5140 / 5131/5131 | 49/57 / 49/59 | 145/146 / 145/153 | 57/57 | 57/57 | 5 (3.4/h) | 0.15 |
|  | pool 1 s | 5134/5140 / 5134/5134 | 51/57 / 51/58 | 145/146 / 145/151 | 57/57 | 57/57 | 1 (0.7/h) | 0.13 |
|  | pool + hysteresis | 5134/5140 / 5134/5134 | 51/57 / 51/58 | 145/146 / 145/151 | 57/57 | 57/57 | 1 (0.7/h) | 0.16 |
|  | pool + 2-state chain | 5134/5140 / 5134/5134 | 51/57 / 51/58 | 145/146 / 145/151 | 57/57 | 57/57 | 1 (0.7/h) | 0.13 |
| (b) CLIP zero-shot | none (frame threshold) | 4677/5140 / 4677/4681 | 52/57 / 52/460 | 114/146 / 114/202 | 48/57 | 57/57 | 777 (523.4/h) | 0.26 |
|  | pool 1 s | 4883/5140 / 4883/4896 | 41/57 / 41/174 | 133/146 / 133/273 | 43/57 | 55/57 | 147 (99.0/h) | 0.17 |
|  | pool + hysteresis | 4937/5140 / 4937/4951 | 38/57 / 38/112 | 132/146 / 132/280 | 39/57 | 54/57 | 73 (49.2/h) | 0.23 |
|  | pool + 2-state chain | 4927/5140 / 4927/4946 | 33/57 / 33/106 | 125/146 / 125/291 | 39/57 | 53/57 | 67 (45.1/h) | 0.25 |
| (b2) CLIP + logistic | none (frame threshold) | 5132/5140 / 5132/5133 | 50/57 / 50/60 | 144/146 / 144/150 | 56/57 | 57/57 | 6 (4.0/h) | 0.16 |
|  | pool 1 s | 5132/5140 / 5132/5132 | 49/57 / 49/58 | 145/146 / 145/153 | 57/57 | 57/57 | 1 (0.7/h) | 0.16 |
|  | pool + hysteresis | 5136/5140 / 5136/5139 | 51/57 / 51/58 | 142/146 / 142/146 | 57/57 | 57/57 | 1 (0.7/h) | 0.14 |
|  | pool + 2-state chain | 5132/5140 / 5132/5132 | 49/57 / 49/57 | 146/146 / 146/154 | 57/57 | 57/57 | 0 (0.0/h) | 0.16 |
| (c) DINOv3 + logistic | none (frame threshold) | 5131/5140 / 5131/5131 | 49/57 / 49/59 | 145/146 / 145/153 | 57/57 | 57/57 | 4 (2.7/h) | 0.15 |
|  | pool 1 s | 5134/5140 / 5134/5134 | 51/57 / 51/57 | 146/146 / 146/152 | 57/57 | 57/57 | 0 (0.0/h) | 0.12 |
|  | pool + hysteresis | 5135/5140 / 5135/5135 | 52/57 / 52/57 | 146/146 / 146/151 | 57/57 | 57/57 | 0 (0.0/h) | 0.11 |
|  | pool + 2-state chain | 5134/5140 / 5134/5134 | 51/57 / 51/57 | 146/146 / 146/152 | 57/57 | 57/57 | 0 (0.0/h) | 0.12 |

Per frame (5 frames/s, frame threshold, no window step; frames within 0.3 s of a true event left out and counted apart):

| method | underwater R / P | air R / P | air recall % | air precision % | crossing frames called air |
|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | 25720/25759 / 25720/25720 | 790/790 / 790/829 | 100.0 | 95.3 | 138/171 |
| (a) colour stats, 18 numbers | 25756/25759 / 25756/25761 | 785/790 / 785/788 | 99.4 | 99.6 | 102/171 |
| (b) CLIP zero-shot | 24533/25759 / 24533/24590 | 733/790 / 733/1959 | 92.8 | 37.4 | 43/171 |
| (b) CLIP zero-shot, untuned threshold 0 | 23952/25759 / 23952/23979 | 763/790 / 763/2570 | 96.6 | 29.7 | |
| (b2) CLIP + logistic | 25757/25759 / 25757/25762 | 785/790 / 785/787 | 99.4 | 99.7 | 108/171 |
| (c) DINOv3 + logistic | 25758/25759 / 25758/25760 | 788/790 / 788/789 | 99.7 | 99.9 | 108/171 |

Window step `pool + 2-state chain`, windows with a true event within 0.3 s of their edge left out (truth timing +-0.3 s):

| method | underwater R / P | crossing R / P | air R / P |
|---|---|---|---|
| (a0) mean saturation, 1 number | 5118/5122 / 5118/5118 | 20/23 / 20/24 | 130/130 / 130/133 |
| (a) colour stats, 18 numbers | 5122/5122 / 5122/5122 | 23/23 / 23/24 | 129/130 / 129/129 |
| (b) CLIP zero-shot | 4910/5122 / 4910/4918 | 12/23 / 12/73 | 120/130 / 120/284 |
| (b2) CLIP + logistic | 5122/5122 / 5122/5122 | 23/23 / 23/23 | 130/130 / 130/130 |
| (c) DINOv3 + logistic | 5122/5122 / 5122/5122 | 23/23 / 23/23 | 130/130 / 130/130 |

### A. check (00007-00009) (5342 s of video)

Window states: `correct/truth (recall) / correct/predicted (precision)`; events matched within 0.5 s (and 1.0 s), false events per hour, median |timing error| of the 0.5 s matches.

| method | window step | underwater R / P | crossing R / P | air R / P | events found 0.5 s | 1.0 s | false events | median abs err (s) |
|---|---|---|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | none (frame threshold) | 5217/5232 / 5217/5217 | 44/54 / 44/61 | 53/55 / 53/63 | 43/54 | 54/54 | 32 (21.6/h) | 0.16 |
|  | pool 1 s | 5229/5232 / 5229/5250 | 33/54 / 33/52 | 36/55 / 36/39 | 40/54 | 52/54 | 18 (12.1/h) | 0.35 |
|  | pool + hysteresis | 5228/5232 / 5228/5241 | 45/54 / 45/50 | 47/55 / 47/50 | 47/54 | 48/54 | 4 (2.7/h) | 0.15 |
|  | pool + 2-state chain | 5225/5232 / 5225/5231 | 45/54 / 45/55 | 49/55 / 49/55 | 48/54 | 52/54 | 7 (4.7/h) | 0.15 |
| (a) colour stats, 18 numbers | none (frame threshold) | 5229/5232 / 5229/5229 | 51/54 / 51/54 | 55/55 / 55/58 | 51/54 | 54/54 | 3 (2.0/h) | 0.07 |
|  | pool 1 s | 5229/5232 / 5229/5230 | 50/54 / 50/54 | 54/55 / 54/57 | 53/54 | 54/54 | 1 (0.7/h) | 0.09 |
|  | pool + hysteresis | 5230/5232 / 5230/5232 | 50/54 / 50/54 | 53/55 / 53/55 | 52/54 | 54/54 | 2 (1.3/h) | 0.11 |
|  | pool + 2-state chain | 5229/5232 / 5229/5230 | 50/54 / 50/54 | 54/55 / 54/57 | 53/54 | 54/54 | 1 (0.7/h) | 0.09 |
| (b) CLIP zero-shot | none (frame threshold) | 4968/5232 / 4968/4977 | 45/54 / 45/282 | 20/55 / 20/82 | 44/54 | 54/54 | 422 (284.4/h) | 0.18 |
|  | pool 1 s | 5074/5232 / 5074/5089 | 36/54 / 36/123 | 40/55 / 40/129 | 40/54 | 50/54 | 94 (63.3/h) | 0.15 |
|  | pool + hysteresis | 5081/5232 / 5081/5100 | 25/54 / 25/95 | 37/55 / 37/146 | 33/54 | 48/54 | 63 (42.5/h) | 0.33 |
|  | pool + 2-state chain | 5090/5232 / 5090/5118 | 23/54 / 23/92 | 31/55 / 31/131 | 31/54 | 46/54 | 61 (41.1/h) | 0.25 |
| (b2) CLIP + logistic | none (frame threshold) | 5230/5232 / 5230/5231 | 52/54 / 52/55 | 54/55 / 54/55 | 54/54 | 54/54 | 2 (1.3/h) | 0.12 |
|  | pool 1 s | 5231/5232 / 5231/5231 | 53/54 / 53/54 | 55/55 / 55/56 | 54/54 | 54/54 | 0 (0.0/h) | 0.07 |
|  | pool + hysteresis | 5232/5232 / 5232/5235 | 51/54 / 51/54 | 52/55 / 52/52 | 54/54 | 54/54 | 0 (0.0/h) | 0.07 |
|  | pool + 2-state chain | 5231/5232 / 5231/5234 | 51/54 / 51/52 | 54/55 / 54/55 | 52/54 | 52/54 | 0 (0.0/h) | 0.07 |
| (c) DINOv3 + logistic | none (frame threshold) | 5227/5232 / 5227/5229 | 48/54 / 48/55 | 53/55 / 53/57 | 51/54 | 54/54 | 9 (6.1/h) | 0.11 |
|  | pool 1 s | 5228/5232 / 5228/5230 | 48/54 / 48/54 | 53/55 / 53/57 | 53/54 | 54/54 | 1 (0.7/h) | 0.13 |
|  | pool + hysteresis | 5230/5232 / 5230/5233 | 49/54 / 49/54 | 52/55 / 52/54 | 52/54 | 54/54 | 2 (1.3/h) | 0.11 |
|  | pool + 2-state chain | 5228/5232 / 5228/5230 | 48/54 / 48/54 | 53/55 / 53/57 | 53/54 | 54/54 | 1 (0.7/h) | 0.13 |

Per frame (5 frames/s, frame threshold, no window step; frames within 0.3 s of a true event left out and counted apart):

| method | underwater R / P | air R / P | air recall % | air precision % | crossing frames called air |
|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | 26209/26244 / 26209/26212 | 302/305 / 302/337 | 99.0 | 89.6 | 134/162 |
| (a) colour stats, 18 numbers | 26241/26244 / 26241/26241 | 305/305 / 305/308 | 100.0 | 99.0 | 96/162 |
| (b) CLIP zero-shot | 25502/26244 / 25502/25556 | 251/305 / 251/993 | 82.3 | 25.3 | 40/162 |
| (b) CLIP zero-shot, untuned threshold 0 | 25175/26244 / 25175/25209 | 271/305 / 271/1340 | 88.9 | 20.2 | |
| (b2) CLIP + logistic | 26243/26244 / 26243/26243 | 305/305 / 305/306 | 100.0 | 99.7 | 101/162 |
| (c) DINOv3 + logistic | 26239/26244 / 26239/26239 | 305/305 / 305/310 | 100.0 | 98.4 | 98/162 |

Window step `pool + 2-state chain`, windows with a true event within 0.3 s of their edge left out (truth timing +-0.3 s):

| method | underwater R / P | crossing R / P | air R / P |
|---|---|---|---|
| (a0) mean saturation, 1 number | 5221/5223 / 5221/5224 | 23/25 / 23/26 | 32/35 / 32/33 |
| (a) colour stats, 18 numbers | 5222/5223 / 5222/5222 | 24/25 / 24/25 | 35/35 / 35/36 |
| (b) CLIP zero-shot | 5083/5223 / 5083/5096 | 10/25 / 10/69 | 22/35 / 22/118 |
| (b2) CLIP + logistic | 5223/5223 / 5223/5226 | 23/25 / 23/23 | 34/35 / 34/34 |
| (c) DINOv3 + logistic | 5222/5223 / 5222/5222 | 24/25 / 24/25 | 35/35 / 35/36 |

### A. dev clip 2025-12-20 (cross-deployment) (95 s of video)

Window states: `correct/truth (recall) / correct/predicted (precision)`; events matched within 0.5 s (and 1.0 s), false events per hour, median |timing error| of the 0.5 s matches.

| method | window step | underwater R / P | crossing R / P | air R / P | events found 0.5 s | 1.0 s | false events | median abs err (s) |
|---|---|---|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | none (frame threshold) | 85/89 / 85/85 | 2/4 / 2/5 | 2/2 / 2/5 | 1/4 | 2/4 | 5 (188.8/h) | 0.10 |
|  | pool 1 s | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 2/4 | 2/4 | 2 (75.5/h) | 0.20 |
|  | pool + hysteresis | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 1/4 | 2/4 | 3 (113.3/h) | 0.10 |
|  | pool + 2-state chain | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 1/4 | 2/4 | 3 (113.3/h) | 0.10 |
| (a) colour stats, 18 numbers | none (frame threshold) | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 2/4 | 4/4 | 2 (75.5/h) | 0.20 |
|  | pool 1 s | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 2/4 | 4/4 | 2 (75.5/h) | 0.20 |
|  | pool + hysteresis | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 1/4 | 4/4 | 3 (113.3/h) | 0.10 |
|  | pool + 2-state chain | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 2/4 | 4/4 | 2 (75.5/h) | 0.20 |
| (b) CLIP zero-shot | none (frame threshold) | 76/89 / 76/76 | 3/4 / 3/16 | 2/2 / 2/3 | 3/4 | 4/4 | 29 (1095.3/h) | 0.10 |
|  | pool 1 s | 88/89 / 88/88 | 4/4 / 4/5 | 2/2 / 2/2 | 3/4 | 4/4 | 3 (113.3/h) | 0.30 |
|  | pool + hysteresis | 88/89 / 88/88 | 3/4 / 3/4 | 2/2 / 2/3 | 3/4 | 4/4 | 1 (37.8/h) | 0.10 |
|  | pool + 2-state chain | 89/89 / 89/89 | 4/4 / 4/4 | 2/2 / 2/2 | 3/4 | 4/4 | 1 (37.8/h) | 0.30 |
| (b2) CLIP + logistic | none (frame threshold) | 88/89 / 88/88 | 3/4 / 3/4 | 2/2 / 2/3 | 2/4 | 4/4 | 2 (75.5/h) | 0.10 |
|  | pool 1 s | 88/89 / 88/88 | 3/4 / 3/4 | 2/2 / 2/3 | 2/4 | 4/4 | 2 (75.5/h) | 0.10 |
|  | pool + hysteresis | 89/89 / 89/89 | 4/4 / 4/4 | 2/2 / 2/2 | 4/4 | 4/4 | 0 (0.0/h) | 0.20 |
|  | pool + 2-state chain | 88/89 / 88/89 | 2/4 / 2/3 | 2/2 / 2/3 | 1/4 | 3/4 | 2 (75.5/h) | 0.10 |
| (c) DINOv3 + logistic | none (frame threshold) | 76/89 / 76/76 | 2/4 / 2/13 | 2/2 / 2/6 | 2/4 | 3/4 | 18 (679.8/h) | 0.30 |
|  | pool 1 s | 83/89 / 83/83 | 2/4 / 2/6 | 2/2 / 2/6 | 2/4 | 3/4 | 6 (226.6/h) | 0.30 |
|  | pool + hysteresis | 84/89 / 84/84 | 2/4 / 2/5 | 2/2 / 2/6 | 2/4 | 3/4 | 4 (151.1/h) | 0.30 |
|  | pool + 2-state chain | 84/89 / 84/84 | 2/4 / 2/5 | 2/2 / 2/6 | 2/4 | 3/4 | 4 (151.1/h) | 0.30 |

Per frame (5 frames/s, frame threshold, no window step; frames within 0.3 s of a true event left out and counted apart):

| method | underwater R / P | air R / P | air recall % | air precision % | crossing frames called air |
|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | 433/449 / 433/433 | 16/16 / 16/32 | 100.0 | 50.0 | 11/12 |
| (a) colour stats, 18 numbers | 443/449 / 443/443 | 16/16 / 16/22 | 100.0 | 72.7 | 11/12 |
| (b) CLIP zero-shot | 430/449 / 430/430 | 16/16 / 16/35 | 100.0 | 45.7 | 6/12 |
| (b) CLIP zero-shot, untuned threshold 0 | 413/449 / 413/413 | 16/16 / 16/52 | 100.0 | 30.8 | |
| (b2) CLIP + logistic | 446/449 / 446/446 | 16/16 / 16/19 | 100.0 | 84.2 | 9/12 |
| (c) DINOv3 + logistic | 420/449 / 420/420 | 16/16 / 16/45 | 100.0 | 35.6 | 12/12 |

Window step `pool + 2-state chain`, windows with a true event within 0.3 s of their edge left out (truth timing +-0.3 s):

| method | underwater R / P | crossing R / P | air R / P |
|---|---|---|---|
| (a0) mean saturation, 1 number | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 |
| (a) colour stats, 18 numbers | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 |
| (b) CLIP zero-shot | 89/89 / 89/89 | 4/4 / 4/4 | 2/2 / 2/2 |
| (b2) CLIP + logistic | 88/89 / 88/89 | 2/4 / 2/3 | 2/2 / 2/3 |
| (c) DINOv3 + logistic | 84/89 / 84/84 | 2/4 / 2/5 | 2/2 / 2/6 |

### A. Zenodo 4902325 (cross-deployment) (1184 s of video)

Window states: `correct/truth (recall) / correct/predicted (precision)`; events matched within 0.5 s (and 1.0 s), false events per hour, median |timing error| of the 0.5 s matches.

| method | window step | underwater R / P | crossing R / P | air R / P | events found 0.5 s | 1.0 s | false events | median abs err (s) |
|---|---|---|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | none (frame threshold) | 184/356 / 184/184 | 0/40 / 0/9 | 786/786 / 786/989 | 0/40 | 0/40 | 12 (36.5/h) | - |
|  | pool 1 s | 194/356 / 194/194 | 0/40 / 0/4 | 786/786 / 786/984 | 0/40 | 0/40 | 4 (12.2/h) | - |
|  | pool + hysteresis | 196/356 / 196/196 | 0/40 / 0/6 | 786/786 / 786/980 | 0/40 | 0/40 | 6 (18.2/h) | - |
|  | pool + 2-state chain | 192/356 / 192/192 | 0/40 / 0/6 | 786/786 / 786/984 | 0/40 | 0/40 | 6 (18.2/h) | - |
| (a) colour stats, 18 numbers | none (frame threshold) | 225/356 / 225/928 | 30/40 / 30/152 | 6/786 / 6/102 | 15/40 | 20/40 | 213 (647.8/h) | 0.11 |
|  | pool 1 s | 228/356 / 228/951 | 23/40 / 23/123 | 11/786 / 11/108 | 8/40 | 12/40 | 142 (431.9/h) | 0.36 |
|  | pool + hysteresis | 227/356 / 227/972 | 16/40 / 16/91 | 10/786 / 10/119 | 3/40 | 6/40 | 97 (295.0/h) | 0.36 |
|  | pool + 2-state chain | 229/356 / 229/981 | 17/40 / 17/79 | 15/786 / 15/122 | 1/40 | 4/40 | 81 (246.3/h) | 0.43 |
| (b) CLIP zero-shot | none (frame threshold) | 311/356 / 311/477 | 31/40 / 31/635 | 66/786 / 66/70 | 29/40 | 36/40 | 1361 (4139.1/h) | 0.20 |
|  | pool 1 s | 345/356 / 345/775 | 17/40 / 17/221 | 175/786 / 175/186 | 18/40 | 26/40 | 262 (796.8/h) | 0.29 |
|  | pool + hysteresis | 341/356 / 341/886 | 9/40 / 9/84 | 197/786 / 197/212 | 7/40 | 17/40 | 77 (234.2/h) | 0.17 |
|  | pool + 2-state chain | 345/356 / 345/906 | 17/40 / 17/68 | 197/786 / 197/208 | 17/40 | 26/40 | 51 (155.1/h) | 0.24 |
| (b2) CLIP + logistic | none (frame threshold) | 189/356 / 189/225 | 29/40 / 29/171 | 671/786 / 671/786 | 29/40 | 35/40 | 218 (663.0/h) | 0.21 |
|  | pool 1 s | 206/356 / 206/243 | 26/40 / 26/127 | 697/786 / 697/812 | 29/40 | 34/40 | 115 (349.7/h) | 0.24 |
|  | pool + hysteresis | 316/356 / 316/366 | 22/40 / 22/100 | 683/786 / 683/716 | 27/40 | 34/40 | 74 (225.1/h) | 0.24 |
|  | pool + 2-state chain | 213/356 / 213/227 | 22/40 / 22/40 | 768/786 / 768/915 | 26/40 | 30/40 | 14 (42.6/h) | 0.24 |
| (c) DINOv3 + logistic | none (frame threshold) | 332/356 / 332/334 | 35/40 / 35/151 | 694/786 / 694/697 | 33/40 | 40/40 | 263 (799.8/h) | 0.16 |
|  | pool 1 s | 348/356 / 348/360 | 32/40 / 32/108 | 712/786 / 712/714 | 31/40 | 39/40 | 111 (337.6/h) | 0.16 |
|  | pool + hysteresis | 351/356 / 351/371 | 31/40 / 31/102 | 708/786 / 708/709 | 31/40 | 39/40 | 87 (264.6/h) | 0.16 |
|  | pool + 2-state chain | 350/356 / 350/366 | 31/40 / 31/82 | 731/786 / 731/734 | 30/40 | 38/40 | 58 (176.4/h) | 0.16 |

Per frame (5 frames/s, frame threshold, no window step; frames within 0.3 s of a true event left out and counted apart):

| method | underwater R / P | air R / P | air recall % | air precision % | crossing frames called air |
|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | 957/1823 / 957/957 | 3976/3976 / 3976/4842 | 100.0 | 82.1 | 120/120 |
| (a) colour stats, 18 numbers | 1248/1823 / 1248/5010 | 214/3976 / 214/789 | 5.4 | 27.1 | 54/120 |
| (b) CLIP zero-shot | 1745/1823 / 1745/4159 | 1562/3976 / 1562/1640 | 39.3 | 95.2 | 58/120 |
| (b) CLIP zero-shot, untuned threshold 0 | 1680/1823 / 1680/3184 | 2472/3976 / 2472/2615 | 62.2 | 94.5 | |
| (b2) CLIP + logistic | 1112/1823 / 1112/1435 | 3653/3976 / 3653/4364 | 91.9 | 83.7 | 54/120 |
| (c) DINOv3 + logistic | 1764/1823 / 1764/1922 | 3818/3976 / 3818/3877 | 96.0 | 98.5 | 58/120 |

Window step `pool + 2-state chain`, windows with a true event within 0.3 s of their edge left out (truth timing +-0.3 s):

| method | underwater R / P | crossing R / P | air R / P |
|---|---|---|---|
| (a0) mean saturation, 1 number | 192/343 / 192/192 | 0/16 / 0/6 | 775/775 / 775/936 |
| (a) colour stats, 18 numbers | 223/343 / 223/953 | 7/16 / 7/65 | 15/775 / 15/116 |
| (b) CLIP zero-shot | 335/343 / 335/884 | 6/16 / 6/54 | 189/775 / 189/196 |
| (b2) CLIP + logistic | 206/343 / 206/216 | 10/16 / 10/23 | 758/775 / 758/895 |
| (c) DINOv3 + logistic | 340/343 / 340/353 | 11/16 / 11/56 | 723/775 / 723/725 |

Tuned window steps (on the LOPO out-of-fold scores of 00004-00006):

| method | C | none | pool | hysteresis | chain |
|---|---|---|---|---|---|
| sat | 1.0 | bias=2 | k=9, bias=4 | k=3, bias=4, h=2 | k=1, bias=4, penalty=3 |
| color | 0.01 | bias=3 | k=3, bias=3 | k=3, bias=3, h=0.5 | k=3, bias=3, penalty=1 |
| clip0 | None | bias=0.5 | k=9, bias=0.5 | k=9, bias=0.5, h=0.5 | k=3, bias=0.5, penalty=3 |
| clipLR | 0.01 | bias=0.5 | k=3, bias=1 | k=5, bias=2, h=0.5 | k=3, bias=1, penalty=30 |
| dino | 0.1 | bias=-0.5 | k=3, bias=0.5 | k=3, bias=1, h=0.5 | k=3, bias=0.5, penalty=1 |


## Appendix B. Run v1 (arg-max step choice) on the cross-deployment sets

### B. dev clip 2025-12-20 (cross-deployment) (95 s of video)

Window states: `correct/truth (recall) / correct/predicted (precision)`; events matched within 0.5 s (and 1.0 s), false events per hour, median |timing error| of the 0.5 s matches.

| method | window step | underwater R / P | crossing R / P | air R / P | events found 0.5 s | 1.0 s | false events | median abs err (s) |
|---|---|---|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | none (frame threshold) | 85/89 / 85/85 | 2/4 / 2/5 | 2/2 / 2/5 | 1/4 | 2/4 | 5 (188.8/h) | 0.10 |
|  | pool 1 s | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 2/4 | 2/4 | 2 (75.5/h) | 0.20 |
|  | pool + hysteresis | 86/89 / 86/86 | 1/4 / 1/4 | 2/2 / 2/5 | 1/4 | 2/4 | 3 (113.3/h) | 0.30 |
|  | pool + 2-state chain | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 1/4 | 2/4 | 3 (113.3/h) | 0.10 |
| (a) colour stats, 18 numbers | none (frame threshold) | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 2/4 | 4/4 | 2 (75.5/h) | 0.20 |
|  | pool 1 s | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 2/4 | 4/4 | 2 (75.5/h) | 0.20 |
|  | pool + hysteresis | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 1/4 | 4/4 | 3 (113.3/h) | 0.10 |
|  | pool + 2-state chain | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 | 2/4 | 4/4 | 2 (75.5/h) | 0.20 |
| (b) CLIP zero-shot | none (frame threshold) | 81/89 / 81/81 | 4/4 / 4/13 | 1/2 / 1/1 | 4/4 | 4/4 | 24 (906.5/h) | 0.20 |
|  | pool 1 s | 88/89 / 88/88 | 4/4 / 4/5 | 2/2 / 2/2 | 3/4 | 4/4 | 3 (113.3/h) | 0.30 |
|  | pool + hysteresis | 88/89 / 88/88 | 3/4 / 3/4 | 2/2 / 2/3 | 3/4 | 4/4 | 1 (37.8/h) | 0.10 |
|  | pool + 2-state chain | 89/89 / 89/89 | 4/4 / 4/4 | 2/2 / 2/2 | 3/4 | 4/4 | 1 (37.8/h) | 0.30 |
| (b2) CLIP + logistic | none (frame threshold) | 88/89 / 88/88 | 3/4 / 3/4 | 2/2 / 2/3 | 2/4 | 4/4 | 2 (75.5/h) | 0.10 |
|  | pool 1 s | 89/89 / 89/89 | 4/4 / 4/4 | 2/2 / 2/2 | 3/4 | 4/4 | 1 (37.8/h) | 0.10 |
|  | pool + hysteresis | 89/89 / 89/89 | 4/4 / 4/4 | 2/2 / 2/2 | 4/4 | 4/4 | 0 (0.0/h) | 0.10 |
|  | pool + 2-state chain | 89/89 / 89/93 | 1/4 / 1/1 | 1/2 / 1/1 | 0/4 | 1/4 | 1 (37.8/h) | - |
| (c) DINOv3 + logistic | none (frame threshold) | 82/89 / 82/82 | 3/4 / 3/10 | 2/2 / 2/3 | 2/4 | 4/4 | 16 (604.3/h) | 0.10 |
|  | pool 1 s | 83/89 / 83/83 | 2/4 / 2/6 | 2/2 / 2/6 | 2/4 | 3/4 | 6 (226.6/h) | 0.30 |
|  | pool + hysteresis | 86/89 / 86/86 | 2/4 / 2/4 | 2/2 / 2/5 | 2/4 | 3/4 | 2 (75.5/h) | 0.30 |
|  | pool + 2-state chain | 85/89 / 85/85 | 2/4 / 2/4 | 2/2 / 2/6 | 2/4 | 3/4 | 2 (75.5/h) | 0.30 |

Per frame (5 frames/s, frame threshold, no window step; frames within 0.3 s of a true event left out and counted apart):

| method | underwater R / P | air R / P | air recall % | air precision % | crossing frames called air |
|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | 433/449 / 433/433 | 16/16 / 16/32 | 100.0 | 50.0 | 11/12 |
| (a) colour stats, 18 numbers | 443/449 / 443/443 | 16/16 / 16/22 | 100.0 | 72.7 | 11/12 |
| (b) CLIP zero-shot | 440/449 / 440/441 | 15/16 / 15/24 | 93.8 | 62.5 | 3/12 |
| (b) CLIP zero-shot, untuned threshold 0 | 413/449 / 413/413 | 16/16 / 16/52 | 100.0 | 30.8 | |
| (b2) CLIP + logistic | 446/449 / 446/446 | 16/16 / 16/19 | 100.0 | 84.2 | 9/12 |
| (c) DINOv3 + logistic | 435/449 / 435/435 | 16/16 / 16/30 | 100.0 | 53.3 | 10/12 |

Window step `pool + 2-state chain`, windows with a true event within 0.3 s of their edge left out (truth timing +-0.3 s):

| method | underwater R / P | crossing R / P | air R / P |
|---|---|---|---|
| (a0) mean saturation, 1 number | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 |
| (a) colour stats, 18 numbers | 87/89 / 87/87 | 2/4 / 2/4 | 2/2 / 2/4 |
| (b) CLIP zero-shot | 89/89 / 89/89 | 4/4 / 4/4 | 2/2 / 2/2 |
| (b2) CLIP + logistic | 89/89 / 89/93 | 1/4 / 1/1 | 1/2 / 1/1 |
| (c) DINOv3 + logistic | 85/89 / 85/85 | 2/4 / 2/4 | 2/2 / 2/6 |

### B. Zenodo 4902325 (cross-deployment) (1184 s of video)

Window states: `correct/truth (recall) / correct/predicted (precision)`; events matched within 0.5 s (and 1.0 s), false events per hour, median |timing error| of the 0.5 s matches.

| method | window step | underwater R / P | crossing R / P | air R / P | events found 0.5 s | 1.0 s | false events | median abs err (s) |
|---|---|---|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | none (frame threshold) | 184/356 / 184/184 | 0/40 / 0/9 | 786/786 / 786/989 | 0/40 | 0/40 | 12 (36.5/h) | - |
|  | pool 1 s | 194/356 / 194/194 | 0/40 / 0/4 | 786/786 / 786/984 | 0/40 | 0/40 | 4 (12.2/h) | - |
|  | pool + hysteresis | 193/356 / 193/193 | 0/40 / 0/4 | 786/786 / 786/985 | 0/40 | 0/40 | 4 (12.2/h) | - |
|  | pool + 2-state chain | 194/356 / 194/194 | 0/40 / 0/4 | 786/786 / 786/984 | 0/40 | 0/40 | 4 (12.2/h) | - |
| (a) colour stats, 18 numbers | none (frame threshold) | 225/356 / 225/928 | 30/40 / 30/152 | 6/786 / 6/102 | 15/40 | 20/40 | 213 (647.8/h) | 0.11 |
|  | pool 1 s | 228/356 / 228/951 | 23/40 / 23/123 | 11/786 / 11/108 | 8/40 | 12/40 | 142 (431.9/h) | 0.36 |
|  | pool + hysteresis | 227/356 / 227/972 | 16/40 / 16/91 | 10/786 / 10/119 | 3/40 | 6/40 | 97 (295.0/h) | 0.36 |
|  | pool + 2-state chain | 251/356 / 251/1061 | 9/40 / 9/22 | 3/786 / 3/99 | 0/40 | 0/40 | 22 (66.9/h) | - |
| (b) CLIP zero-shot | none (frame threshold) | 328/356 / 328/686 | 27/40 / 27/471 | 22/786 / 22/25 | 25/40 | 33/40 | 931 (2831.4/h) | 0.20 |
|  | pool 1 s | 345/356 / 345/775 | 17/40 / 17/221 | 175/786 / 175/186 | 18/40 | 26/40 | 262 (796.8/h) | 0.29 |
|  | pool + hysteresis | 341/356 / 341/886 | 9/40 / 9/84 | 197/786 / 197/212 | 7/40 | 17/40 | 77 (234.2/h) | 0.17 |
|  | pool + 2-state chain | 345/356 / 345/906 | 17/40 / 17/68 | 197/786 / 197/208 | 17/40 | 26/40 | 51 (155.1/h) | 0.24 |
| (b2) CLIP + logistic | none (frame threshold) | 201/356 / 201/238 | 29/40 / 29/227 | 633/786 / 633/717 | 29/40 | 36/40 | 321 (976.2/h) | 0.13 |
|  | pool 1 s | 276/356 / 276/324 | 24/40 / 24/173 | 667/786 / 667/685 | 26/40 | 35/40 | 204 (620.4/h) | 0.19 |
|  | pool + hysteresis | 352/356 / 352/457 | 20/40 / 20/110 | 612/786 / 612/615 | 20/40 | 32/40 | 92 (279.8/h) | 0.20 |
|  | pool + 2-state chain | 354/356 / 354/406 | 20/40 / 20/46 | 728/786 / 728/730 | 22/40 | 27/40 | 24 (73.0/h) | 0.16 |
| (c) DINOv3 + logistic | none (frame threshold) | 352/356 / 352/383 | 29/40 / 29/339 | 460/786 / 460/460 | 28/40 | 38/40 | 624 (1897.7/h) | 0.18 |
|  | pool 1 s | 348/356 / 348/360 | 32/40 / 32/108 | 712/786 / 712/714 | 31/40 | 39/40 | 111 (337.6/h) | 0.16 |
|  | pool + hysteresis | 356/356 / 356/400 | 28/40 / 28/83 | 699/786 / 699/699 | 29/40 | 37/40 | 55 (167.3/h) | 0.23 |
|  | pool + 2-state chain | 346/356 / 346/347 | 24/40 / 24/26 | 785/786 / 785/809 | 21/40 | 26/40 | 5 (15.2/h) | 0.10 |

Per frame (5 frames/s, frame threshold, no window step; frames within 0.3 s of a true event left out and counted apart):

| method | underwater R / P | air R / P | air recall % | air precision % | crossing frames called air |
|---|---|---|---|---|---|
| (a0) mean saturation, 1 number | 957/1823 / 957/957 | 3976/3976 / 3976/4842 | 100.0 | 82.1 | 120/120 |
| (a) colour stats, 18 numbers | 1248/1823 / 1248/5010 | 214/3976 / 214/789 | 5.4 | 27.1 | 54/120 |
| (b) CLIP zero-shot | 1781/1823 / 1781/4911 | 846/3976 / 846/888 | 21.3 | 95.3 | 49/120 |
| (b) CLIP zero-shot, untuned threshold 0 | 1680/1823 / 1680/3184 | 2472/3976 / 2472/2615 | 62.2 | 94.5 | |
| (b2) CLIP + logistic | 1190/1823 / 1190/1556 | 3610/3976 / 3610/4243 | 90.8 | 85.1 | 46/120 |
| (c) DINOv3 + logistic | 1815/1823 / 1815/2496 | 3295/3976 / 3295/3303 | 82.9 | 99.8 | 22/120 |

Window step `pool + 2-state chain`, windows with a true event within 0.3 s of their edge left out (truth timing +-0.3 s):

| method | underwater R / P | crossing R / P | air R / P |
|---|---|---|---|
| (a0) mean saturation, 1 number | 194/343 / 194/194 | 0/16 / 0/4 | 775/775 / 775/936 |
| (a) colour stats, 18 numbers | 244/343 / 244/1025 | 3/16 / 3/14 | 3/775 / 3/95 |
| (b) CLIP zero-shot | 335/343 / 335/884 | 6/16 / 6/54 | 189/775 / 189/196 |
| (b2) CLIP + logistic | 342/343 / 342/383 | 7/16 / 7/30 | 720/775 / 720/721 |
| (c) DINOv3 + logistic | 339/343 / 339/340 | 10/16 / 10/11 | 774/775 / 774/783 |


v1 Tuned window steps (on the LOPO out-of-fold scores of 00004-00006):

| method | C | none | pool | hysteresis | chain |
|---|---|---|---|---|---|
| sat | 1.0 | bias=2 | k=9, bias=4 | k=5, bias=3, h=2 | k=1, bias=4, penalty=10 |
| color | 0.01 | bias=3 | k=3, bias=3 | k=3, bias=3, h=0.5 | k=3, bias=3, penalty=10 |
| clip0 | None | bias=1 | k=9, bias=0.5 | k=9, bias=0.5, h=0.5 | k=3, bias=0.5, penalty=3 |
| clipLR | 0.01 | bias=1 | k=5, bias=2 | k=5, bias=3, h=1 | k=3, bias=2, penalty=30 |
| dino | 0.1 | bias=3 | k=3, bias=0.5 | k=5, bias=2, h=1 | k=3, bias=1, penalty=30 |
