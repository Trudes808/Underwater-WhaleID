# 06_clips_e2e (WP7): from a folder of tag video to kept clips

`uwid scan DIR` (`src/uwid/pipeline.py`). One run per preset on the 2025-02-03 deployment (`2025-2-3 Mini-mother
11-96`, tag CC-11-96, `.mov` pieces 00004-00012, 9 x 29.7 min = 4.45 h) and the dev clip `2025-12-20 CG.m4v`; a
first run on a second deployment (`TestSubset 2025-1-2 CATS -85`, tag CC-09-85). Nothing was fitted or tuned on a
test video in this work package. Truth for the measures: WP3's AI by-eye frame labels (not ground truth).

Scripts: `jobs/run_scan.sh` (all runs, detached), `jobs/run_cats85.sh`, `evaluate.py` (measures, `results/eval.json`,
`results/tables_auto.md`), `sheets.py` (review sheets, dropped-window check), `hq_scales.py` (dev-clip scales of
the quality parts), `offset_demo.py` (offset check with wrong offsets), `gpu_preprocess_check.py`.

## Settings (fixed before any test-piece number was looked at, except where marked)

| item | value | where it comes from |
|---|---|---|
| window | 1 s of video time, frames at +0.25 s and +0.75 s | user's unit; WP5a (2 frames 0.5 s apart) |
| detector | OWLv2-base, prompt "whale", box >= 0.25; boxes in the burned-in clock dropped; prompts "fish", "school of fish", "small fish", "air bubbles" scored on the same boxes; processor run on the GPU (same torch ops; 120 test frames: whale score max difference 0.031, mean 0.002, 0 decisions at 0.25 changed, 2.6x faster) | WP5a, dev clip |
| presets | `strict`: both frames fire; `sensitive`: either frame | task |
| whole-frame boxes | a box covering > 60 % of the frame counts only when its score >= 0.6 and, with a sensor log, the tag is >= 1 m deep (default since the user's decision 2026-11-20 late) | 0.6 = above every whole-frame "whale" box on the dev clip (max 0.58; whale-free dev windows up to 0.49). The idea came from WP5a's test misses; the 1 m depth limit was added after `scan_wholeframe_sensitive_01.jpg` showed 20 of the 22 whole-frame windows shallower than 1 m to be the tagged whale's back mirrored in the surface at a breath: **both not clean tests** |
| size | a "whale" box shorter than 300 px (longest side, 1920 px frame) is `small` (fish or far whale) and keeps nothing | user's decision 2026-11-20 late; 300 = just above the longest labelled fish on 2025-02-03 (50-280 px, WP3) |
| keep logic | whale windows padded 2 s each side; stretches whose gap after padding is < 5 s merged; nothing dropped for being short | task |
| sensors | tag state (air / flight / floating / whale) from `tagdata`; lens out of water from depth: surfacing `start_t` + 0.3 s to `end_t` - 0.4 s; attaching drop: on the whale from the impact; release: off the whale from the gyro jump (rise counted as floating) | WP1 |
| clock | sensor time = burned-in clock - 2.03 s (`--clock-offset`), clock fitted per file from the tenth changes (60 probes x 6 frames) | WP1; fit agrees with WP1's to 1.4 ms on 00004 |
| detector runs on | windows with tag state `whale` and water `underwater` or `crossing` (sensor path); picture water `underwater` / `crossing` (no log) | task |
| reasons | tag in air, drone flight, tag floating, surfaced, underwater with no whale, small (fish or far whale), unreadable | task + the user's size decision |
| fish | max fish-prompt score per window ("fish", "school of fish", "small fish"), information only | task |

## 1. Result on 2025-02-03 (tag CC-11-96, `.mov` 00004-00012, 267.1 min; truth = AI by-eye labels on 00004-00010)

A labelled frame "is kept" when its `.mov` time falls inside a kept stretch (label frame indices refer to the `.m4v`;
the same index shows the same burned-in clock in both forms on 3 frames of each of the 7 pieces, 21/21).
"Clear whale" = at least one clearly visible other whale; "whale-free" = no clear and no faint whale.

| run | clear-whale frames kept | whale-free frames kept | faint-only kept | fish frames kept | kept min (00004-00010) | kept min (all 9) | hq_whale windows min |
|---|---|---|---|---|---|---|---|
| **sensitive** (default) | **199/221** (90 %) | **32/570** (5.6 %) | 15/49 | 30/45 | 59.6 of 207.8 | **66.7 of 267.1** (25 %) | 11.9 |
| **strict** | **190/221** (86 %) | **12/570** (2.1 %) | 11/49 | 28/45 | 52.1 of 207.8 | **58.2 of 267.1** (22 %) | 11.1 |

Minutes dropped by reason (all 9 pieces): sensitive: tag in air 0.1, drone flight 1.5, tag floating 4.3, surfaced 1.8,
underwater with no whale 183.3, small (fish or far whale) 9.4, unreadable 0.0 (the last part-second of each file);
strict: 0.1 / 1.5 / 4.3 / 2.3 / 189.0 / 11.7 / 0.0. The pieces on disk are almost all on-whale time (the deck, failed
drops and the float are in pieces 00001-00003 and 00013-00014, not on disk), so nearly all the dropping is "no whale".
Kept clear-whale frames per piece (sensitive): 00004 25/26, 00005 26/27, 00006 59/64, 00007 35/36, 00008 20/22,
00009 21/33, 00010 13/13. Whale-free frames kept: 25 of the 32 sit in padding, 7 in a whale window.

By the size and quality of the largest other whale (WP3b's AI by-eye labels, added to `frames.csv` 2026-11-20 23:45
by another agent; size = longest extent / frame width in 0.1 steps; 300 px = 0.16):

| run | size >= 0.3 | size 0.2 | size 0.1 | quality hq | ok | low |
|---|---|---|---|---|---|---|
| **sensitive** | **103/103** | 76/78 | 20/40 | **11/11** | **82/82** | 106/128 |
| **strict** | **103/103** | 72/78 | 15/40 | **11/11** | **82/82** | 97/128 |
| sensitive, 200 px | 103/103 | 78/78 | 32/40 | 11/11 | 82/82 | 120/128 |
| sensitive, 450 px | 100/103 | 42/78 | 10/40 | 11/11 | 80/82 | 61/128 |

Every labelled large (>= 0.3 of the frame width), `ok` or `hq` whale frame is kept by both presets; all losses are
`low` (small, hazy, sliver, cut) frames.

What was lost (sensitive, 22 labelled clear-whale frames, `scan_lost_sensitive_01..04.jpg`, sizes by eye on 960 px
tiles): by the scan's reason 17 "small (fish or far whale)" (the detector found them, but shorter than 300 px) and 5
"underwater with no whale". By eye (AI): **no large whale lost**; 19 are clearly small, far or a part cut by the frame
edge (whale heads at the back's horizon, flipper tips, hazy shapes at the top edge), 3 are borderline at about
240-300 px (00009 f203 whole whale silhouette ~280 px, 00005 f17532 head at the horizon ~260 px, 00009 f5984 body cut
by the top edge ~240 px). Piece 00009 alone has 12 of the 22 (a far whale through most of the piece). Strict loses
9 more (WP3 notes: heads cut by the top edge, hazy whales with fluke, a head at the horizon; not looked at again).

Size limit (the one setting the user asked to see; sensitive / strict):

| min_whale_px | clear-whale kept | whale-free kept | kept min (all 9) | small dropped min |
|---|---|---|---|---|
| 200 | 213/221 / 210/221 | 38/570 / 14/570 | 75.1 / 67.0 | 3.9 / 5.2 |
| **300 (default)** | 199/221 / 190/221 | 32/570 / 12/570 | 66.7 / 58.2 | 9.4 / 11.7 |
| 450 | 152/221 / 145/221 | 21/570 / 7/570 | 48.6 / 41.0 | 23.6 / 26.9 |
| none (WP5a settings, strict, plain 60 % filter) | 215/221 (strict) | 32/570 | 75.1 | 0 |

Options, each run once (NOT clean tests: the ideas came from test-piece failures):
- Near whales (box over 60 % of the frame kept when score >= 0.6): before the 1 m depth limit it added 1.1 min, all
  of it at breaths (22 whole-frame windows shallower than 1 m; 20 by eye are the tagged whale's back mirrored in the
  surface, `scan_wholeframe_sensitive_01.jpg`); with the limit the keep decision is identical to the plain 60 %
  filter (sensitive 199/221, 32/570, 66.7 min both) because the near whales WP5a missed are kept through their other
  boxes; it still changes the quality tier (hq 11.9 vs 11.6 min). The 45 whole-frame windows deeper than 1 m are near
  whales by eye (`scan_wholeframe_sensitive_01..03.jpg`), the most valuable footage.
- Fish veto (whale box only if its whale score beats its fish score): sensitive 198/221 (-1), whale-free 25/570 (-7),
  65.5 min (-1.2); strict 190/221 (0), 8/570 (-4), 57.7 min. Off by default.

Same detector on `.mov` vs WP5a's cached `.m4v` boxes (WP5a settings, labelled windows the scan scanned): 802 of 822
decisions agree, 5 fire only on `.m4v`, 15 only on `.mov`.

## 2. Dropped-window check (AI by-eye, the costly error)

Seed 20261120, `sheets.py dropcheck`: 150 windows drawn at random from the 10,953 sensitive windows dropped as
"underwater with no whale", plus 30 from the 564 dropped as "small (fish or far whale)"; centre frame of each looked at
on 960 px tiles (`scan_dropcheck_01..30.jpg`, calls in `results/dropcheck_calls.json`).

| set | none | fish only | small whale | faint whale | large whale | other |
|---|---|---|---|---|---|---|
| no whale (150) | 146 | 1 | 2 (#71 head at the back's horizon ~90 px, #79 far whale ~140 px) | 1 (#73 dark part at the left edge ~300 px) | **0** | 0 |
| small (30) | 0 | 9 | 20 | 0 | **0** (borderline #164: calf cut by the top edge ~250 px) | 1 (ray-like shape) |

So in the "no whale" drops about 2 % of windows (3/150) show any whale and none a large one; scaled to the pool that
is roughly 200 windows (3 min) of small / faint whale. The "small" drops are mostly real whales (20/30), as the user's
size decision intends; they are what a smaller `--min-whale-px` would keep. Found whales:
`scan_dropcheck_found_01..04.jpg`.

## 3. Pieces 00011 and 00012 (no labels)

00011: kept 4.0 of 29.7 min in 11 stretches (sensitive). 00012: kept 3.2 min; the release (gyro jump at 14:38:44
sensor time, 1516 s into the piece) and the 30 s rise are counted as floating; 4.3 min dropped as "tag floating", 6 s as
"tag in air" (spray at the surface), the rest "no whale". By eye (`scan_release_00012.jpg`, 18 frames 1480-1780 s):
the windows kept at 1508-1517 s show a large whale below (the detached tag looking at its whale, or the calf) and are
right to keep; the float from 1600 s on is open water, dropped correctly; but two float frames show whale (a far
shape at 1520-1525 s, a near pectoral fin at 1555 s): **whale footage during a float is dropped as "tag floating"**,
because the detector does not run on floating time. Timeline: `scan_timeline_00012.jpg`.

## 4. Dev clip, no sensor log (picture water state)

`2025-12-20 CG.m4v` (95 s, 191 labelled frames): sensitive keeps 1.52 of 1.59 min (clear-whale frames 145/145,
whale-free 28/37: the clip is a crowded group, whale-free frames sit inside whale stretches); strict 1.42 min (144/145,
20/37). Dropped: 0-1 s and 94-95 s "surfaced", 1-3 s "small", last 0.3 s "unreadable" (no picture state for a partial
window). Picture water state against the label state per frame: 168/180 underwater frames underwater, 8 crossing,
4 "surfaced" (false); 6/8 air frames surfaced. Timeline: `scan_timeline_dev.jpg`.

## 5. Clock offset check

Picture exits / entries (`uwid.waterstate`, DINOv3) against depth exits (`start_t` + 0.3 s) and entries (`end_t` -
0.4 s), mapped with burned-in clock - 2.03 s: 156 of 156 depth events paired; picture minus depth median +0.04 s
(exits -0.04, entries +0.07, MAD 0.11 s) -> verdict ok, implied offset 2.07 s. With a deliberately wrong offset
(`offset_demo.py`): 3.03 s -> -0.96 s, 1.53 s -> +0.54 s, 12.03 s -> -9.96 s, each flagged OFFSET LOOKS WRONG and each
recovering 2.07 s. My clock fit agrees with WP1's on 00004 to 1.4 ms.

## 6. Quality tier `hq_whale` (first guess, no truth)

hq score = mean of parts in 0-1: cover (whale box share of the frame, 0.02 -> 0.30), detector score (0.25 -> 0.70),
brightness / contrast against the surrounding water / sharpness inside the box (0 at the dev clip's 10th percentile,
1 at its 90th: 0.23-0.40, 0.003-0.057, log10 Laplacian variance -4.58 to -2.67; `hq_scales.py`), no bubbles (largest
"air bubbles" box share 0.5 -> 0; it never fired >= 0.3 here), depth (30 m -> 0 m, only with a log). hq_whale = whale
window, box >= 10 % of the frame, hq score >= 0.6. No SAM masks (boxes only).

Against WP3b's AI quality labels (another agent's eye, not truth): the windows of the 11 `hq` frames are all
hq_whale, 25 of 82 `ok`, 1 of 128 `low` (median hq score 0.65 / 0.56 / 0.47).

Result (sensitive): 717 of 2,985 whale windows (11.9 min); padded and merged hq stretches 23.0 min; 95 of the 154
clips contain an hq window (clip tier `hq_whale`). Per piece hq stretch min: 00004 2.2, 00005 2.7, 00006 6.3, 00007
4.4, 00008 2.3, 00009 0.9, 00010 1.4, 00011 1.8, 00012 1.0. Depth: the best 40 windows are at 0.1-4.1 m (median
1.3 m); all hq windows 10 / 50 / 90 % = 1.5 / 21.5 / 51.4 m, all whale windows 4.6 / 25.3 / 73.9 m. Large-whale
windows not hq: 1,557 because the box is under 10 % of the frame, 711 below the score. Sheets for the user:
`scan_hq_top_01..04.jpg` (best 40: near whales just under the surface), `scan_hq_border_01..04.jpg` (40 around 0.6:
whole whales at 20-50 m, dark), `scan_hq_rejected_01..04.jpg` (40 random rejects with the reason).

## 7. Cost (shared RTX 4000 Ada, heavily used by other projects during this run)

| stage | time | per hour of video |
|---|---|---|
| sensor cache (one pass over the 3.1 GB CSV) + tables | 143 s + 1.4 s per deployment (WP1) | - |
| clock fit (60 probes x 6 frames) | ~7 s per 30-min piece (CPU) | 14 s |
| picture water state (DINOv3, 5 frames/s, offset check) | 70-77 s GPU-held per piece when the GPU was free (up to 377 s under contention) | ~150 s GPU |
| detector (OWLv2, 2 frames per window, GPU preprocessing) | 28,616 frames in 1,221 s GPU-held (23.4 frames/s; 3.4 frames/s with CPU preprocessing) | ~300 s GPU (0.08 GPU-h) |
| re-scoring a preset / option from the cache | ~10 s per deployment | - |
| cutting 154 clips (copy) + clock at both ends | ~5 min | ~70 s |

Whole chain without waiting: about 0.13 GPU-h and ~9 min wall per hour of video. Waiting for the shared GPU lock
dominated the real wall time (the final model run waited 96 min of its 117). Sensors first: the detector ran on 15,519
of 16,035 windows (96.8 %): on these pieces it saves only 3 %, because they are on-whale time; on the whole recorded
deployment (6.6 h, a third deck / flight / float by WP1) it would save about a third, and on CATS-85 below about half.

## 8. Clips (sensitive, copy mode)

154 clips, 68.2 min (incl. keyframe lead), 0.75 GB against 2.16 GB of source `.mov` (35 %); tiers 95 hq_whale / 59
whale. Three clips decoded (`results/clips_check.json`): start on the keyframe at or before the wanted start (0.4-0.5 s
early), frame count and length equal the index (350, 2,473, 952 frames), first and last frame equal the source frames
the index names (NCC 1.000), burned-in clock text equal at both ends. Sheets `scan_clips_01..02.jpg`.

## 9. What broke or does not work yet

- Threaded HEVC decoding deadlocks once torchvision is loaded (PyAV 19 + torchvision 0.21): fixed (one thread in a
  process with torch; model stages decode in a spawned child with all threads).
- The near-whale rule first counted the tagged whale's back mirrored in the surface at breaths as whales and put them
  at the top of the quality ranking: fixed with the 1 m depth limit (not a clean test; no fix without a log).
- Whales next to a floating tag are dropped as "tag floating" (00012). The detector could run on floating time too
  (4 min here).
- The picture-only path calls 4 of 180 underwater dev frames "surfaced"; with no log a deck and a breath are both
  "surfaced".
- hq tier: boxes, not masks; scales from the dev clip; threshold a guess; bubbles part never fired.
- Small whales (< 300 px) are dropped by design; 20 of 30 sampled "small" drops are whales, 3 of the 22 lost labelled
  whales are borderline 240-300 px.

## 10. First run on a new deployment: `TestSubset 2025-1-2 CATS -85` (tag CC-09-85), nothing fitted on it

Copy complete at 00:30 (9 `.mov` 3.8-18 min, 4K HEVC 30 frames/s, keyframe every 0.5 s, 26 GB; sensor CSV 2.9 GB).
Command, unchanged defaults: `uwid scan "/home/sat3737/NOAA_whales/underwater_video/TestSubset 2025-1-2 CATS -85"`
(job `jobs/run_cats85.sh`, log `underwater_cache/scan/logs/run_cats85.log`).

What broke or differed:
- The burned-in clock has another format (`02/01/25 12:20:57.447`, DD/MM/YY, milliseconds, 4K): the glyph reader
  reads 0 of 360 frames per file, so sensor time falls back to the log's camera time or the file name.
- The camera splits each 30-min recording session into 4 GB files (two per session), so file sequence numbers do not
  match the log's camera pieces; the camera-time fallback mapped 00006 (11:30, on deck) onto on-whale time (fixed:
  the camera-time match is now used only when it agrees with the file name's start time within 3 s; all 9 files map
  by file name, +-1 s on the earlier tag).
- File names carry `-HRA` after the tag number: `uwid.names` does not parse them (date taken from the folder).
- Sensors: no drone flight found (no drop: tag put on by other means); deck 0-2:20 h, on the whale 12:31:28-16:11:33
  local (3.7 h, 84 surfacings), release by gyro jump at 16:10:55, then air (no float detected).
- Tag state per file from the sensors (minutes): 00005 deck 17.8, 00006 deck 11.6, 00009 deck 17.3, 00010 deck 1.5 +
  whale 10.6, 00011 whale 25.6, 00012 whale 3.8, 00017 whale 23.2, 00023 whale 22.7, 00024 whale 4.9 + air 1.2 +
  floating 0.6: 49.4 of 141.0 min are the tag in air (deck) and 0.6 floating, dropped by the sensors alone; the
  detector runs on 88.9 min (on-whale 90.8 min, the rest surfaced).
- Clock offset check (picture model run on the 5 on-whale files done by 05:45, events mapped by file-name time since
  no clock is readable): per file picture minus depth -0.38 (00010, 7 pairs), +0.73 (00011, 14), -0.73 (00012, 2),
  +0.44 (00017, 5), -0.58 s (00023, 19); pooled -0.38 s over 34 of 68 depth events, MAD 0.20 -> "OFFSET LOOKS WRONG".
  That is the file names' whole-second start times (each file off by its own fraction of a second), which the check
  sees; one global offset (`--clock-offset auto`) cannot correct per-file errors. With the 2 s padding the keep
  decision is not affected; per-file correction is not built. On 00023 only 19 of 37 depth events have a picture event.
- Detector, kept minutes and kept-window sheets: the job was still waiting for the shared GPU lock at 05:45 (it holds
  the lock at most 12 min at a time; 6 picture-model pieces took 114-200 s each in 4K). When it finishes:
  `underwater_cache/scan/TestSubset 2025-1-2 CATS -85/sensitive/summary.json` and
  `.venv/bin/python experiments/06_clips_e2e/sheets.py kept --dep "TestSubset 2025-1-2 CATS -85"` makes
  `scan_kept_202512_NN.jpg`.
