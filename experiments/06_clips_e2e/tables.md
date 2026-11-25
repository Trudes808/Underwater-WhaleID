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

## 11. WP9: a tag the tool was not built on (`TestSubset 2025-1-2 CATS -85`, tag CC-09-85), and the rule changes

Nothing was fitted on this deployment except where marked (the clock's glyphs and the clock-to-sensor offset are
measured from its own frames and events by design; the back filter came from looking at its kept windows: NOT a
clean test). Truth: WP3b's AI by-eye labels of this deployment (420 frames, `labels/CC-09-85-*/frames.csv`, spans
files), not ground truth; by-eye counts below are mine (AI), on sheets.

### 11.1 What broke when the tool met this tag (first run, 2026-11-21, unchanged defaults)

File names with an extra part (`-HRA-`) not parsed; the burned-in clock in another layout (`02/01/25 15:45:02.475`,
DD/MM/YY, milliseconds, 4K, proportional font) read on 0 frames, so video time was mapped by file name (whole
seconds; each file off by -0.5 to +0.6 s); the sensor log's camera pieces did not match file numbers (4 GB file
splits); the offset check paired 40 of 108 depth events and said OFFSET LOOKS WRONG (picture minus depth -0.43 s at
the CC-11-96 offset 2.03 s, per file -0.73 to +0.73 s). Sheets drew raw 1920-px detector boxes on 4K frames (wrong
place, a quarter of the size).

### 11.2 Burned-in clock (`uwid.timecode`, glyphs learned from 00005 and its file name)

Layout found by itself (`find_layout`: box 3488-3815 x 2128-2160, 21 cells, `DD/MM/YY`), digit templates learned in
3 passes (pass 0 read 640 / 640 sampled frames already). Read 10 consecutive frames every 10 s of every recording
(`clock_report.py`, `results/clock_CC-09-85.json`):

| recording | frames read | monotonic | slope - 1 (ppm) | rms (ms) | max residual (ms) | clock - file name (s) |
|---|---|---|---|---|---|---|
| 00005 | 1070/1070 | yes | +0.9 | 13.7 | 32 | +0.48 |
| 00006 | 700/700 | yes | +9.1 | 13.3 | 35 | -0.32 |
| 00009 | 1040/1040 | yes | +0.1 | 14.0 | 30 | -0.52 |
| 00010 | 730/730 | yes | +5.5 | 12.5 | 37 | -0.32 |
| 00011 | 1540/1540 | yes | -1.0 | 12.5 | 28 | -0.51 |
| 00012 | 230/230 | yes | +16.9 | 14.0 | 30 | +0.22 |
| 00017 | 1390/1390 | yes | +0.6 | 13.2 | 33 | -0.52 |
| 00023 | 1370/1370 | yes | -0.4 | 12.6 | 28 | +0.49 |
| 00024 | 410/410 | yes | +10.0 | 13.6 | 33 | +0.64 |

8,480 of 8,480 frames read, 0 outliers. The rms of 13 ms is the clock's own stamping jitter (consecutive frames step
10-50 ms, median 30 ms; the last digit is a constant 5 or 7 per session, so the clock ticks in 10 ms). The file
names carry the start time rounded to the second (clock - name -0.52 to +0.64 s). The first tag's clock reader is
unchanged (its glyph set is picked by `glyphs_for`; tests unchanged and passing).

### 11.3 Time link and clock offset

Each file is mapped through its own clock fit (`plain readings`, 360 per file). The offset for this unknown tag is
estimated (`--clock-offset tag` -> auto): picture exits / entries (`uwid.waterstate`) against depth exits / entries,
as the midpoint of the exit and entry medians. Two fixes for a tag the check was not built on: (1) depth "surfacings"
shorter than the lens lags (brushes of the surface, 0.1-0.9 s, 44 of 108 on this tag) are no longer events; (2) the
midpoint, because this tag's lens leaves the water earlier and enters later than the CC-11-96 lags assume (exits
-0.29 s, entries +0.20 s: half-difference 0.25 s), which makes a pooled median depend on the mix of kinds.

| | before (file names, offset 2.03) | after (clock, estimated offset -0.04 s) |
|---|---|---|
| depth events / paired | 108 / 40 | 64 / 53 |
| pooled picture minus depth | -0.43 s, "OFFSET LOOKS WRONG" | midpoint 0.00 s (exits -0.25, entries +0.25), "ok" |
| 00010 | -0.38 s (7 pairs) | -0.12 s (8: 4 exits, 4 entries) |
| 00011 | +0.73 s (14) | +0.11 s (14) |
| 00012 | -0.73 s (2) | -0.46 s (2: 1 exit, 1 entry) |
| 00017 | +0.44 s (5) | -0.08 s (5) |
| 00023 | -0.58 s (19) | +0.01 s (19) |
| 00024 | - | -0.26 s (5) |

Within 0.2 s on 4 of the 6 recordings with surfacings; not on 00012 (2 events) and 00024 (5 events, near the
release); the per-event scatter is MAD 0.31 s, so a median of 2-5 events is uncertain by about that. Burned clock -
sensor time: CC-09-85 -0.04 s, CC-11-96 2.03 s. Default decided from this: `--clock-offset tag` = the measured
offset of a known tag (table `KNOWN_CLOCK_OFFSETS`, only CC-11-96), otherwise estimated, written to
`offset_check.json` with per-recording medians, and the scan STOPS with an error when fewer than 10 events (3 exits,
3 entries) match. The 2.03 s of the other tag would have shifted every lens state here by 2 s.

### 11.4 Rule changes in this work package (user decisions 2026-11-22)

- Keep rule: every window with anything detected is kept: a whale box of any size >= 0.25 or a fish box >= 0.5
  (WP5a's dev-chosen fish point). Dropped: tag in air / flight, surfaced, floating or underwater with nothing
  detected, unreadable. `--whale-only` gives the old rule.
- Tier per window and per kept stretch (a label, not a filter): hq (= `hq_whale`), ok (whale box >= 400 px of a 1920
  px frame), low, small, fish, none. The 400 px came from the dev clip only (`tier_rule_dev.py`: the largest limit
  that loses none of the 109 hq / ok dev frames; it also passes 32 of 36 low frames: box size barely separates ok from
  low on that crowded clip, and no score limit helped without losing hq / ok frames). `advanced_ok` + `hq_spans` per
  stretch; a 1-2 sentence description per stretch built from the window columns; `clips.txt` per deployment.
- Floating time is scanned where the picture says the lens is under water.
- Silhouette flag (label only): whale box >= 576 px, mean grey inside <= 0.40, ring around it >= 0.02 brighter.
  Chosen on the dev clip only (`silhouette_rule_dev.py`): it finds 14 of 29 dev silhouettes with 44 false of 116;
  every box measure has AUC 0.43-0.67 on that clip. Weak: a box contains other whales and water; masks would be needed.
- Back filter (NOT clean, from this deployment's kept windows): a "whale" box across >= 95 % of the width, reaching
  the bottom edge and starting below 30 % of the height is the tagged whale's own back. On this tag the back fills
  the lower half (box cover 0.5-0.6, under the 60 % whole-frame limit): 125 of 432 whale windows and 122 of 247
  hq windows rested on such a box before the filter.
- `uwid scan --cut` cuts every kept stretch; `--min-tier` optional.

### 11.5 Result on CC-09-85 (`.mov` x 9, 140.8 min)

| run | kept min | hq / ok / low / small / fish windows (min) | stretches hq / ok / low / small / fish | dropped: air / floating / surfaced / nothing |
|---|---|---|---|---|
| first run 2026-11-21 (whale only, by file name) | 15.6 | hq 4.1 (246 windows) | - | 49.5 / 0.6 / 1.0 / 73.1 (+ small 1.0) |
| sensitive, whale only, no floating, back filter | 13.2 | 2.9 / 2.8 / 0.8 / - / - | 33 / 15 / 4 / 0 / 0 | 49.5 / 0.6 / 1.5 / 76.0 |
| sensitive, no floating | 16.4 | 2.9 / 2.8 / 0.8 / 2.1 / 0.1 | 33 / 15 / 2 / 10 / 0 | 49.5 / 0.6 / 1.4 / 72.9 |
| **sensitive (default)** | **16.6** | 2.9 / 2.8 / 0.8 / 2.1 / 0.1 | 33 / 15 / 2 / 11 / 0 | 49.5 / 0.4 / 1.4 / 72.9 |
| sensitive, no back filter | 18.8 | 4.1 / 2.4 / 0.7 / 2.0 / 0.1 | 50 / 7 / 2 / 7 / 0 | 49.5 / 0.4 / 1.0 / 71.1 |
| strict | 11.3 | 2.3 / 1.7 / 0.3 / 2.6 / 0.2 | 24 / 10 / 1 / 10 / 1 | 49.5 / 0.6 / 1.6 / 77.9 |

Per recording (sensitive): 00005 / 00006 / 00009 boat: 0 kept, all "tag in air"; 00010 0.7 of 12.1 min; 00011 1.0 of
25.6; 00012 0.1 of 3.8; 00017 2.8 of 23.2; 00023 8.8 of 22.7; 00024 3.2 of 6.8. Size on disk (copy mode, estimated from
the 23.4 Mbit/s source, clips not cut): every kept stretch ~2.9 GB of 26 GB; hq stretches ~1.9 GB; hq + ok ~2.5 GB.

Against WP3b's AI by-eye labels (`evaluate.py --dep d2`, `results/eval_d2.json`), sensitive / strict:

| measure | sensitive | strict |
|---|---|---|
| frames with another whale kept | 27/31 (hq 2/2, ok 18/18, low 7/11) | 24/31 (hq 2/2, ok 17/18, low 5/11) |
| frames with any activity (clear or faint whale, fish) kept | 36/51 | 32/51 |
| fish frames kept | 7/7 | 7/7 |
| empty underwater frames kept (cheap error) | 31/292 (21 padding, 8 "whale", 1 small, 1 fish) | 14/292 |
| hq or ok frames in a tier ok or better | 13/20 | 11/20 |
| frames in tier ok or better graded low or whale-free | 13/26 | 4/15 |
| frames in tier hq (advanced_ok) graded low or whale-free | 1/11 | 1/9 |
| hq frames in tier hq | 2/2 | 2/2 |

Grade (rows, labels) against tier (columns), sensitive: hq: 2 hq; ok: 8 hq, 3 ok, 1 small, 6 none (kept by padding);
low: 1 ok, 1 low, 2 small, 7 none; whale-free: 1 hq, 10 ok, 4 small, 1 fish, 358 none; faint only: 1 ok, 1 small, 13 none.
The 10 whale-free frames tiered ok: 3 near jacks 360-524 px called whale (the 300-500 px band), 5 "just under the
surface" in 00024, 1 exhalation bubbles (00010), 1 sediment / bottom seen sideways (00017 511 s).
Lost activity (15 frames): 5 faint whales in 00010 0-80 s that the labels call underwater while the sensors call the
tag "in air" (12 labelled-underwater frames fall in sensor "air" there: the tag was in the water before the log's
on-whale state; not resolved), 10 faint or low whales dropped as "nothing detected". No hq or ok whale frame was lost.
Tag / water state against the labels: 67 of 71 air frames air; 320 of 338 underwater frames underwater (12 sensor
"air" in 00010 above, 3 crossing, 3 surfaced).

### 11.6 Dropped-window check (AI by-eye, seed 20261122)

100 windows drawn from the 4,371 "nothing detected" windows (`sheets.py d2_dropcheck`, sheets
`scan_d2_dropcheck_01..17.jpg`, 960 px tiles, calls in `results/dropcheck_d2_calls.json`): large whale 0, small or
far whale 0, fish 2 (one ~90 px, one ~40 px), unsure specks / an edge shape 4, empty 65; view blocked: lens covered
(black) 7, stirred-up sand / murk 5, bubbles / foam at a breath 1; inside the labeller's seafloor spans 16 (at 960 px
I could not recognise the bottom in them: haze with a darker band). These windows were drawn before the back filter;
the filter only adds windows to "nothing detected" (back boxes), it removes none.

### 11.7 Seafloor, sand and dark spans (labeller's spans, per 1 s window, sensitive)

| span | windows | whale fires | small | fish | kept | hq | picture water state |
|---|---|---|---|---|---|---|---|
| seafloor visible (00017 842-1390 s, 00023 0-306 s) | 854 | 138 | 17 | 5 | 210 | 65 | 854 underwater |
| seafloor unsure (00017 296-842 s) | 556 | 14 | 10 | 9 | 57 | 8 | 556 underwater |
| sand (blocked) | 203 | 0 | 2 | 1 | 8 | 0 | 203 underwater |
| dark (lens covered) | 215 | 0 | 1 | 1 | 3 | 0 | 215 underwater |

The bubble measure (an "air bubbles" box >= 0.3) fired in none of these windows. By eye on 24 of the 180 seafloor-span
windows with a detection (`scan_d2_seafloor_fired_01..02.jpg`, seed 20261122): 20 are real other whales passing
over the bottom (hazy at 40-48 m), 1 a fish called whale (jack, 326 px), 2 small boxes on fish / sediment specks, 1
small box at the frame edge on a whale's fluke; 0 on the bottom itself, 0 on rocks or reef, 0 on the tagged whale's
shadow. Elsewhere one small box (239 px) sat on the dark object at the bottom centre of 00017 frames (tag part or
shadow, `scan_d2_seafloor_01.jpg`), and one labelled frame tiered ok rests on sediment / bottom seen sideways (00017
511 s). Sand sheet: `scan_d2_sand_01.jpg` (2 of 203 sand windows had a detection: small 0.48 and fish 0.52 boxes on
dark sediment flakes). So the detector does not take the seafloor or sand for whales here; the picture water model
calls all of it underwater (right), and nothing marks the footage as sand or dark yet (WP10's columns).

### 11.8 What is kept, by eye (sheets `scan_d2_clips_middle_01..05`, `scan_d2_clips_late_01..04`, `scan_d2_hq_*`)

Middle recordings (00010-00017, all 29 kept stretches, first / middle / last frame): most stretches in 00010 and
00011 are at breaths (depth 0-1 m): the tagged whale's blow, its back mirrored in the surface and the surface itself
are boxed as "whale" (boxes under 60 % of the frame, so the 1 m whole-frame rule does not apply), and they get tier
hq; real other whales in 00010 387-411 s, 00011 168-174 s, 450 s and 841-847 s (far, hazy). Whale windows shallower
than 1 m: 123 of 389, 71 of the 174 hq windows; top-12 hq windows: 5 breaths (blow / mirror), 7 real near whales
(00023, 00024). Late recordings (00023, 00024, 20 stretches at random): mostly real other whales near the tagged
whale, several very close and sharp (the best footage of this deployment), and fish (jacks).

### 11.9 4K

Decoding sets the speed: the detector held the GPU 820 s for 10,666 frames (13 frames/s; 23 at 1080p), of which
decoding 817 s and the model 377 s; the picture water model 114-200 s per 30-min piece (70-77 s at 1080p). Frames are
decoded straight to 1920x1080 for the detector (OWLv2 sees 960x960 either way); box sizes and the size limits are in
pixels of a 1920 px wide frame, i.e. shares of the frame width, so a whale that is large in a 4K frame is judged by
the same rule; the quality measures are taken on a 960 px copy. Fixed: sheets drew detector boxes unscaled on 4K
frames (now scaled to the image). Still tag-specific: the clock-box filter for boxes (`detect.CLOCK_BOX`, the first
tag's corner; this tag's clock lies inside it).

## 12. WP9: the first deployment (2025-02-03, CC-11-96) under the new rules (re-scored from the cache)

Only 00012's floating frames were new detector work (518 frames). Truth: WP3 / WP3b AI by-eye labels, 00004-00010
(the grade rule hq / ok / low was confirmed by the user on 39 example frames; the frames themselves were graded by an
AI agent, not by the user). The tier's 400 px and the silhouette limits came from the dev clip only; nothing here
was fitted on these pieces. The back filter changes nothing here (`sensitive+nofloat+whaleonly` = the old 66.7 min,
199/221, 32/570 exactly).

| run | kept min (all 9) | clear-whale frames kept | any activity kept (clear / faint whale, fish) | empty underwater kept | fish frames kept |
|---|---|---|---|---|---|
| before (whale only, no floating) | 66.7 | 199/221 | 215/281 | 28/542 | 30/45 |
| **sensitive (anything detected, floating scanned)** | **86.6** | **219/221** | **259/281** | 53/542 | 41/45 |
| strict | 76.5 | 216/221 | 249/281 | 29/542 | 38/45 |

The 22 activity frames still lost (sensitive): 2 clear whales (faint-ish, "nothing detected"), 13 faint-only, 6
fish-only frames with a few fish, 1 at a breath.

Minutes per tier (sensitive, windows): hq 12.1, ok 25.8, low 12.2, small 16.7, fish 1.1; kept stretches by their best
tier: hq 89 (64.4 min), ok 37 (17.6), low 10 (2.0), small 23 (2.3), fish 2 (0.2). Size on disk (copy mode,
estimated from the 2.16 GB / 267 min source): every stretch ~0.70 GB, stretches with an hq window ~0.52 GB,
hq + ok ~0.66 GB; the hq spans alone (`hq_spans`, what goes to 3D / identification) are much shorter.

Grade (rows, AI labels) against tier (columns), sensitive, labelled frames:

| grade | hq | ok | low | small | fish | none |
|---|---|---|---|---|---|---|
| hq (11) | 11 | 0 | 0 | 0 | 0 | 0 |
| ok (82) | 25 | 56 | 1 | 0 | 0 | 0 |
| low (128) | 1 | 35 | 44 | 38 | 0 | 10 |
| whale-free (570) | 7 | 0 | 0 | 10 | 2 | 551 |
| faint only (49) | 0 | 1 | 3 | 11 | 2 | 32 |

- hq or ok frames in a tier ok or better: 92/93 (the costly error: 1 ok frame tiered low).
- frames the tool calls ok or better that the labels grade low or whale-free: 43/136 (35 low, 7 whale-free,
  1 faint); strict 38/129.
- frames in tier hq (`advanced_ok`) graded low or whale-free: 8/44 (1 low, 7 whale-free); strict 2/38. Not looked
  at one by one; the 7 whale-free frames are the costly case for the advanced work.
- ok against low: of 128 low frames, 36 land in hq / ok: box size does not tell a hazy 0.3-width whale from a clear
  one, as on the dev clip.

Silhouette flag against WP3b's `silhouette` column (labelled frames with a whale, sensitive): labelled yes 34: 13
flagged; labelled no 187: 27 flagged. Weak, as on the dev clip (14 of 29). Box brightness measures do not carry it;
masks would.

Floating time, 00012 (the release and 25-min float; `float_check.py`, results `float_check_00012.json`): 260
floating windows, 259 under water by the picture, all scanned; 23 have something detected (hq 8, ok 7, low 4,
small 4), all at 14:38:46-14:39:42 just after the release; kept 4.0 -> 4.7 min of 29.7. By eye
(`scan_d2_float_01..03.jpg`): a large whale passes the floating tag at 14:39:30-14:39:42 (pectoral fin and body very
near, the clearest whale of the float) - footage the old rule dropped. CATS -85: 38 s floating, all under water, 3
windows with something (2 fish, 1 small), 13 s kept; 16.4 -> 16.6 min.

Mid-sized boxes (the band where a near fish can pass as a whale, user 2026-11-22): whale windows whose box is
300-500 px: 1,289 of 3,002 on 2025-02-03 (9 hq), 99 of 389 on CATS -85 (0 hq); 40 at random on `scan_midsize_01..04.jpg`
(seed 20261120), not judged by me. Example from the user's sheet: 00006 frame 14470, a 287 px fish with whale 0.54
against fish 0.49 (13 px under the size limit; "whale beats fish" would not catch it). CATS -85: 3 near jacks
(360-524 px) are tiered ok.

Sheets now show the decision, not raw boxes (user feedback 2026-11-22): one box per object (boxes of different
prompts on one region merged), red whale / orange small (fish-sized "whale") / cyan fish (information), each with the
score that decided it and its size; boxes that play no part are left out; boxes scaled to the image (4K). Re-made with
the same windows: `scan_hq_top_01..04`, `scan_hq_border_01..04`, `scan_hq_rejected_01..04`; new: `scan_hq_band_050 /
040 / 030_01..02` (24 random windows per band below 0.6, seed 20261120; pools 487 / 194 / 28 windows; threshold
unchanged) and the same for CATS -85 (`scan_d2_hq_band_*`). In the user's two flagged windows the tool was right
(00007 12:12:30.2: whale box 0.96 kept and hq, the sheet had drawn the "small fish" box of the same region over it;
00006 11:20:53.1: the four fish-sized "whale" boxes were `small`, kept as hq for the large 989 px box).

## 13. WP11b: box-classifier veto, sand / dark labels, 30 s activity guard (both deployments, re-scored)

Nothing was tuned on the test deployments here. The box classifier (`uwid.boxclf`, weights `boxclf_v1.npz`) was
fitted by WP10 on all its labelled boxes: 1,054 objects (dev 335, 2025-02-03 560, 2025-01-02 75 reference objects
and 84 extra 300-800 px boxes, among them the 24 jacks of 00023). The label scores below are on frames whose boxes it
was fitted on: **not a clean test**. WP10's breath boxes (138) were not in its training set. Truth: AI by-eye labels.

What changed in `uwid scan`:
- Veto: a rule-A "whale" box of >= 300 px stays a whale only if the classifier says whale; "fish" -> a fish object
  (activity, tier fish), "other" -> dropped from the whale decision (kept in the table and on sheets, grey, "(clf)").
  Shallow boxes go through the same veto (no depth rule). Below 300 px the size rule decides. Only frames with such a
  box are classified (each decoded once, together with the window frames below); results cached per recording in
  `<cache>/pclass/`, so re-scores cost nothing. Without the weights file the scan logs it and the rule decides alone.
- Window labels: `dark` (every frame's mean grey < 20) and `sand` (WP10's frame model, fitted in 00017 only, against
  labels the user has not checked) from the scanned windows' frames at 1280 px; labels only; sand and dark windows
  are left out of the hq spans; descriptions say "View blocked by sand for x s." / "Lens dark for x s.". `seafloor`
  stays empty: the user found WP10's class on whale backs and haze (2026-11-23); WP9's sheets `scan_d2_seafloor_01`
  and `scan_d2_seafloor_fired_01..02` and section 11.7's seafloor rows rest on the same withdrawn spans (void), and
  `evaluate.py --dep d2` marks seafloor spans "labels withdrawn 2026-11-23".
- Clips (user 2026-11-23): a clip starts at the first active window and ends only after >= 30 s with nothing active
  (`--guard`), never across tag in air / drone flight or the lens out of the water for more than 20 s (a breath does
  not end a clip); 2 s buffer each side (`--pad`), clipped at the file ends and at those hard windows; `hq_spans`
  keep their own tight cut; descriptions say "x s of y s without anything detected". Back-to-back files: one clip per
  file, marked `continues_in` / `continues_from` when a clip runs into the next file (files within 3 s by their
  clocks): 0 such joins on either deployment.
- `evaluate.py`: whales seen from the drone through the surface (frames labelled air) are not activity the scan can
  keep (5 frames of CC-09-85 00010).

The two steps separately (sensitive; "before" = no veto, guard 9 s, which reproduces the WP9 clips: 86.5 / 161 and
16.6 / 61):

| deployment, run | kept min | clips | median s | longest s | clips < 10 s | quiet s inside clips | whale windows | hq windows | hq windows < 1 m |
|---|---|---|---|---|---|---|---|---|---|
| 2025-02-03 before | 86.5 | 161 | 19.0 | 258 | 48 | 1,123 | 3,002 | 723 | 53 |
| 2025-02-03 veto only | 84.1 | 155 | 20.0 | 258 | 44 | 1,038 | 2,897 | 647 | 27 |
| 2025-02-03 guard only | 99.7 | 94 | 24.5 | 607 | 18 | 1,911 | 3,002 | 723 | 53 |
| **2025-02-03 veto + guard (default)** | **94.3** | **100** | 23.0 | 607 | 19 | 1,646 | 2,897 | 647 | 27 |
| 2025-02-03 strict | 83.2 | 93 | 26.0 | 286 | 11 | 1,371 | 2,621 | 635 | 18 |
| CC-09-85 before | 16.6 | 61 | 10.0 | 89 | 29 | 479 | 389 | 174 | 71 |
| CC-09-85 veto only | 13.0 | 46 | 10.5 | 75 | 21 | 332 | 248 | 132 | 41 |
| CC-09-85 guard only | 23.1 | 35 | 12.0 | 225 | 16 | 867 | 389 | 174 | 71 |
| **CC-09-85 veto + guard (default)** | **17.6** | **27** | 20.0 | 149 | 8 | 609 | 248 | 132 | 41 |
| CC-09-85 strict | 11.5 | 22 | 15.5 | 118 | 8 | 375 | 199 | 114 | 28 |

The orchestrator's cross-check of the guard alone (about 93 clips / 99.9 min and 35 / 23.1) agrees (94 / 99.7, 35 /
23.1). Clips by best tier (default): 2025-02-03 hq 52, ok 26, small 17, fish 3, low 2; CC-09-85 small 10, ok 9, hq 8.
Clips with `advanced_ok`: 89 -> 52 and 32 -> 8; hq span time 758 -> 675 s and 188 -> 147 s.

Veto, by WP10's labelled boxes (tier of the window holding the box): the 24 near jacks of CC-09-85 00023 (fitted on:
not clean) were low 12 / ok 9 / hq 3, now fish 18 / small 6 (none ok or hq); the breath boxes labelled blow /
surface / mirrored back (not fitted on): CC-09-85 hq 22 / ok 21 -> none 42 / small 1; 2025-02-03 hq 24 / ok 2 ->
none 21 / small 3 / hq 1 / fish 1. Mid-sized whale windows (300-500 px box): 1,289 -> 1,267 and 99 -> 38.

Windows the veto made inactive (active without the classifier, not with it; corrected after review,
`wp11b_measures.py` `veto_effect_*`): 59 on 2025-02-03, of them 42 still inside a kept clip (quiet time) and 17 no
longer kept; 71 on CC-09-85, 10 inside a clip and 61 no longer kept. The flag `vetoed_other_only` (78 / 71) also
counts 19 windows on 2025-02-03 that the shallow rule had already emptied before the veto. 60 of the 149 flagged
windows at random looked at by me (`scan_veto_dropped_01..05.jpg`, seed 20261122, AI by-eye): 3 held a real whale:
the whale passing the floating tag in 2025-02-03 00012 (it loses its whale label on 3 windows, 14:39:37-40, the
classifier calling its fin / body fish or other, but stays inside its clip) and one possible fin at the frame edge
(00007 12:07:09); 57 are blows, bubbles, foam, the surface from below or the back mirrored in it (most at < 1 m), or
empty water.

Label scores (`evaluate.py`, `results/eval.json`, `eval_d2.json`; before -> default):

| measure | 2025-02-03 | CC-09-85 |
|---|---|---|
| frames with another whale kept | 219 -> 220 / 221 | 27 -> 29 / 31 |
| any activity kept (air frames excluded) | 258 -> 264 / 279 | 36 -> 38 / 46 |
| empty underwater frames kept (veto only / guard only) | 53 -> 72 / 542 (45 / 85) | 31 -> 20 / 292 (10 / 47) |
| hq or ok frames in a tier ok or better | 92 -> 92 / 93 | 13 -> 13 / 20 |
| tier ok or better frames graded low or whale-free | 44/136 -> 36/128 | 13/26 -> 1/14 |
| tier hq frames graded low or whale-free | 8/44 -> 0/37 | 1/11 -> 0/10 |
| hq frames in tier hq | 11/11 -> 11/11 | 2/2 -> 2/2 |

Dark / sand against the labels' `view` (CC-09-85): 8 of 11 frames labelled dark flagged, 0 false of 409; sand 9 of
10 flagged, 4 false of 410 (in-sample: the sand model was fitted in 00017). Windows: dark 208 (0 kept, 0 hq), sand 242
(31 kept by the guard's quiet time, 0 hq); 2025-02-03: dark 0, sand 3 (0 kept).

Cost: the picture-class stage (decode + DINOv3 for frame classes on every scanned window frame + crops for the
classifier) held the GPU 2,135 s for 31,546 frames on 2025-02-03 (0.13 GPU-h per video hour) and 1,183 s for 10,740
4K frames on CC-09-85 (0.14 GPU-h per video hour, decode-bound at ~9 frames/s); the classifier alone is a small part
(frames with a large box: ~6,000 and ~800). Run as `flock ... timeout 900 nice -n 10 uwid scan ... --no-gpu-lock
--gpu-budget-min 13` repeated until `summary.json` says complete (`jobs/run_wp11b.sh`): 3 passes per deployment.

Sheets re-made with the classifier's decision drawn (`(clf)` = decided by the classifier, grey = other):
`scan_d2_hq_top / border / rejected_01..04`, `scan_hq_top / border / rejected_01..04` (windows re-selected: the hq
set changed), `scan_hq_band_*`, `scan_d2_hq_band_*`, `scan_midsize_01..04`, new `scan_veto_dropped_01..05`. The
top-12 hq windows of CC-09-85 are now all real near whales (were 5 breaths of 12).

Still wrong / not tested: the whale beside the floating tag (00012, 14:39:37-40) loses its whale label on 3 windows
(known error, nothing tuned on it; it stays inside its clip); the
classifier's large-fish result rests on one school of jacks it was fitted on; 41 (CC-09-85) and 27 (2025-02-03) hq
windows under 1 m remain (WP10 counts 34 / 27 real shallow whales there, not checked window by window); sand is
fitted on one recording and unchecked by the user; seafloor has no detector; the 30 s guard makes clips mostly quiet
water in places (2025-02-03: 1,646 of 5,658 kept seconds without a detection).

After review (2026-11-23): sheets draw only what the decision uses (a fish box only at or above the fish threshold or
when the classifier turned a whale box into fish; "(clf)" from `boxclf_class`); clip descriptions say "up to n fish
boxes" only when n >= 1 and give the clip's length in seconds including a part window at a file's end (e.g. 00005
1777.0-1781.4 s: 4.4 s); `scan_veto_dropped_01..05`, `scan_hq_*`, `scan_d2_hq_*`, `scan_hq_band_*`,
`scan_d2_hq_band_*`, `scan_midsize_*` re-made with the same samples. The orchestrator checked the hq windows still
under 1 m: 26 of 27 on 2025-02-03 show a real near whale (1 doubtful, 00006 11:22:32); all 41 on CC-09-85 lie in the
two near-whale passes at 16:01 and 16:07-16:08.
