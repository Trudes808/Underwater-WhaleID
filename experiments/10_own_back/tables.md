# 10_own_back (WP14): the tagged whale's own back against a calf in front of it; bubbles after a breath

Run 2026-11-23 evening. Study only: `src/uwid/` untouched (the tracker run WP13 imports it). All truth is the user's
two statements of 2026-11-23 plus AI by-eye looks (mine), named as such. 2025-02-03 00005 604-629 s was a held-out
stretch until tonight; it is a development case from here on. Nothing was tuned on other test stretches; the one
threshold used (tau 0.5) was set before looking at the candidates, on the confirmed case only.

## 1. Calf against own back

### What refused the calf (00005, tracker run 607-627 s)

Not the patch classifier. The tracker's own back is the SAM 3 back object (object 0, prompted from the bottom 25 % of
the frame); the calf swims along the back's horizon, inside that prompt band, and the back object swallows it: the
two calf tracks lie 99.8 % and 99.9 % inside object 0 (median over 40 / 17 sampled frames), but only 1.6 % and 1.3 %
inside the patch classifier's `back` class (WP5b `clf_w1280_h0_all.pt`). Object 0 grows from 7 % to 10 % of the frame
while the calf is there. The patch map draws the back as a thin band (plus scattered false `back` blobs in open water,
WP5b's known 8.7 %), not too tall.

### The template (proposed code: `ownback.py`, tests `test_ownback.py`, 4 passed)

Per recording: patch called `back` by the WP5b head in >= tau of the recording's under-water on-whale frames (every
10th scanned window: 1,539 frames of 2025-02-03, 530 of 2025-01-02), grown by 1 patch. Own back = SAM back object
inside the template. Frames are the ones the scan chose; the maps needed one 3-minute GPU piece (the scan's cache
holds class shares per frame, not maps).

(a) Labelled back pixels kept inside the template (reference masks):

| tau | 2025-02-03 (14 frames) | 2025-01-02 (29 frames) | dev clip (31 frames; template from those same frames, the head was fitted on them) |
|---|---|---|---|
| 0.3 | 99.7 % (813,387 / 815,627 px) | 98.9 % (16,738,511 / 16,921,965) | 88.4 % (3,021,005 / 3,416,147) |
| 0.5 | 99.0 % (807,077) | 97.5 % (16,501,159) | 74.7 % (2,553,385) |
| 0.7 | 98.6 % (804,266) | 84.7 % (14,332,011) | 62.9 % (2,147,568) |

The dev clip's back moves (competitive group, the whale rolls), so a fixed template keeps less of it.

(b) 00005 604-629 s (tau 0.5): 25 windows with a scan whale box; median 29 % of a box inside the template; 25 / 25
boxes pass the back test (< 60 % on the template). The two tracker objects: inside the SAM back 0.998 / 0.999 -> inside
the proposed own back 0.377 / 0.185, both under the 0.6 `on_back` limit: both would get numbers.

(c) Risk. 21 stored online runs (WP11 runs + the first WP13 spans; 48 tracked objects): only those two objects are
`on_back` today and only those two would flip. The SAM back object's area outside the template: median 0.2 %, 90th
percentile 9.2 %, 99th 32.3 % of it (1,374 sampled frames; 46 frames above 20 %, most of them the calf frames of 00005).
Template drift (templates of 300 s blocks against the whole recording, IoU): 2025-02-03 00004-00006 and 00008 >= 0.83;
00007 0.57, 00009 0.67, 00010 0.64, 00011 0.49, 00012 0.64 (min over blocks); 2025-01-02 00017 0.21 (rolling on the
bottom), 00023 0.81, 00010 / 00011 0.64 / 0.49 for the larger back. Up to 43 % of a block's back lies outside the
dilated whole-recording template (00011). So the back does move within a recording; a per-block template (300 s) is
the safer form, and the drift is where own-body numbers could appear. Not seen in the 21 runs; not measured beyond
them.

(d) Other stretches. Without a tracker run the SAM back object is unknown, so the count uses its prompt band: scan
whale objects of >= 3 windows whose best box has >= 60 % of its height in the bottom 25 % of the frame in at least half
the windows: 17 (2025-02-03 16: 00005 x6, 00006 x2, 00007 x2, 00008 x2, 00010, 00011, 00012 x2; 2025-01-02 1: 00023
1017 s). All 17 pass the template test. By my look on the sheet all 17 are other animals (calf or whales low in the
frame, a white pectoral fin beside the floating tag in 00012), none the mother's own body; the user decides.
Sheet: `review_sheets/underwater/ownback_candidates_01..03.jpg` (plain frame | scan box red, proposed own back purple).

## 2. Bubbles after a breath (one confirmed case: no rule)

Tracked objects (first second; the run's own WP5b patch map):

| object | truth | patch map under the outline | contrast to ring (grey) | IoU outline 1 s later | area ratio over 1 s |
|---|---|---|---|---|---|
| 00024 no. 2 (125.2 s) | bubbles (user) | whale 94 %, bubbles 0 % | +10.5 | 0.01 | 1.8 |
| 00024 no. 3 (125.2 s, 7.6 s) | bubbles (user) | whale 56 %, water 44 % | +25.1 | 0.48 | 1.5 |
| 00004 622.8 s | whale | whale 99 % | -25.1 | 0.00 | (leaves frame) |
| 00005 1566.3 s | unsure (WP11c) | whale 91 % | +6.4 | 0.53 | 1.6 |
| 00005 1575.8 s | whale | whale 89 % | -6.9 | 0.67 | 1.1 |
| 00006 912.7 s | whale | whale 90 % | -3.1 | 0.06 | 11.0 |
| 00007 1471.2 s | whale (calf) | whale 98 % | -28.7 | 0.42 | 1.4 |

The patch map calls the bubbles whale; outline change over a second overlaps with real whales. The only measure that
splits these 6 is "paler than its surroundings" (2 / 2 bubbles positive, 4 / 4 whales negative, the unsure one
positive), which is also what a white pectoral fin or a pale belly looks like.

Scan whale objects beginning within 10 s after a lens-out run: 24 (2025-02-03 19, 2025-01-02 5). Paler than around
the box: 6 (2025-02-03 00007 1351, 00012 1553, 00012 1560, 00005 1566; 2025-01-02 00024 125 = the confirmed case,
00023 400); patch map bubbles / surface / air >= 20 % under the box: 5. By my look on the sheet: 00012 1553 / 1560 are a
real whale's white pectoral fin, 00005 1566 a pale whale in haze, 00007 1351 a box over the blow and surface with a
whale at its edge, 00023 400 a pale streak far away (bubbles or a far whale: unsure). So "paler" flags 2 real fins
out of 6. Candidate sheet for the user: `review_sheets/underwater/bubbles_candidates_01..02.jpg` (all 24, flagged
first; plain frame | scan box, the measures in the caption).

## Files

`common.py`, `sample.py` (frames, CPU), `maps.py` (GPU, WP5b head), `runs.py`, `measure.py`, `bubbles.py`,
`sheets.py`, `ownback.py` + `test_ownback.py` (proposed code), `look_run.py` (diagnostic view), `results/measure.json`.
Cache `underwater_cache/ownback/` (frames, maps, runs_tracks / runs_back parquet, drift.csv, bubbles_*.csv).

## WP17 (2026-11-24): the template behind the tracking flag `own_back_template` (off by default), replayed

**Code.** `src/uwid/ownback.py` (template, block frames, block cache, run file `own_back.npz`); `trackstage.run_span(...,
own_back_template=False, suffix="")` builds the template of the span's 300 s block from the scan's window table
(every 10th scanned under-water on-whale window, frame_0) with the tracker's own patch classifier (WP5b, frames
resized to 1920 px as in `patch_pass`), caches the block's class maps under
`underwater_cache/tracks/<deployment>/<recording>/ownback/block_<b>.npz`, stores the template with the run, and gives
it to `track_online(..., own_back=)` (start-box back test: back object inside the template) and, through
`tracks.read_run` -> `Tracks.own_back`, to `tracks.clean` (`on_back` role: share inside back object & template >= 0.6).
Fewer than 5 frames in a block = no template = behaviour as flag off (recorded in meta `own_back`). Flag off: no new
meta key, `own_back(mask, None)` returns the same array, the clean-up reads no template. The scan's cache holds class
shares per frame, not maps, so the tool computes maps for ~30 frames per block (52 blocks: 1 lock piece, ~3 s each).
Tests: `tests/test_ownback.py` (4; template, block frames + cache, clean-up with / without / full template = off);
suite 86 passed (82 before). The experiment's `ownback.py` now imports from the package.

**Parameters frozen** as in WP14: share 0.5, grown by 1 patch, 300 s blocks, every 10th window, `on_back` limit 0.6 and
the box test's 0.6 unchanged. They were chosen in WP14 on the labelled back pixels of 2025-02-03 (14 mask frames),
2025-01-02 (29) and the dev clip (31) and on the one confirmed calf stretch; **the replay below is on the same two
deployments** (no new deployment), so it shows what the flag does, not how well it carries.

### (a) Offline over WP13's 135 stored spans (no re-tracking)

Start boxes (>= the start size) with a stored back object at the previous frame: 1,618. Dropped by the back test
today: 0. With the flag: 0. Newly passing: 0, so no span was re-tracked. Clean-up roles changed: 0 (no object had
the `on_back` role, and the own back can only shrink). Templates: 133 of 135 spans (2 in 2025-02-03 00012's last
block, which has 2 under-water on-whale windows: the tag floats). Flag-off clean-up with the new code reproduces
WP13's number count on 135 of 135 spans.

### (b, c) The calf stretch and the 16 candidate stretches, flag on (`*_online_ownback`)

The candidate 00005 596-667 s holds the calf stretch and was replaced by it (607-627 s, the span of the stored run).
Flag-off result: the stored run (calf); a `_ownback_off` run where a start box passes only with the flag on and no
stored run exists (00007 262-265, 00008 5-16); elsewhere no start box differs, so the tracker saw the same starts and
flag off = the flag-on run cleaned without the template. Lock-held tracking 1,477 s (24.6 min) of the 40 min budget:
nothing left out.

| stretch (2025-02-03 unless noted) | template | numbers off -> on | `on_back` objects off -> on | new numbers (share in back object / in own back, lasts) |
|---|---|---|---|---|
| 00005 607-627 calf | yes (30 frames, 8.9 % of frame) | 0 -> 2 | 2 -> 0 | no. 1 calf (0.998 / 0.347, 19.8 s); no. 2 dark wedge at the left edge (0.999 / 0.148, 8.4 s) |
| 00011 1053-1056 (unclear) | yes | 1 -> 1 | 0 -> 0 | none: the dark wedge already has number 1 with the flag off (0.028 in the back object, 1.8 s) |
| 00012 1553-1558 (unclear) | **no** (2 frames: tag floating) | 0 -> 0 | 0 -> 0 | none: 7 of 8 starts refused as bubbles / surface / air, both ways |
| 00007 262-265 | yes | 0 -> 0 (run off too) | 0 -> 0 | none |
| 2025-01-02 00023 1017-1020 | yes | 1 -> 1 | 0 -> 0 | none |
| 00005 589-594 | yes | 0 -> 1 | 1 -> 0 | no. 1 (0.999 / 0.368, 4.2 s) |
| 00005 1682-1687 | yes | 1 -> 1 | 0 -> 0 | none |
| 00006 1449-1454 | yes | 0 -> 0 | 0 -> 0 | none |
| 00008 965-971 | yes | 0 -> 1 | 1 -> 0 | no. 1 (0.986 / 0.289, 5.2 s) |
| 00005 1520-1528 | yes | 1 -> 1 | 0 -> 0 | none |
| 00007 1335-1344 | yes | 1 -> 1 | 0 -> 0 | none |
| 00008 5-16 | yes | 0 -> 3 (run off too) | 3 -> 0 | no. 1 (0.966 / 0.501, 0.3 s), no. 2 fluke shape behind the hump (0.999 / 0.068, 2.3 s), no. 3 (0.998 / 0.000, 6.8 s) |
| 00012 1508-1519 | **no** (as above) | 0 -> 0 | 1 -> 1 | none: the shape stays inside the back object |
| 00005 360-372 | yes | 1 -> 1 | 0 -> 0 | none |
| 00006 1401-1413 | yes | 1 -> 1 | 0 -> 0 | none |
| 00010 1515-1530 | yes | 0 -> 1 | 1 -> 0 | no. 1 (0.997 / 0.572, 14.8 s) |
| 00005 1534-1558 | yes | 0 -> 1 | 1 -> 0 | no. 1 (0.988 / 0.121, 23.2 s) |

New numbers with the flag on: 9 in 6 stretches (7 last >= 3 s, the calf's two included); every one lies 0.97-1.0
inside the back object and 0.0-0.57 inside the own back. Of the 10 `on_back` objects, 9 become numbers and 1 stays
(00012, no template); 10 of the 17 stretches had none. (Counts corrected in WP18: first written as "7 stretches",
"6 last >= 3 s" and "10 objects become numbers".)

### (d) The risk, as the tool behaves (not decided: the user's call)

- The calf stretch: besides the calf, the dark wedge at the left edge (an earlier labeller note called it probably
  the tagged whale's own flipper) now gets number 2 for 8.4 s.
- 00008 5-16 s: three numbers on shapes at the back's horizon; number 2 is a fluke shape behind the hump. Whether these
  are another whale or the tagged whale's own fluke / body is not something the tool knows.
- 00011 1053-1056 s (dark wedge beside the back): number 1 for 1.8 s with the flag OFF as well; the flag changes nothing.
- 00012 1553-1558 s (white flipper at the surface): no template in that block (the tag is floating), so the flag does
  nothing; no number either way (starts refused as surface / bubbles).
- Template area 8-13 % of the frame on 2025-02-03 and 31-41 % on 2025-01-02 (the back is taller there).

### (e) Frame store vs decoding, one 1080p stretch (00011 1053-1056 s, flag on, 68 frames)

Masks: 0 of 68 frames differ (all objects); table identical; endings and refused starts identical; numbers 1 = 1;
tracking 16.1 s decoded vs 15.7 s from the store.

Sheets (`review_sheets/underwater/`, tool's decision only: numbered outlines, own back purple as the tool uses it):
`ownback_replay_calf_01.jpg` (flag off above on, the same 8 frames), `ownback_replay_calf.mp4` (flag on, 2.0 MB),
`ownback_replay_candidates_01..04.jpg` (one row of 6 frames per stretch, the two unclear cases first). No (a) sheet
(nothing re-tracked).

Files: `wp17.py` (blocks, offline, track, track_off), `wp17_report.py`, `wp17_sheets.py`, `results/wp17.json`; cache
`underwater_cache/ownback/wp17/` (offline_wp13.csv, stretches.csv, lock_seconds.json), runs
`underwater_cache/tracks/2025-2-3 Mini-mother 11-96/<recording>/*_online_ownback[_off|_store]/` and the 2025-01-02
one; block maps `.../<recording>/ownback/`.

## WP18 (2026-11-24): own-back template on by default; far specks dropped from whale outlines

User (2026-11-24, on the WP17 sheets): "Yes you may make the switch default."; number 2 in the calf stretch "is a
calf but it is quite far in distance"; and the fault "the last few seconds show something very small in distance
getting picked up as part of the 1 track".

**Default on.** `trackstage.run_span(own_back_template=True)`, `trackstage.run_deployment(own_back_template=True)`
(passed to every span), `uwid tracks ... --no-own-back-template` for the old behaviour (only the `tracks` subparser
and `cmd_tracks` changed in `cli.py`). A block without a template still behaves as before.

**The 135 stored WP13 runs** now carry the template a default run would store (no re-tracking, masks and tables
untouched): `own_back.npz` added to 133 folders (2 spans in 2025-02-03 00012's last block have no template: only the
meta key, `why_none`), meta key `own_back` added to all 135 (template info as `run_span` writes it). Undo list:
`underwater_cache/tracks/wp18_ownback_added.csv` (folder, npz added, meta backup); meta backups in
`underwater_cache/ownback/wp18/meta_backup/`. Numbers per span, before and after: identical on 135 / 135 (157
before links, 149 after, both as in WP13). The 17 replay stretches keep their `*_online_ownback` folders; their
default-result folders are listed in `underwater_cache/tracks/extra_stretches.csv` (deployment, recording, t0, t1,
folder, note); their number counts are unchanged (16 = 16).

**Speck rule** (`tracks.despeckle`, constants `SPECK_MAX_SHARE` / `SPECK_MIN_GAP_PX` at the top of `tracks.py`):
per frame and whale object, the largest 8-connected piece is kept; another piece is dropped only when its area is
under 5 % of the object's area in that frame AND its gap to the largest piece is over 40 px at store size (1280 x
720). Applied in `Tracks.mask` when `Tracks.despeck` is on, which `trackstage.load_online` sets; stored masks stay
raw (`Tracks.raw_mask`); roles, numbers and the link rules are computed on the raw outlines (`trackstage.link`
switches the rule off while it links). Values: the orchestrator's starting values, kept after the dev clip's stored
runs only (2025-12-20 CG, 4 runs, 33 numbers: the dropped pieces were scattered speckle in bubble clouds and far thin
fragments; the pieces of whales split by a flipper or the frame edge were kept), before the test deployments were
looked at. Number labels: the video renderer already put them on the largest piece; the WP17 sheet code anchored
them on the whole mask, which put "1" on the speck.

| runs | object-frames | with >= 2 pieces | changed (a piece dropped) | >= 2 pieces, all kept | dropped share of the outline per changed object-frame: median / max | numbers touched |
|---|---|---|---|---|---|---|
| 2025-12-20 clip (dev; 4 stored runs) | 7,942 | 2,526 | 1,214 | 1,312 | 0.74 % / 26.8 % | 26 of 33 |
| 2025-02-03 (118 hq spans + 15 stretches) | 22,964 | 2,596 | 788 | 1,808 | 0.41 % / 7.3 % | 66 of 153 |
| 2025-01-02 (17 hq spans + 2 stretches) | 4,367 | 1,020 | 49 | 971 | 0.05 % / 4.7 % | 11 of 20 |

(numbers = numbers per run before links.) Numbers identical on every stored run checked: 135 hq spans (157 / 149),
17 stretches (16), the dev clip's online run (5); the speck rule cannot change them, since they are computed before it.
Sheets: `speck_fix_changed_01.jpg` (12 changed object-frames, seed 20261124, before | after; most dropped pieces are
a few pixels), `speck_fix_kept_01.jpg` (12 object-frames with 2+ pieces all kept: whales cut by a flipper, the frame
edge or the back). Calf: `ownback_replay_calf_v2.mp4` (1.9 MB), `ownback_replay_calf_v2_01.jpg` (last 4 s, v1 above
v2: the far speck and its "1" are gone).

Tests: `tests/test_ownback.py` +3 (default on and opt-out through `uwid tracks`, the speck rule keeping large or close
pieces, `load_online` applying it with numbers unchanged); suite 90 passed. Script `experiments/10_own_back/wp18.py`
(dev, retro, check, stats, extra, calf); results `results/wp18_dev.json`, `wp18_check.json`, `wp18_speck.json`.
