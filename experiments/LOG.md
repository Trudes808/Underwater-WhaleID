# Experiment log

Newest entry first. One entry per finished piece of work:

    ## YYYY-MM-DD HH:MM  <experiment folder>: <one-line result>
    What was run, on which data (videos, number of frames), the numbers, the verdict, the caveats, output paths.

Rules of evidence are in `CLAUDE.md` (split by video, AI by-eye labels are named as such, one run is one run).

## 2026-11-24 22:00  12_export (WP19 final run): dev clip tracked (10 spans, 46 -> 45 numbers in 45.5 s, 16 lasting >= 3 s, 1 link, 82 candidates "not lone"); bundle with all three deployments: 162 tracked spans + clips, 173 MB; checker 0 problems, 162 / 162 clips frame-exact, IoU min 0.979 / median 0.992
- Dev clip: `uwid tracks 2025-12-20_CG` (default settings as of WP18, nothing tuned), one locked piece 21:44-21:52,
  416 GPU s. Fault found first: trackstage looked for the dev clip's scan files under its real name (with a space)
  while the scan stores them with `_`, so no start boxes would have been found; fixed with `trackstage.scan_key`
  (windows table and detector cache lookup only; test `test_scan_key_for_names_with_spaces`). Old dev run
  `0024.00-0036.00_online` moved aside as `*_pre_wp19`. No sensor log: blocked frames by the air-share rule (100
  frames). The own-back template was not built for any dev span ("fewer than 5 frames in the block"). Endings:
  appearance drift 12, shape jump 4, area jump 4, swallowed another track 2, re-found 1; starts refused 14
  (surface / bubbles 11, box classifier other 2, fish 1). With many whales in view the purity rule ends numbers
  early and the link rule never finds a lone break: many short numbers (median life 1.07 s, 21 of 45 under 1 s).
- Export changes: group `hq` = what `uwid tracks <deployment>` runs (on the two test deployments exactly the 135
  WP13 spans, checked folder by folder; the dev clip adds 10); `extra` from `tracks/extra_stretches.csv` with its
  `note` in the index and the track file; `links[].others_numbers`; spaces in path components replaced by `_` (dev
  clip); dev clip windows table found. Format document: clock facts (2025-02-03 burned-in clock = sensor time + 2.03
  s; dev clip about two hours later than the field notes, open), caution on the 2025-02-03 `.m4v` copies (dropped
  frames, drift up to 1.5 s, washed-out levels: every time refers to the `.mov`). README command block: `uwid
  tracks` and `uwid export` with their options.
- Final export (JSON / CSV rewritten with the WP18 clean-up; existing clips kept, 10 new): 3 deployments, 19
  recordings, 162 spans (hq 145, extra 17), 208 whale numbers; 173 MB (clips 128 MB, track JSON 34 MB).
- Checker: (i) `/usr/bin/python3 -I`: 0 problems; (ii) 162 / 162 clips decode to the JSON frame count and
  1280 x 720; 20 picks seed 20261124 (other picks than the first run: the span list changed): IoU min 0.9793 (a
  small outline, 8,180 px), median 0.9921. Picture `review_sheets/underwater/export_check_02.jpg` (3 dev-clip tiles
  first). 91 tests pass. Tables: `experiments/12_export/tables.md` (final run section).

**Reviewer's note (main thread, 22:02).** Heading time matches the clock (entry 22:00, `date` 22:01). My own run of
the tests: 91 passed. My own walk of the final bundle with `/usr/bin/python3 -I`: 3 deployments, 162 spans (hq 118 +
17 + 10, extra 16 + 1), 208 numbers (144 + 19 + 45), every track, clip and poster path exists, 19 windows files; the
17 extra spans carry their notes. Dev clip from the track files: 45 numbers, median life 1.07 s, 16 lasting >= 3 s
(22 under 1 s by first-to-last time; the entry's 21 counts outlined time). The GPU piece ran 21:44-21:52 under the
lock with `timeout 900`. `export_check_02.jpg` (12 frames drawn from the bundle alone, 3 from the dev clip): outlines
on whales in all 12; on the first dev tile a pale whale in mid-frame has no outline, so in crowded frames not every
whale is outlined (AI by-eye). Bundle 173 MB, cache 13 GB, nothing of ours running.
Two things the dev-clip run showed that were not known before: `uwid tracks` could not find the scan files of a
recording whose name has a space (fixed, `trackstage.scan_key`, with a test: the dev clip was the only such name, so
no earlier result is affected); and without a sensor log the own-back template is never built, so on the dev clip the
default of WP18 has no effect. The second is a limit, not fixed: it is written into `HANDOFF_WEB.md`.
`HANDOFF_WEB.md` (repo root) is mine; I corrected its polygon figure to the final run's (IoU 0.98 or better) and its
counts to the final bundle.

## 2026-11-24 21:42  12_export (WP19): web export bundle `uwid-web-export` v1: 3 deployments, 19 recordings, 152 tracked spans with outlines as polygons + 152 plain H.264 clips (frame j = track frame j, 152 / 152 checked), 154 MB; polygons vs masks IoU min 0.986, median 0.994
- New `src/uwid/export.py` + `uwid export [deployment ...] [--force] [--no-clips] [--out]` (cli: own subparser
  only); test `tests/test_export.py` (synthetic video + hand-made run: fields, polygons in 0..1 rasterise back at IoU
  > 0.9, clip frame count). Format contract `docs/EXPORT_FORMAT.md` (copied into the bundle as README.md): index.json,
  per deployment stretches.json + windows/<recording>.csv (scan table under plain names, sensor / model / rule
  marked), tracks/<group>/<recording>/<t0>-<t1>.json (frames, whale numbers with per-frame box + polygons, small
  tracks, the back every 5th frame, blocked, endings, every link candidate, refused starts), clips/... .mp4 (1280 x
  720, H.264 yuv420p, CFR = track fps, +faststart) + poster .jpg. Coordinates = fractions of the frame; numbers
  are per span (purity rule), not animals. Outlines from `load_online` + `link` only.
- Groups: hq = the 135 WP13 spans; extra = the 17 `*_online_ownback` runs (extra_stretches.csv not there yet).
- Run 2026-11-24 21:34-21:40 (CPU, 5.3 min): bundle `underwater_cache/export/web_v1/` 154 MB (clips 112 MB, track
  JSON 31 MB, windows CSV 3.4 MB, posters 6.8 MB).
- Checker `experiments/12_export/check_bundle.py`: (i) `/usr/bin/python3 -I`, standard library: 0 problems over 152
  track files (27,331 outlined frames, 29,755 polygons); (ii) `.venv`: decoded frame count and size = JSON for 152 /
  152 clips; 20 random picks (seed 20261124) IoU of polygons rasterised at 1280 x 720 vs `tr.mask`: min 0.9862,
  median 0.9939. Review picture `review_sheets/underwater/export_check_01.jpg` (12 frames from the bundle only).
- Media facts (`experiments/12_export/media_facts.py`, header codec tags): 2025-02-03 `.mov` HEVC hev1 1080p (no
  browser), its `.m4v` copies H.264 (yes); 2025-01-02 `.mov` HEVC hvc1 4K (Safari only); scan clips in
  `underwater_cache/clips/` 155 of 158 HEVC hev1; review videos and export clips H.264 yuv420p (all browsers).
  The `._*` files beside the 2025-02-03 recordings are macOS resource forks, not videos.
- Caveats: made before WP18's final clean-up change (re-run asked for later: JSON is always rewritten, clips are
  kept unless --force); the dev clip's window table is named with `_` for the space in its file name (export
  looks for that too); holes in masks are not exported; 90 tests pass. Tables: `experiments/12_export/tables.md`.

**Reviewer's note (main thread, 21:44).** Heading time matches the clock (entry 21:42, `date` 21:43). Read with
`/usr/bin/python3 -I` by my own short script, not the agent's checker: `index.json` lists 3 deployments, 19
recordings and 152 spans (2025-02-03 118 hq + 16 extra, CC-09-85 17 + 1, dev clip 0), and every track, clip and
poster path it names exists; the calf stretch's track file has numbers 1 (607.2-627.0 s, 592 frames) and 2
(613.7-622.1 s), `own_back_template` true, and one piece per frame for number 1 in its last 90 frames (the WP18 speck
rule is in that file). A search of every JSON / CSV / MD in the bundle for position and photo terms finds them only
in the README's sentence on what is left out. `export_check_01.jpg` (12 frames drawn from the bundle alone): the
outlines sit on the whales in all 12 (AI by-eye). Bundle 154 MB, cache 13 GB.
Asked for in the final run: tracks for the dev clip's 10 hq spans (none were ever made under the frozen rules, and
it is the one clip with a large group), group `hq` defined by what `uwid tracks` runs, group `extra` from
`extra_stretches.csv`, numbers beside object ids in the link evidence, a caution on the `.m4v` copies (dropped
frames, drift up to 1.5 s, so they do not agree frame by frame with the tables) and the clock facts.

## 2026-11-24 21:37  10_own_back (WP18): own-back template on by default (`--no-own-back-template` for the old behaviour); the 135 WP13 runs carry it with numbers identical (157 / 149); far specks dropped from whale outlines on read (837 of 27,331 test object-frames, median 0.4 % of the outline), numbers unchanged everywhere
User 2026-11-24: switch may be default; the calf stretch's number 2 is a far calf; fix the far speck in number 1.
- Default: `trackstage.run_span` / `run_deployment` `own_back_template=True`; `uwid tracks --no-own-back-template`.
- WP13 runs: `own_back.npz` added to 133 of 135 folders (2 without a template), meta key `own_back` to all 135; undo
  list `underwater_cache/tracks/wp18_ownback_added.csv`, meta backups in `underwater_cache/ownback/wp18/meta_backup/`;
  numbers per span identical (135 / 135; 157 before links, 149 after). Replay stretches listed in
  `underwater_cache/tracks/extra_stretches.csv` (17, folders unchanged).
- Speck rule (`tracks.despeckle`, applied by `Tracks.mask` in runs loaded with `load_online`; stored masks raw, link
  rules on raw outlines): drop a piece only if < 5 % of the object's area AND > 40 px from the largest piece (values
  kept after the dev clip). Changed object-frames: dev 1,214 of 7,942; 2025-02-03 788 of 22,964 (dropped share median
  0.41 %, max 7.3 %; 66 of 153 numbers touched); 2025-01-02 49 of 4,367 (0.05 % / 4.7 %; 11 of 20). Object-frames
  with 2+ pieces all kept: 1,808 / 971 (whales split by a flipper, the back or the frame edge).
- Numbers identical: 135 spans, 17 stretches, the dev clip's online run.
- WP17 counts corrected (6 stretches, 9 of 10 `on_back` objects, 7 lasting >= 3 s).
Sheets: `speck_fix_changed_01.jpg`, `speck_fix_kept_01.jpg`, `ownback_replay_calf_v2.mp4`, `ownback_replay_calf_v2_01.jpg`.
Tests 90 passed. Tables: `experiments/10_own_back/tables.md` section WP18.

**Reviewer's note (main thread, 21:40).** Heading time matches the clock (entry 21:37, `date` 21:38). My own run of
the tests: 90 passed. My own recount with `load_online` + `link` over the 135 folders of `wp13_spans.csv`: 149
numbers after links, 0 spans differ from `wp13_numbers.csv`; `own_back.npz` is in 133 folders (the two spans of
00012's last block have no template). Defaults read in the code: `run_span` and `run_deployment` have
`own_back_template=True`, `uwid tracks --no-own-back-template` turns it off; the speck values 0.05 and 40 px are at
the top of `tracks.py` and are the ones I gave before any count was seen. On the calf stretch number 1 loses a piece
on 75 of its last 90 frames by my own count (`raw_mask` against `mask`), which is the fault the user saw.
Sheets (AI by-eye): `ownback_replay_calf_v2_01.jpg` shows the far speck and its label gone in the last 4 s with the
calf's outline unchanged; on `speck_fix_changed_01.jpg` the 12 changes are a few pixels each and I see no whale part
removed; on `speck_fix_kept_01.jpg` whales cut in two by a flipper, the back or the frame keep both pieces.
Limits: the rule cleans what readers see, not what the tracker stored or decided (numbers, links and endings still
use the stored outlines, by design); it applies to whale numbers only; a far small piece that is in fact part of the
same whale (a fluke tip showing behind another animal) would be dropped too, and none of the 12 sampled changes is
such a case. The surface outline of 00005 380 s (number 2, 0.9 s) is still there: the surface refusal was not part of
this.

## 2026-11-24 21:27  10_own_back, 10_mesh3d (user review): own-back switch approved as the default; the dark wedge is a far calf; a far speck joins number 1 at the end of the calf video (to fix); 3D work paused; hand-over to the web-app agent asked for
Written by the main thread from the user's message of 2026-11-24 evening (no run). The user looked at
`ownback_replay_calf_01.jpg`, `ownback_replay_calf.mp4` and the 3D trial report.
- "The yellow 2 dark wedge is a calf but it is quite far in distance so its ok if thats hard to tell." (number 2 of
  00005 607-627 s with the flag on: an animal, so the number is right. My by-eye reading had left it open between
  another animal and the mother's own flipper.)
- "The video with the switch on looks quite good. Yes you may make the switch default."
- "One very small issue is it looks like the last few seconds show something very small in distance getting picked
  up as part of the 1 track?" (the agent had noted the same: in the last frames number 1's outline takes in a far
  speck.)
- "We can put a puase in the 3d work for now, go ahead and add where we are at in the notes we may pick it up
  later." State written into PLAN.md under "Parked".
- "Once that last thing is ready please pack things up to be passed off to another agent who will be working on the
  web app and will be creating a new page for the video analysis".
Follow-up: WP18 (default + speck fix), WP19 (export bundle + `HANDOFF_WEB.md`).

## 2026-11-24 04:44  10_own_back (WP17): own-back template wired in behind `own_back_template` (off by default; flag off unchanged, 86 tests); replay: 0 of 1,618 WP13 start boxes affected; on the calf stretch the calf gets a number (19.8 s), and 9 new numbers in 6 of 17 stretches, incl. shapes the tool cannot tell from the tagged whale's own body
Code: `src/uwid/ownback.py`; `trackstage.run_span(own_back_template=, suffix=)` builds the span's 300 s block template
from the scan's window table with the tracker's WP5b patch classifier (cached per block), stores it with the run
(`own_back.npz`); `track_online(own_back=)` start test and `tracks.clean` `on_back` role use back object & template.
Parameters frozen from WP14 (share 0.5, +1 patch, 300 s, limit 0.6), chosen on the same two deployments as this replay.
Tests: `tests/test_ownback.py` (4 new), suite 86 passed.
- (a) 135 WP13 spans offline: 1,618 start boxes, 0 dropped by the back test today, 0 newly pass, 0 roles changed;
  flag-off clean-up = WP13 numbers on 135 / 135; nothing re-tracked.
- (b, c) 17 stretches tracked with the flag on (lock 1,477 s of 40 min; 2 extra flag-off runs): calf 00005 607-627
  0 -> 2 numbers (calf 19.8 s; dark wedge at the left edge 8.4 s); new numbers in 00005 589-594, 1534-1558, 00008
  965-971, 5-16 (3), 00010 1515-1530: 9 in all in 6 stretches, 7 lasting >= 3 s (calf included; 9 of the 10
  `on_back` objects become numbers, 1 stays; counts corrected in WP18), each 0.97-1.0 inside the back object and 0.0-0.57
  inside the own back. No template in 2025-02-03 00012's last block (tag floating): flag has no effect there.
- (d) unclear cases: 00011 1053-1056 numbered (1.8 s) with the flag off as well; 00012 1553-1558 no template, no
  number either way. Risk shown, not decided: the calf stretch's dark wedge and 00008 5-16's fluke shape behind the
  hump now get numbers.
- (e) frame store vs decoded, 00011 1053-1056 (1080p, 68 frames): 0 mask frames differ, table / endings / refusals /
  numbers identical.
Sheets: `review_sheets/underwater/ownback_replay_calf_01.jpg`, `ownback_replay_calf.mp4` (2.0 MB),
`ownback_replay_candidates_01..04.jpg`. Tables: `experiments/10_own_back/tables.md` section WP17. Flag stays off.

**Reviewer's note (main thread, 04:47).** Heading time matches the clock (entry 04:44, `date` 04:44). My own run of
the tests: 86 passed. Recomputed from `underwater_cache/ownback/wp17/` (`offline_wp13.csv`, `stretches.csv`,
`lock_seconds.json`) and from the 17 stored `*_online_ownback` runs read with `load_online` + `link`: 135 spans,
1,618 start boxes, 0 dropped by the back test with or without the flag, 0 roles change, 157 numbers both ways
(WP13's count before links); tracking 1,477 s; 15 of 17 stretches have a template (the two of 00012's last block do
not). Three counts in the entry and in tables.md are off by one against the agent's own table: the new numbers are
in 6 stretches, not 7 (00005 607-627, 589-594, 1534-1558; 00008 5-16, 965-971; 00010 1515-1530); 9 of the 10
`on_back` objects become numbers and 1 stays, not 10 and 1; "6 lasting >= 3 s" leaves out the calf (7 with it).
After the lone links the three numbers of 00008 5-16 are one number (10.2 s), so 7 new numbers as the tool shows
them. Flag off is the code path of WP13 by test and by the 135-span replay.
My look at the sheets (AI by-eye; the user has not seen them): with the flag on the calf holds number 1 for the
whole stretch. Of the other new numbers, 00005 1534-1558 s rises from the back line into a whole whale in open
water (23 s that were thrown away before) and 00010 1515-1530 s does the same (15 s); 00005 589-594 s and 00008
965-971 s are a low body with a flipper where the calf is in the calf stretch; 00008 5-16 s is a fluke shape at
the back's horizon; number 2 of the calf stretch is a tall dark wedge at the left edge that I cannot name (another
animal or the mother's own flipper). Dark wedges of the same kind get a number with the flag off too (00011
1053-1056 s, 00005 1520-1528 s), so the flag does not create that question, it adds one case to it. Also: 00005
360-372 s on `ownback_replay_candidates_04.jpg` is the stretch of my unnamed WP13 shape (365-367 s): six frames
show a small whale from behind going up to the surface, so that WP13 number was most likely a whale.
Limits: one mother-calf deployment carries every new number (CC-09-85: 1 stretch, no change); the template's values
were chosen on these deployments; most flag-off results are the flag-on run cleaned without the template (same
starts), only 3 are separate runs. The flag stays off until the user has seen the calf sheet and the wedge.

## 2026-11-24 03:55  10_mesh3d (WP15 phase 2): TripoSG on WP13's tracks of both test deployments: the frame rule keeps 28 frames of 13 of 79 numbers (3,191 of 4,158 judged frames fail first at the frame edge); arm A (whole outline) 18/28 meshes read as a whale, arm B (whales cut by the frame, 45 numbers) 15/45 whole whale, 23/45 a whale part, 7/45 invented; meshes do not separate numbers (AUC 0.63)
- Picks (`pick_frames.py`, fixed before any mesh): the 135 folders of `wp13_spans.csv`, shown numbers after
  `load_online` + `link`, numbers >= 3 s: 79 (2025-02-03 68, CC-09-85 11); frame rule S0 unchanged, every 5th outlined
  frame, 4,158 judged. First failing clause: frame edge 3,191, below back-box top 630, area < 3 % 126, sharpness 95,
  back outline 7, drawn small 5; passed 104. Numbers with 0 / 1 / 2 / 3 / >= 4 passing frames: 2025-02-03 56 / 2 / 1
  / 2 / 7, CC-09-85 10 / 0 / 0 / 0 / 1. Picks 28 (the 240 cap never bit). The reviewer's two numbers (00005 365-367
  n1, 1.70 s; 00005 375-389 n2, 0.93 s) are under 3 s: dropped by the life rule and by name.
- Arm B (coordinator's addition; `pick_partial.py`, fixed before any arm-B mesh): the 66 numbers with 0 arm-A frames,
  rule without the edge clause, one frame each (largest area): 45 picks, 21 numbers dropped (20 fail every frame at
  the back, 1 too small).
- Meshes: TripoSG fp16, seed 42, 50 steps, guidance 7, hierarchical decoder (as the smoke test); input cropped to the
  outline box + 16 px (TripoSG crops there itself); stored decimated to 100k faces (0.010 chamfer from full, = the
  resampling floor). Arm A 29-42 s per mesh (median 31.7), torch peak <= 5.28 GB, lock held 949 s; arm B 29-61 s
  (median 34.1), torch peak <= 6.14 GB, nvidia-smi <= 6.31 GB, lock held 1,715 s of 2,700; pieces <= 783 s.
- 4a AI by-eye calls (written to CSV first): arm A whale 18 / part 2 / invented 8 (2025-02-03 18 / 2 / 6, CC-09-85
  0 / 0 / 2); most "whales" are generic (fluke + spindle body), 3 with humpback-like flippers; invented = flat views
  from below with flippers spread (4), the CC-09-85 flank (2), fluke-on-blob (2). Arm B whale 15 / part 23 /
  invented 7 (2025-02-03 13 / 17 / 6, CC-09-85 2 / 6 / 1); "part" keeps the cut as a flat face. Breakdowns by
  elongation, size and long-axis angle in tables.md S5.3-S5.4.
- 4b (arm A, 2025-02-03): chamfer same number median 0.119 (24 pairs) vs different numbers 0.168 (50 of 301),
  AUC 0.63, 13 of 50 different pairs closer than the same-number median; whale-only 0.101 (15) vs 0.121 (20). Does not
  separate numbers.
- 4c IoU with the own outline: arm A median 0.84, 6/28 < 0.70; arm B 0.88, 5/45 < 0.70. The seen side is mostly kept;
  IoU does not say whale-like (the highest arm-A IoU is a blob).
- Verdict: single-frame meshes keep the outline and invent the rest; whale-shaped results come from clean views from
  below / behind, cut whales mostly give parts. Not usable as an individual shape signature; at most a rough
  whole-whale proxy for the minority of frames that show the whole animal.
Caveats: no ground truth; one seed; calls by one AI viewing; arm A is dominated by 4 numbers (15 of 28 meshes) and the
2025-02-03 deployment (26 of 28); 00006 898-909 n1 and CC-09-85 00024 n1 were also smoke-test animals.
Outputs: code `experiments/10_mesh3d/` (pick_frames.py, pick_partial.py, trial_report.py, jobs/run_trial.sh,
jobs/run_partial.sh), tables.md S5, results/trial_summary_{A,B}.json; cache `underwater_cache/mesh3d/trial/` and
`trial_partial/` (picks, judged frames, inputs, meshes, renders, eye_calls.csv, pairs.csv; 293 MB with the smoke
test); sheets `review_sheets/underwater/mesh3d_trial_a_best_iou_01`, `_b_random_01..02`, `_c_number_01..06`,
`mesh3d_trial_partial_01..04`.

**Reviewer's note (main thread, 03:58).** Heading time matches the clock (entry 03:55, `date` 03:55). Recomputed
from `underwater_cache/mesh3d/trial/` and `trial_partial/` (`trial_table.csv`, `eye_calls.csv`, `pairs.csv`): arm A
28 meshes from 13 numbers (12 + 1), calls 18 whale / 2 part / 8 invented; arm B 45 meshes (36 + 9), 15 / 23 / 7;
outline IoU median 0.84 (6 of 28 under 0.70) and 0.88 (5 of 45 under 0.70); same number 0.119 (24 pairs) against
different numbers 0.168 (50 pairs), AUC 0.63 by my own count, 13 of 50 different-number pairs closer than the
same-number median; 29-42 s and at most 5.28 GB (A), 29-61 s and 6.14 GB (B); arm B lock held 1,715 s. All agree.
The call files are older than the tables computed from them (03:17 before 03:18; arm B 03:54). `mesh3d/` 293 MB,
cache 13 GB, nothing of this job still running.
My look at four sheets (best 12, random 12, one number's 4 meshes, first 12 cut whales; AI by-eye): I agree with
the calls on the tiles I saw (9 of 12 random arm-A tiles read as a whale to me). What the meshes are: grey shapes
without any surface, so no scar, pigment or fluke pattern is in them; seen from the camera they follow the outline,
the other side and the thickness are guessed (a whole silhouette from below becomes a generic whale, a cut whale a
block with a flat face where the frame cut it). Four frames of one whale 1-4 s apart give four different bodies
(`mesh3d_trial_c_number_01.jpg`: one is a disc on a stalk). So: a single frame through a general image-to-3D model
gives a plausible whale shape for 18 of 28 whole views and tells nothing about which whale it is.
Limits: one stand-in model (TripoSG), not the model the user named (SAM 3D Objects: no access); arm A is 26 of 28
from one deployment and 15 of 28 from 4 numbers; the calls are one agent's by-eye calls with 3 + 13 marked
uncertain; arm B was added by me after seeing the pick counts and is reported apart.

## 2026-11-24 01:42  07_tracks_demo/online (WP13, rules frozen as of WP11c): tracker on ALL hq spans of both test deployments: 135 spans / 786 s tracked, 157 -> 149 numbers (8 lone links, all breaks <= 0.53 s), 0 own-back refusals, 9 bubble suspects; 2.25 h lock-held
- Spans = `uwid tracks <deployment>`: every hq span of a kept stretch with advanced_ok in the scan's `sensitive`
  intervals.json, capped at 30 s around its centre (none longer): CC-09-85 17 spans / 147 s (3 recordings), 2025-02-03
  118 / 675 s (9 recordings); tracked from each span's first start-box frame: 142 s + 644 s. Reused: 00024 108-118
  (04:01, current rule); 7 spans of the old rule moved aside (`*_pre_wp13`) and re-run.
- Run under CPU overload (load 47-60 / 37 cores, 21:00-01:00): v1 `uwid tracks` piece (21:50-22:05) finished 3 short
  4K spans, then was cut off at 900 s in 00023 120-138 (its output buffered: fixed with `python -u`; its "0 spans"
  / epoch "held" line was the stopped waiting driver, total unaffected). v2 = `online/wp13_queue.py` pieces (same
  `run_span`, 1080p shortest first then 4K, per-span estimates, no re-run of a cut span by the same path) with the
  frame store: the one `src/uwid/trackstage.py` change during the run (flag `frame_store`, default off: `store_frames`,
  `span_frames`, `span_start`; `run_span` / `patch_pass` / `track_online` read a complete store when given one; meta
  `frames_from`; test `test_frame_store_gives_the_decoded_frames`). Store check: 00017 290-291 (4K) from the store =
  identical (0 / 23 mask frames differ, table, scores, patch labels, endings, numbers equal), 19.5 -> 6.7 s. Paths:
  store 98 spans (2025-02-03 95, CC-09-85 3), decoded 36 (23 + 13, incl. v1's 3 and 00023 120-138 re-run decoded at
  01:29 when the load was 6-24), reused 1. Lock held 8,088 s by the driver + the store-check piece (< 10 min).
- Results (tables.md D9): CC-09-85 19 -> 18 numbers (11 >= 3 s), 1 link, 0 blocked frames, 1 start refused;
  2025-02-03 138 -> 131 (68 >= 3 s), 7 links, 11 "not lone" + 3 "checks contradict" not made, 32 blocked frames, 60
  starts refused (surface 47, fish 11, other 2). GPU tracking 9.45 / 7.75 min per span-minute. No link crosses a
  breath. Calf suspects 0 (no own-back role anywhere; the 00005 604-629 calf stretch is not an hq span).
- Review pack (`review_sheets/underwater/`, tool's decision only): `wp13_a_sample_01..04` (24 numbers >= 3 s: all 11
  of CC-09-85 + 13), `wp13_b_links_01..02` (8), `wp13_c_bubble_suspects_01` (9; e.g. 00005 380 s number 2 on the
  surface); videos `demo_wp13_01..06_*` (9.2 MB). Lists: `underwater_cache/tracks/wp13_*.csv`; deployment
  `links.csv` now includes the encounter rows. 82 tests pass. Not looked at by eye beyond checking the sheets render.
- Code: `experiments/07_tracks_demo/online/wp13_queue.py`, `wp13_predecode.py`, `wp13_store_check.py`,
  `wp13_report.py`, `jobs/run_wp13.sh`; `src/uwid/trackstage.py` (frame store flag); `tests/test_trackstage.py`.

**Reviewer's note (main thread, 01:47).** Heading time matches the clock (entry 01:42, `date` 01:43). Recomputed
from `underwater_cache/tracks/wp13_{spans,numbers,links}.csv`: 135 spans, all done, every folder holds `masks.npz`,
`meta.json`, `tracks.parquet`; CC-09-85 17 spans / 147.0 s / 142.1 s tracked / 19 -> 18 numbers / 11 lasting >= 3 s /
22.4 GPU min; 2025-02-03 118 spans / 675.3 s / 643.7 s tracked / 138 -> 131 / 68 / 83.2 GPU min; links 8 made
(gaps 0.10-0.53 s), 11 "not lone", 3 "checks contradict"; refusals 48 surface, 11 fish, 2 other; own-back role 0.
All agree with the entry. `run_wp13.log` ends `ALL_DONE 01:32:14`, lock held 8,088 s; nothing of ours is running,
`tmp_frames/` is empty, cache 13 GB.
My look at the sheets (AI by-eye, the user has not seen them):
- Sample (24 numbers, 144 frames): every outline is on a whale or part of one; no number moves to another animal.
  Checked closer with my own frames: 00006 1766-1781 s (the outline's last frame looked like another whale: it is
  the same whale coming closer, 12 frames in a row) and CC-09-85 00023 252-271 s (the outline reaches down to the
  tagged whale's back line on 252-261 s: the other whale is alongside and fills that corner; ragged border at the
  flipper base, no leak onto the back).
- Links (8): each joins the same shape in the same place across a break of 3-16 frames; none looks wrong.
- After a lens-out (9 numbers): 7 are whales; 00005 379.7-380.7 s number 2 is on the sea surface (0.93 s, a
  miss of the surface refusal); 00005 365-367 s number 1 is a faint shape low in the frame under a pale puff that I
  cannot name (a far whale's tail or a calf from behind, or nothing). Neither lasts 3 s, so neither reaches the 3D
  trial.
- The six longest numbers (videos 01-06): close whales, one number each for 17.7-26.2 s.
Limits: hq spans are short (median 4 s on 2025-02-03) and 106 of 118 hold one number, so this run says little
about many-whale scenes or about breaks across a breath (no link here crosses one); the own-back refusal was never
used, so the calf case (WP14) is not tested by it. The frame-store path was compared on one 4K span only (identical);
95 of the 118 spans of 2025-02-03 ran from the store without a 1080p comparison: one 1080p span compared both ways
is owed before the store becomes the normal path.

## 2026-11-23 23:21  09_seafloor_sand (WP12 step B): the refitted picture `sand` label finds 58-59 of 77 held-out sediment windows of 00017 but calls 55-75 % of the confirmed seafloor windows and 333-1,373 of 1,736 windows of 2025-02-03 00006 sand: not usable as a scan label; the sensors (deep + rolled over + active) mark the scraping episodes and fire on 0 of 19,248 windows elsewhere
Truth: the user's confirmation of the 22 sediment frames (00017); 77 sediment windows in 4 episodes, 920
not-sediment (`none` + user-confirmed `visible`). One recording, one day, one bottom; near-copy windows. CPU only
(cached frame embeddings), logistic regression, leave one episode out (5 s buffer), thresholds from an inner
leave-one-episode-out. Variant A: 00017 only; B: + the other 2025-01-02 recordings as negatives (each scored without
itself). 2025-02-03 and dev never fitted on.
- Held out: S1 4-5/9, S2 19/31, S3 25/25, S4 9-11/12 (A 59/77, B 58/77). Confirmed seafloor windows called sand
  49-66 of 81-89; murk 101-114 of 104-114.
- False calls (A / B): 00010 17 / 0 of 616, 00011 67 / 1 of 1,499, 00023 91 / 3 of 1,297 (back with haze 5 / 0 of
  291), 00024 108 / 102 of 386 (B: 1 of 272 on-whale, the rest after the tag came off), 2025-02-03 00006 1,373 / 333
  of 1,736, 2025-02-03 labelled 739 / 508 of 821, dev 20 / 0 of 91. Old sand model 0-1 everywhere (in-sample on
  2025-02-03 and dev). Old sand label vs new truth: 42/77 sediment and 45/920 not-sediment windows labelled sand.
- Sensors (00017, medians): scraping episodes (seafloor, sediment, black, murk) 1.5-2.4 m below the max depth, roll
  about 165 deg from normal, ODBA 0.21-0.26; resting 3.8 m below max, not rolled, ODBA 0.02; swimming 41 m above max.
  Trial flag (roll > 90 deg off + within 5 m of max depth + deeper than 10 m, my thresholds after looking): 53/77
  sediment, 55/92 seafloor, 190/208 black, 95/115 murk vs 7/524 resting, 29/307 swimming in 00017; 0 of 3,886 windows
  of the other 2025-01-02 recordings and 0 of 15,362 of 2025-02-03.
- Verdict: picture sand label not usable; a sensor `bottom contact` flag is the better route (one event, one
  recording); a clean test needs another bottom contact on another day or tag. Nothing switched: `sand_v2_A/B.npz`
  saved beside `sand_all.npz`.
Outputs: `experiments/09_seafloor_sand/sand_redo.py`, `sand_sheet.py`, `results/sand_redo.json`, tables.md section
"WP12 step B"; cache `underwater_cache/seafloor/`; sheet `review_sheets/underwater/sand_calls_01.jpg`.

**Reviewer's note (main thread, 23:22).** Heading time matches the clock. Checked against `results/sand_redo.json`:
held-out sediment windows A 4+19+25+11 = 59 of 77, B 5+19+25+9 = 58 of 77; confirmed seafloor windows called sand
49-64 of 81-89 (A), 50-66 (B); false calls B: 00010 0/616, 00011 1/1,499, 00023 3/1,297, 00024 102/386, 2025-02-03
dense 333/1,736 and labelled 508/821, dev 0/91 (all match the report). Sheet viewed: the calls outside 00017 are
plain blue or green water (00024 after the tag came off, deep water on 2025-02-03), and the one call in 00011 is a
frame above the surface, so the "underwater" window set lets a surface frame through; no sediment anywhere on it.
Agreed: the picture head has learnt the look of one place (and of empty water), not sediment; it is not switched
in and the scan's `sand` label stays as it was. The sensor finding is the useful part: the scraping episodes are
the whale rolled about 165 degrees, active, within 1.5-2.4 m of the recording's deepest point, and the trial flag
fires on 0 of 3,886 + 0 of 15,362 windows elsewhere. It is exploratory (thresholds set after looking, one event,
one recording), and "near the recording's deepest point" is not "near the bottom" in general. Not built. A
`bottom contact` label from the sensors is the proposal to put to the user, with more bottom footage as the test.

## 2026-11-23 22:35  10_mesh3d (WP15 set-up + smoke test): SAM 3D Objects not usable tonight (gated, this account has no access; stated need 32 GB); stand-in TripoSG (MIT) runs in 5.2 GB / 32 s per frame; 4 of 10 smoke meshes read as a whale, 3 as a humpback (one animal seen from below), the rest are invented shapes
- SAM 3D Objects (official HF card, repo, issues; nothing installed): `facebook/sam-3d-objects` is gated with manual
  approval; checkpoint files return 403 for the logged-in account. The user requests access at
  https://huggingface.co/facebook/sam-3d-objects (full legal name, date of birth, full organisation name). Licence:
  SAM License (use / modify / redistribute, acknowledge in publications, no military / ITAR use). Stated need "at least
  32 Gb of VRAM" (setup.md; maintainer: smallest reliable GPU a V100); users report 16 GB working once, OOM on 16 and
  24 GB, 12 GB taking about 1 h; weights 13.2 GB (two generators 6.7 + 4.9 GB). Under our 9 GB cap it would need
  half precision + per-stage offload + skipped mesh decoder, reported by nobody; not measured.
- Stand-in: TripoSG (VAST-AI/TripoSG, MIT, 1.5B, image -> mesh), fp16, own env `underwater_cache/envs/triposg`
  (5.7 GB), code `underwater_cache/third_party/TripoSG` (fc5c409, unmodified), weights `underwater_cache/weights/
  TripoSG` (7.5 GB). Input = frame + tracker outline as alpha (TripoSG's RMBG not used); hierarchical decoder (no
  nvcc for `diso`).
- Frame rule written before any mesh (tables.md S0): whale number, >= 3 % of the frame, outline >= 6 px from the
  edges, clear of the back outline and above the fixed back box, one piece, not blocked, sharpness >= the number's
  median. 16 existing online runs (00004-00007, CC-09-85 00024, dev clip): 43 frames pass, from 5 numbers; no calf
  frame passes, so the calf frame is a stated exception (fails only the fixed back-box clause).
- Smoke test, 10 frames (AI by-eye viewpoints): side (CC-09-85), below (00005), behind x 3 frames of one number
  (00006), comp group (dev n2), below x 3 frames of one number (dev n3), calf (00006, exception). GPU in 3 locked
  pieces (22:05-22:27, <= 4.2 min each): 30-39 s per mesh (median 32), torch peak 5.18-5.22 GB, nvidia-smi 5.6 GB,
  load 1.3-3.5 s warm / 134 s cold.
- By eye (AI by-eye, one viewing): reads as a whale 4/10 (02 below; 07, 08, 09 = dev n3 from below), as a humpback
  3/10 (07-09, long flippers, taper, fluke); 06 whale-shaped but dolphin-like (short flippers). Not whales: 01 flank
  (blob), 03-05 behind views (fluke on a stalk, a fish tail with scales, a T), 10 calf silhouette (giant fluke on a
  blob). The unseen side is invented in all; foreshortened views are lifted as flat outlines.
- Consistency (scale-normalised, rigid ICP, chamfer in RMS-radius units, resampling floor 0.02): same number 1.2-2.8 s
  apart 0.087-0.117 (6 pairs); different animals in other deployments 0.084-0.285 (29 pairs, 7 as close as the worst
  same-number pair); whale-like meshes of different animals 0.084-0.133 (4 pairs). The distance says "whale-shaped or
  not", not which animal.
- Proportions on the 4 whale-like meshes: flipper estimate 0.17-0.31 of length (published: up to one-third, FAO
  Marine Mammals of the World 1993 p. 60); length / depth 2.95-5.34 (three frames of one animal span 2.95-5.34); no
  verified published fineness ratio found.
- Verdict: the trial pipeline works inside the memory rule; single-frame meshes are mostly invented. Usable as a
  rough whale-shaped proxy only for clean views from below with flippers spread; not for side / behind / near
  silhouettes, and not (yet) as a per-animal shape signature.
- Phase 2 prepared: `pick_frames.py` (rule + hq spans + spread over outline shape, up to 4 per number); about 53 GPU
  min per 100 frames in about 14 locked pieces. Tried on CC-09-85 mid-WP13: 2 picks.
Caveats: no ground truth; one seed per frame; 10 frames from 5 numbers; IoU only checks the mesh against its own
input; flipper estimate also counts the fluke when it is the widest part. `underwater_cache` is now 23 GB (over the
20 GB budget in CLAUDE.md; this WP added 13.3 GB: env 5.7, weights 7.5, outputs 0.12). Experiment folder number 10 is
also used by WP14's `10_own_back`.
Outputs: `experiments/10_mesh3d/` (framerule.py, select_smoke.py, prep_inputs.py, run_triposg.py, mesh_tools.py,
smoke_report.py, pick_frames.py, jobs/, tables.md S0-S4, results/smoke_metrics.json); meshes
`underwater_cache/mesh3d/smoke/meshes/`; sheets `review_sheets/underwater/mesh3d_smoke_01_side.jpg` ...
`mesh3d_smoke_10_calf.jpg`.

**Reviewer's note (main thread, 22:38).** Sheets 07 (from below, flippers spread) and 04 (fluke from behind) viewed:
07 gives a mesh that reads as a humpback, 04 a fluke on a stalk with a scaled surface; the agent's by-eye table is
fair on these two. What the test shows, in my words: the model returns the outline it was given with a guessed
depth. Where the outline alone says "humpback" (from below, flippers out) the mesh does too; where it does not, the
mesh is an invention, and the unseen side is invented in every case. The chamfer numbers agree: same animal 1-3 s
apart 0.087-0.117, different animals 0.084-0.285 with 7 of 29 pairs as close as the worst same-animal pair, so the
mesh carries no identity. SAM 3D Objects stays untested: access is the user's step (the form asks for personal
details and cannot be edited), and the stated 32 GB is above this 20 GB card whatever our cap is.
Cache budget: the brief I wrote put the environment and weights under `underwater_cache` and so broke the 20 GB cap
in CLAUDE.md (23 GB). My error. Fixed at 22:38: the TripoSG weights (7.5 GB, a download, not derived data) moved to
`/home/sat3737/NOAA_whales/underwater_models/TripoSG` with a link left in `underwater_cache/weights/`; the cache is
back under the cap. Folder number 10 is used twice (`10_mesh3d`, `10_own_back`); left as is for now.
Phase 2 goes ahead on WP13's tracks when that run is done, with TripoSG, and is reported as what it is.

## 2026-11-23 22:22  10_own_back (WP14): the calf in 00005 604-629 s was refused because the tracker's SAM back object swallowed it (99.8-99.9 % inside it, 1.3-1.6 % inside the patch map's back); a per-recording back template (tau 0.5) frees both calf tracks and keeps 99.0 % / 97.5 % of labelled back pixels; 17 similar stretches found for the user; no measure separates the after-breath bubbles from whales yet
Study only (`src/uwid` frozen during WP13). Truth: the user's two statements of 2026-11-23; everything else AI by-eye
(mine). 00005 604-629 s is a development case from now on.
- Cause: own back = SAM 3 object 0 prompted from the bottom 25 %; the calf on the back's horizon is absorbed (object 0
  grows 7 -> 10 % of the frame). The WP5b patch map draws the back as a thin band and is not the cause.
- Template (`experiments/10_own_back/ownback.py`, 4 tests): patches called back in >= 50 % of a recording's under-water
  frames (every 10th scanned window), grown 1 patch; own back = SAM back object inside it. Labelled back kept: 2025-02-03
  99.0 % (807,077 / 815,627 px), 2025-01-02 97.5 % (16,501,159 / 16,921,965), dev 74.7 % (in-sample, the back moves).
  00005: 25 / 25 calf boxes pass; the two calf tracks go 0.998 / 0.999 -> 0.377 / 0.185 inside own back (limit 0.6).
- Risk: in 21 stored runs (48 objects) only the two calf tracks flip; SAM back outside the template median 0.2 %, 90th
  pct 9.2 % of it. The back drifts within some recordings (300 s block templates vs whole: IoU down to 0.49 in
  2025-02-03 00011, 0.21 in 2025-01-02 00017): use 300 s block templates; own-body numbers not seen, not measured
  beyond those runs.
- Other stretches: 17 scan whale objects (>= 3 windows) sit in the back prompt band (2025-02-03 16, 2025-01-02 1); all
  pass the template; by my look all are other animals. Sheet `ownback_candidates_01..03.jpg` for the user.
- Bubbles after a breath: the WP5b map calls both confirmed bubble numbers whale (94 %, 56 %); outline change overlaps
  with real whales; only "paler than surroundings" splits the 6 tracked cases (2 bubbles +, 4 whales -), and it also
  flags 2 real white pectoral fins among 24 scan whale objects after a lens-out (6 flagged). No rule; sheet
  `bubbles_candidates_01..02.jpg` (all 24) for the user.
Outputs: `experiments/10_own_back/` (scripts, tables.md, results/measure.json), cache `underwater_cache/ownback/`,
sheets `review_sheets/underwater/ownback_candidates_01..03.jpg`, `bubbles_candidates_01..02.jpg`. One GPU piece of
~3 min (WP5b maps on 2,141 frames).

**Reviewer's note (main thread, 22:24).** Heading time corrected from 22:25 to 22:22 (the clock read 22:23 after the
entry existed). Checked against `results/measure.json`: back kept at tau 0.5 807,077 / 815,627 px (2025-02-03),
16,501,159 / 16,921,965 px (CC-09-85), 2,553,385 / 3,416,147 px (dev); 00005 604-629 s: 25 of 25 boxes pass, the two
tracks 0.998 / 0.999 inside the SAM back object and 0.377 / 0.185 inside own back. The 4 tests pass (my run).
`src/uwid` not touched by this work. All five sheets viewed.
- Cause accepted: the refusal came from the tracker's own SAM back object growing over the calf, not from the patch
  map. That also means the `on_back` refusal has, in the 21 stored runs, refused nothing except this calf: its
  measured benefit there is zero. The risk side of the template is therefore still unmeasured where it matters
  (does anything on the mother's own body get a number once the back object is cut to the template). WP13's list of
  starts refused as own back is the test set for that; the change goes in behind a flag after WP13 and is replayed
  on that list, on the calf stretch and on the 17 candidates before it becomes the default. Template per 300 s
  block, as the drift numbers say.
- Candidate sheets: by my look 15 of the 17 own-back candidates are clearly another animal low in the frame. Two I
  cannot place and pass to the user: 00011 1053-1056 s (a dark wedge rising beside the back) and 00012 1553-1558 s
  (a white pectoral fin at 0 m in a bubble field: another whale's, or the tagged whale's own flipper?). If it is
  her own flipper, "own body" is more than the back and the template would let it through.
- Bubbles: agreed, no rule from one case; "paler than its surroundings" also flags real white fins. One thing for
  the user: the scan's frame at 125 s of 00024 shows a pale shape with two lobes on a stalk that reads as a fluke
  to me. The user's call on that stretch was "just bubbles there is no animals there after the breath", made on
  my 1 s sheet; I keep the user's call and ask for one more look at this single frame on the sheet.

## 2026-11-23 21:25  11_photo_link (WP16): tag and boat photos line up once the camera clock is corrected (camera ~3 min ahead); one photographed tag-on event per day gives the offset to < 1 s; the tagged group owns every photo near a tag surfacing; a real test exists only on 2025-02-03
Read-only feasibility check, nothing fitted. Data: tag caches of CC-11-96 (2025-02-03) and CC-09-85 (2025-01-02),
photo EXIF times / catalogue labels / safe clusters of the same days (Dinov3-WhaleID, 202 and 981 photos, one camera
body for 1,172 of them), field notes (`test4.xlsx`). Details and all counts: `experiments/11_photo_link/tables.md`.
- Clocks: photos are camera time taken as HST (confirmed to the hour by the notes: 192/192 and 701/934 labelled
  photos inside their own group's interval at 0 h, fewer at +-1 h). Tag clock - GNSS time in the tag's own log:
  +12.4 s (02-03), +9.3 s (01-02). Camera - tag from photos of the tag going on (AI by-eye on photos, sensor event on
  the tag): 02-03 drone drop -173.3 +- 0.2 s, 01-02 pole tagging -170.2 +- 0.6 s; camera - GNSS 185.7 vs 179.5 s,
  i.e. one camera drifting 0.2 s/day: consistent.
- Overlap: 02-03 tag on 4.42 h, photos 10:11-14:31, 171 photos inside on-whale time, 104 of them the tagged group
  (G01, MCE) over 105 min with 32 surfacings and the tag video on disk throughout. 01-02 tag on 3.67 h but the boat
  photographed the tagged group (G05, CG) for 1.6 min after tag-on (22 photos, 1-2 surfacings), at a time whose tag
  video (pieces 00013-00016) is not on disk; then another group.
- Surfacings 80 (18.1/h) and 84 (22.9/h). Photos within 3/5/15 s of a surfacing at shift 0: 0/0/0 of 185 (chance
  9/12/23) and 1/6/10 of 86; at the event shift 47/57/80 of 185 (12-15 surfacings) and 17/22/22 of 86 (2 surfacings).
  Photos come 0-10 s after the tag leaves the water (median +7.3 s, n 99).
- Offset scan +-600 s: 02-03 peak -178 s (62 photos, 15 surfacings; median 7, p99 45), 4.7 s from the event offset,
  on a broad shoulder: the data alone would sync to ~5 s. 01-02 peak +542 s (35, median 1) is wrong; the right
  offset is a local peak (-172: 22). One run per day.
- Animals at surfacings: only the tagged group's (02-03 G01A03 34, A01 22, A02 1 within 5 s; G02 / G03 0 of 62;
  01-02 G05A04 22 of 22, the notes' "Tagged" nuclear female; notes also: G05A05 "UW fluke from CATS"). On 02-03 the
  drop photos are labelled A01, but the tag is visible on 3 surfacing photos labelled A03 (AI by-eye, clear on zoom):
  catalogue A01 / A03 swapped somewhere, or my reading wrong. Open, for the user.
- Positions: the tag stores raw GNSS (RXM-RAWX) only, no fix; 0 epochs with >= 4 satellites on the whale (02-03),
  132 in 8 sessions, none within 3 s of a surfacing (01-02). Photos carry the boat's GPS (202/202, 970/981).
- Comp-group clip: no photos of 2025-12-20 on disk; its burned clock (11:29) falls in a Pair group of the notes, not
  in the tagged CG (08:10-09:35). Cached tracks (3 runs, 56 s, cleaned, not purity-guarded): up to 6 whales
  outlined at once (AI by-eye reference: up to 5 besides the tagged one), median 3-4 per frame, 5 of 28 numbers
  end at the top edge; with the strict purity guards up to 6 at once, 59 numbers with median life 1.3-2.5 s, 12
  ending at the top edge (`underwater_cache/photo_link/cg_counts.json`).
Verdict: the two data sets can be put on one clock to under a second wherever a tag-on event is photographed, and
roughly (~5 s) from surfacings alone when the boat stays with the tagged group. A real test of linking tag video to
photos exists on 2025-02-03 only (105 min, 32 surfacings, video on disk, 57 photos within 5 s of a surfacing).
Caveats: two days, one camera, one run; AI by-eye readings of 14 + 12 photos; the tag-on photos show a person
(01-02) and are not on the sheet. Outputs: `underwater_cache/photo_link/` (tables, summary.json, cg_counts.json),
sheet `review_sheets/underwater/photo_link_tag_candidates_01.jpg` (new class: user sees it first), code
`experiments/11_photo_link/{photo_link,make_sheet,cg_counts}.py`.

**Reviewer's note (main thread, 21:28).** Recomputed from the cached tables (`underwater_cache/photo_link/`): 02-03
photos within 3 / 5 / 15 s of a tag surfacing at the event offset 47 / 57 / 80 (matches); animals within 5 s
G01A03 34, G01A01 22, G01A02 1 (matches); my own shift scan peaks at -178 s with 62 photos, median 7 (matches);
01-02: 17 / 22 / 22, all 22 G05A04, scan peak at +542 s with 35 (matches, and wrong, as reported). `tables.md` and
the scripts hold no photo file names and no positions (checked by search). Sheet viewed: no person on it; the tag
reads clearly on #4, #5, #7 (catalogue A03) and #9-#11 (G05A04), and on the drop photos (catalogue A01).
What I take from it: (1) the two clocks can be joined to under a second from the photographed tag-on; (2) photos
within a few seconds of a tag surfacing are photos of the tagged group, with no exception on either day; (3) the
tag on the back is visible in boat photos, so the tagged whale can anchor a photo group for the whole deployment.
Open, for the user: (a) on 02-03 the tag shows on photos the catalogue files under A03 (adult male in the notes)
and on photos filed under A01 (adult female; the user calls the tagged whale the mother). One tag, so one of these
is wrong: catalogue labels, the notes' roles, or the by-eye reading. Not resolved here. (b) The dev clip's burned-in
clock (11:29-11:31) falls in a Pair group of the 2025-12-20 notes; the tagged comp group is 08:10-09:35. A clock two
hours ahead of Hawaii time would fit (09:29); my guess only. (c) 01-02: the 22 photos of the tagged whale after
tag-on fall in recordings 00013-00016, which are not on disk. A real test of a video-to-photo link exists on
2025-02-03 only (105 min with the tagged group of three, 32 surfacings, video on disk). Nothing is built.

## 2026-11-23 20:39  07_tracks_demo, 09_seafloor_sand (user review): go-ahead for the whole hq tracker run; two tracker errors named; sediment confirmed

Written by the main thread from the user's answers to the WP11c report. Nothing was run for this entry.

- Tracker run: "Yes go ahead and run the tracker." Started as WP13 (tracking agent), rules frozen as of WP11c.
- 00024 after the breath (numbers 2 and 3 of the whole-encounter run): "the pale animal is just bubbles there is no
  animals there after the breath, and the back ends low in frame it is especially tall in that clip." Both numbers
  are false. Consequences: (1) bubbles after a breath can get a whale number (scan object, then a tracker number
  that lasted about 8 s); (2) the rule proposed at 12:31 (across a breath, link to the single candidate that passes
  size and appearance) is dropped: candidate 1 to 3 passed both checks (1.39, 0.249) and is whale to bubbles, so the
  "not lone" refusal was the only thing that prevented a wrong link; (3) an appearance distance under 0.30 does not
  show that the thing after a breath is an animal at all. The second half of the user's sentence (the back) I read
  as a remark on the tagged whale's own back in that recording; I have not built on it.
- 2025-02-03, 00005 at 604-629 s: "is a calf swimming clone in front of the mom who is tagged with the camera"
  ("close", my reading). The scan (whale, tier ok / low) was right; the tracker's own-back refusal (outlines
  >= 99.8 % inside the back region) was wrong. The deployment is a mother with her calf, so this can recur. WP14
  (classifier agent) measures how often and tests a per-recording own-back template; no live change during WP13.
  This stretch was held out until now and is a development case from here on.
- Sediment: "Yes those are the sediment frames of dragging along seafloor." (`sand_candidates_best_01.jpg`.) The
  sand redo is unblocked and waits behind WP14.
- Breath demos (WP11c): "looks fine".
- New goals: a 3D mesh trial on the hq tracks once the run is done (WP15, set-up started), and a thought experiment
  on what the tag video could add to the photo sorting of the same day (WP16, a time-overlap check started).
- For the user's request for a link on the calf question: plain 34 s video
  `review_sheets/underwater/scan_00005_600s_calf_in_front_plain.mp4` (600-634 s of 00005, no overlay).

## 2026-11-23 13:35  07_tracks_demo/online (WP11c, rules frozen): held-out lone-whale breaths: 1 of 3 encounters ends with the whale on ONE number (2 links made, 2 refused on appearance only, 0.358 / 0.364 vs 0.30); groups: 0 numbers moved, 0 links
- Selection (`online/select_breath.py`, before any tracker run, scan tables + depth only): 112 lens-out runs of 1-12 s
  by the frozen blocked rule; 24 with an active whale window after, 11 also with hq / ok within 5 s before, 8 of them in
  left-out stretches -> 3 breath encounters (all 2025-02-03: 00004 618 s, 00005 1562 s, 00006 909 s; none on CC-09-85).
  The first rule's "at most one scan whale object per window" passed 0 of 11 (00007 1465 s too): the scan puts two
  whale boxes on one near whale; dropped by the coordinator before the runs. Groups: 2 longest of 19 scan runs with
  >= 2 whale objects for >= 3 s (00004 286-306, 00005 607-627).
- By eye on plain frames first (AI by-eye, `results/breath_eye.csv`): all 3 breaths lone (1 whale before, 1 after).
- Tool (`uwid tracks --only`, 8 pieces, 1,025 GPU s of tracking): 00004 linked across 4.77 s, one number; 00005 linked
  across the first breath (7.41 s; the fragment after it is probably the whale, not certain), refused across a second
  lens-out (appearance 0.364) -> 2 numbers; 00006 refused (appearance 0.358) -> 2 numbers, and number 2 stays 2.9 s
  on a surface sliver after the whale left. Refused kinds: (c) checks 2, (a) / (b) / (d) 0. Moved numbers 0.
- Groups: 00004 286-306 = 2 whales + a slender animal: numbers A 1 -> 4, B 2 -> 5, 0 moved, 0 links (5 candidates, all
  not lone). 00005 607-627: both scan "whales" are the tagged whale's own head and flipper (tool: on_back, 0
  numbers); my plain-frame count there (1 whale far below) was wrong.
- 81 tests pass. No rule, threshold or default changed.
Caveats: 3 encounters from one deployment and tag; the after-side is 1 s (00004) and 0.3 s (00006) of whale; one
viewing of 1 s sheets. Outputs: `online/results/selection_breath*.csv`, `breath_eye.csv`, `breath_summary.csv`;
`underwater_cache/tracks/2025-2-3 Mini-mother 11-96/{index,links}.csv`; videos `review_sheets/underwater/`
`demo_breath_01_linked_00004_608s`, `demo_breath_02_linked_00005_1552s`, `demo_breath_03_notlinked_00006_899s`
(.mp4 + .jpg); tables.md D8; code `online/select_breath.py`, `report_breath.py`, `jobs/run_wp11c.sh`.

Reviewer's note (orchestrator, 2026-11-23 13:40; `index.csv`, the encounter `links.csv` files, the three video
sheets and my own 1 s sheet of the group stretch looked at):
- Gaps, size ratios and appearance distances above agree with the three `*_encounter/links.csv` files; the piece
  counts agree with `index.csv`. The first selection rule that found nothing was mine, not the agent's.
- My reading of the three breaths differs in one place. 00006: after the breath the whale is not in view (0.3 s by
  the agent's own plain-frame count) and "2" is a tier-small object on a sliver of surface, so the refused link is a
  RIGHT refusal, not a missed one. So: links made 2 (00004 right; 00005 probably right), refused 2 (00006 right;
  00005 second lens-out a miss, the whale comes back cut by the frame edge, appearance 0.364 against 0.30), wrong
  links 0, numbers moved 0. The breath evidence is now 00007 (in-sample), 00024 (fluke + body, not linked) and
  these three.
- How much there is to gain: both deployments together hold 11 breaths with a good whale within 5 s before and
  whale activity within 12 s after. Linking across breaths can save at most about that many number changes in this
  footage.
- Group stretch 00004 286-306 (`demo_group_00004_286s_review.jpg`): two whales, A at the right edge (1, then 4), B
  top left (2, then 5); on my sheet no number goes from one whale to the other. The new numbers come from breaks
  of 0.5-0.7 s that are refused as "not lone" although size and appearance pass (1 -> 6: 1.30, 0.081). That is the
  rule the user accepted for group scenes; whether a break under 1 s with the outline in the same place should be
  linked even with two whales in view is a question for later, not changed.
- 00005 604-629 s (`scan_00005_604s_own_body_or_whale.jpg`): the scan calls it whale (tier ok / low, up to 3
  objects), the tracker gives no number because the outlines lie inside the back region. On plain frames it is a
  whale-shaped dark object low in the frame that keeps its place for 25 s at 28 m, with a flipper rising at the
  left. It may be the tagged whale's own head and flipper, or an animal close below it (the deployment is called
  "Mini-mother"); I cannot tell. Lying inside the back mask does not prove it is the tagged whale. Asked of the user.
- Loose end: the deployment-level `links.csv` holds the span rows of the three breath encounters but not their
  encounter rows (those are in the `*_encounter/` folders).

## 2026-11-23 13:03  09_seafloor_sand (WP12 step A): on user-confirmed bottom of 00017, a re-fitted head finds held-out episodes per pixel (96.7 % of seafloor pixels, IoU 0.89; combined back + seafloor IoU 0.96) and 59/59 windows of the 3 large episodes but 0/35 of the faint one; it still calls 21-35 % of 00023's back-with-haze windows seafloor (folds 2-4) and 2.9 % of dev whale pixels: separate class not usable as a scan label, combined class usable for masks; within ONE recording
Truth: the user confirmed the 24-frame sheet (back in foreground, seafloor in background; combined class accepted).
22 AI masks on those episodes (seafloor polygons by eye, back by SAM 2.1, edges / sediment / the dark band between
back and bottom ignored); old 2025-01-02 seafloor paint ignored. Linear DINOv3 patch head (as WP5b) fitted on all
reference masks; leave-one-episode-out inside 00017 (E1 292-327, E2 405-422, E3 465-487, E4 510-533 s; 5 s buffer;
window thresholds chosen on each fold's training windows: 0.15 / 0.03 / 0.05 / 0.05). Same recording, day and bottom;
windows next to a training episode are near copies.
- Pixels, held out: seafloor found E1 86 %, E2 93 %, E3 100 %, E4 99 % (all 5,227,026 / 5,407,044 px), IoU 0.89;
  seafloor called back 547 px; back called seafloor 0.13 %; water called seafloor 2.7 %. Combined class IoU 0.96,
  water called background 4.4 %, background called whale 0.03 %.
- Windows: held-out E1 0/35 (faint bottom, median share 0.064 under fold 1's 0.15), E2 17/17, E3 22/22, E4 20/20.
  Hard negatives (withdrawn `visible`): 00017 842-1390 s 0-3/524; 00023 0-306 s 0/291 (fold 1), 103 / 62 / 67 of
  291 (folds 2-4). 00023 weak positives 0-13/25. Other recordings 0-41 of 220-1,499; 00017 sediment 61-93 of ~123
  (clouds next to the bottom); black windows 0-1 of ~206. 2025-02-03 00006 4/1,736 (old head 22), labelled 2/821;
  dev 8/91 (old 1/91).
- Whale safety (heads not fitted on the source): whale pixels called seafloor 2025-02-03 0 (old 0), dev 2.9 % of
  4,987,442 px (old 0.4 %); called back 0.21 % (old 0.34 %) and 0.43 % (old 0.78 %).
- Verdict: separate seafloor class not usable as a scan label; combined background class usable for masks; a clean
  test needs bottom in a second recording and day; a per-window bottom-contact flag (deep + flat depth, dark,
  sediment) is the cheaper label for the user's purpose (not measured).
- Heads saved as `heads/head_v2_{fold1..4,test_d1,test_dev,all}.pt`; `head_all.pt` and `patchseg.SEAFLOOR_CLF`
  unchanged. The scan's picture-class stage reads `head_all.pt` (a switch re-runs 42,282 cached frames, ~40 GPU min
  plus decoding); the tracker's patch pass reads WP5b's `clf_w1280_h0_all.pt` and is not affected.
- Sand: nothing fitted; `sand_candidates_best_01.jpg` for the user (22 sediment frames + the 2 places where the old
  sand label is wrong; missed clouds marked).
Outputs: `experiments/09_seafloor_sand/wp12*.py`, `wp12_prompts.json`, `results/wp12_*.json`, tables.md section
"WP12 step A"; cache `underwater_cache/seafloor/{wp12,frames_v2,heads}/`; sheets `review_sheets/underwater/
seafloor_masks_01..02`, `seafloor_heldout_found_01..02`, `seafloor_heldout_missed_01`, `seafloor_false_01`,
`sand_candidates_best_01`. Tests 81 passed.

Reviewer's note (orchestrator, 2026-11-23 13:08; `wp12_pixels.json`, `wp12_windows.json` and the five sheets
looked at):
- Every number above agrees with the two result files. One row is in tables.md but not in this entry: plain
  no-seafloor windows of 00017 itself (0-292 and 644-673 s) called seafloor 8 / 36 / 34 / 32 of about 305.
- By eye the masks sit on the rough bottom and stay inside it, and on held-out E2 and E3 the head's region follows
  the bottom (`seafloor_heldout_found_01`). In the "missed" episode E1 the region is found too, at the right edge
  (share 0.04-0.05); only the window threshold drops it. So the pixel class works inside this recording and the
  window rule is the weak part: 0.15 misses the faint bottom, 0.03-0.05 fires on other things.
- What the low thresholds fire on (`seafloor_false_01`): dark water at the side of the frame (00010, 00011), the
  haze band beyond the back in 00023, sediment and murk in 00017, bubbles on the dev clip, and once the dark
  underside of a near whale on 2025-02-03 (share 0.23). The last one is the footage the user values most, so the
  new head does not go into the scan as it is.
- E1 at 292-297 s has another whale in view above the back while the bottom shows at the edge.
- I agree with the verdict: no switch of `head_all.pt`, the scan's `seafloor` label stays empty. The scan loses
  nothing by that: its whale detector does not fire on the bottom (WP9) and nothing is kept or dropped by a seafloor
  label. A clean test needs bottom in a second recording; only 9 of this tag's recordings are on disk (00005, 06, 09,
  10, 11, 12, 17, 23, 24), and the user spoke of "a couple of the middle long videos", so more may exist in the ones
  not copied (00013-00016, 00018-00022). Asked of the user.
- The bottom-contact flag is a proposal, not built: nobody has asked for it and it would be set on one recording.

## 2026-11-23 12:31  07_tracks_demo/online (WP11a follow-ups): depth lens-out blocks only where the picture agrees (00024 blocked 117 -> 30 frames, the whale alongside is one number up to the breath); stitched-run link fault fixed; every link candidate written to links.csv
- A: a depth lens-out frame is blocked only where the scan's 1 s picture label `water_picture` is not `underwater`
  (no level fitted; checked on 2025-02-03 00007 1465-1468 s and 00006 581-584 s, not on 00024). One locked GPU job
  (03:57-04:06, 4 spans): 00024 97-120 blocked 117 -> 30 frames, outlined 526 -> 623 of 653, numbers 2 -> 1;
  108-118 blocked 57 -> 0, outlined 226 -> 293 of 293, 2 -> 1; 119-142 unchanged (starts after the breath);
  00007 1459-1475 (real breath) 126 frames still blocked (87 lens-out, all confirmed by the picture), linked across it
  (gap 5.84 s, was 4.4 s after the fact: this route starts numbers only from scan decision objects, first at 1471.24 s).
- B: in the stitched encounter detector boxes on blocked frames counted as an unexplained second animal ("not
  lone"); blocked frames are now left out of the before / after counts. Test
  `test_lone_link_ignores_detector_on_blocked_frames`.
- C (report only): across the breath in 00024, 1 -> 2 (size 2.40, appearance 0.353, checks fail) and 1 -> 3 (1.39,
  0.249, checks pass) are both refused as "not lone" because 2 and 3 are in view together. `links.csv` now holds every
  judged candidate per span, per encounter (`*_encounter/links.csv`) and per deployment beside `index.csv`.
- Whole 00024 encounter: 4 -> 3 numbers (was 5 -> 4); `demo_online_01_cats85_00024_whole_encounter_near_whale_breath`
  re-rendered (26.2 MB). Other spans: made links unchanged (00008 encounter 2 -> 1, same link). 81 tests pass.
- 00024 108-118 and the whole 00024 encounter are no longer held-out for the blocked rule (tables.md D3 note, D7).
Caveats: windows the picture model has no row for keep the sensor decision; the picture label is 1 s coarse, so a
lens-out crossing inside an `underwater` second is not blocked; one GPU run per span. Outputs:
`underwater_cache/tracks/<deployment>/{index,links}.csv`, `*_online_pre_fixA` (old runs, comparison only),
`experiments/07_tracks_demo/online/run_followup.py`, `jobs/run_followup.sh`, tables.md D3 / D6 / D7.

Reviewer's note (orchestrator, 2026-11-23 12:40; both indexes, `links.csv`, the window tables and the video's
frames looked at):
- The blocked, outlined and number counts above agree with `index.csv` for the four re-run spans; no other span's
  row changed. 81 tests pass in my own run. The GPU run was 03:57-04:06; the report came 8 h later because the
  agent's wait on the render did not wake it.
- The picture label is right where it overrules the depth in 00024: depth says crossing at 109, 115 and 117 s and
  surfaced at 118 s, the picture says under water until 119 s, and at about 118.7 s the whale is in view with its
  outline. In 00007 the picture reads crossing / air / air / crossing at 1465-1468 s, as reported.
- After the breath (my frames `demo_online_01_review_breath_after_fix.jpg`): number 2 is a fluke, number 3 a pale
  body at the upper left edge, both in view together for about 1 s; the drift ending at 126.3 s is number 2 (the
  fluke leaves). To me they look like one animal seen belly-up, and possibly the same whale as number 1; not
  certain. This is one case, so no rule is changed on it: the user is asked whether it is the same whale.
- The `*_pre_fixA` folders (4.2 MB) are left in place.
- Still not shown on held-out footage: a lone whale keeping its number across a breath (00007 is the stretch the
  rule was set on), and anything with two or more other whales.

## 2026-11-23 12:29  09_seafloor_sand (user review): the seafloor candidate sheet is confirmed; a combined back / seafloor class is an accepted fallback
The user on `seafloor_candidates_best_01.jpg` (24 frames of 00017; `seafloor_candidates_00023_01.jpg` was linked
beside it): "yes all of those seafloor candidates now show both the back of the whale in forground, and the seafloor
in background. If it helps its not an issue if they got a combined class where it was just back/seafloor. Best if
separate but not as issue if they are combined to just background."
- Orchestrator's reading: the 00017 frames are confirmed seafloor, so masks and a re-fit may start (WP12 step A,
  sent to the WP10 agent 12:30). Separate `seafloor` and `back` preferred, one combined class measured beside it.
  The two faint bands in 00023 are not fitted on; they are scored apart as weak positives.
- Limits known before the fit: 94 s of bottom in one recording, so the test is leave-one-episode-out inside 00017
  plus false calls on every other recording. Whether it carries to another day or bottom cannot be tested.
- Not covered by this confirmation: the sediment (sand) calls. A sheet goes to the user first; nothing fitted.
- No scan change in step A: the new head gets a new name, `head_all.pt` stays until the result is reviewed.

## 2026-11-23 03:34  07_tracks_demo/online (WP11a): `uwid tracks` runs the purity policy ONLINE on the scan's hq spans; held-out (8 stretches + 2 breaths, rule fixed first): 0 numbers moved, 3 lone links all right, no number on the blow / surface (13 of 13 starts refused); ~9 GPU s per video s
- `src/uwid/trackstage.py` + `uwid tracks "<deployment>"`: reads `intervals.json` at run time; per hq span a patch
  pass (blocked frames: depth lens-out, bubbles), then the SAM 3 tracker from the scan's `whale` decision objects
  with three start screens (< 30 % clutter patches, the WP10 box classifier says whale) and the strict guards acting
  online (tracker object removed from the session when a guard fires). Resumable per span; whole encounters run as
  overlapping pieces joined by `stitch`. Output per span + `index.csv` under `underwater_cache/tracks/<deployment>/`.
- Lone-link layer (`tracks.link_lone`, the user's easy cases): fragments keep ids, chains share a number; lone =
  no other whale-sized object apart from the animal at the same time, nor an unexplained detector object on >= 2
  detector frames, 2 s around the break; break <= 8 s (12 s with a sensor breath); size and appearance as vetoes;
  slow non-fish objects under the size limit get a whale number. Set on old stretches only, frozen 01:30: lone gap
  runs 12 links / 12 right (1.70 -> 1.30 tracks per whale), calf+fish gap runs 5 / 5, group and dev runs 0 links.
- Held-out by eye (AI by-eye; every held-out stretch had ONE other whale): 0 moved; whale numbers 12 -> 9 for 8
  whales (1.50 -> 1.13 per whale); 3 links, 0 wrong, 0 lone breaks left; outlined 98-100 % of frames except the
  breath span (226/293, the depth lens-out hides a visible whale); endings 3 drift, 1 area jump, 1 shape jump.
  Box classifier refused 2 starts (a jack, right), withheld no whale; no whale left `small`. Breaths (CC-09-85
  00011 499-506, 00017 176-186): no number started, the clutter screen refused 13/13 blow / surface / mirrored-back
  starts and the box classifier calls the same 13 `other`.
- Online vs after the fact (same WP5a starts): dev clip 01 agrees (5 vs 6 numbers, 5/5 online numbers map to one
  after-the-fact number); clip 05 differs (6 vs 9 numbers, 3/5 map 1:1): online keeps the calf (with links: number
  3 from 459.5 s to the end) but ends more fish tracks.
- Cost: 9.7 GPU s per video s on 2025-02-03 (1080p), 8.8 on CC-09-85 4K (breath spans 5-7), <= 2.3 GB. All hq spans:
  2025-02-03 758 s -> ~2.1 GPU h (1,402 s of hq stretches -> 3.9 h); CC-09-85 189 s -> ~0.5 h (352 s -> 0.9 h). Not
  run: needs the user's go-ahead.
Not tested / caveats: no held-out multi-whale scene (no-link-in-groups rests on the old dev runs); the calf-with-fish
after the fact stays 9 -> 8 numbers (its flipper and fish tracks make breaks "not lone"); depth lens-out over-blocks at
the surface; 4K clock not read in the videos; eye checks one viewing on 1 s sheets. Videos: `review_sheets/underwater/`
`demo_online_01..05_*`, `demo_link_01..03_*` (.mp4 + .jpg); tables `07_tracks_demo/tables.md` section D; code
`src/uwid/trackstage.py`, `tracks.py` (online removal helpers, `LinkConfig`, `link_lone`), `cli.py` (`tracks` block),
`tests/test_trackstage.py`, `tests/test_tracks.py`, `experiments/07_tracks_demo/online/`.

Reviewer's note (orchestrator, 2026-11-23 03:50; index.csv, meta.json and the videos' sheets looked at):
- Per-span counts in `index.csv` agree with the table in tables.md D3 (frames, outlined frames, blocked frames,
  endings). 79 tests pass with the scan agent's changes in the same tree.
- The three held-out links are all short breaks (0.23 s, 0.10 s, 0.97 s): a guard ended the track and the next
  detector frame restarted it. No held-out link crosses a breath or a break of more than 1 s. The user's two named
  cases (after the breath in 00007, the calf with fish) are links on the stretches the rule was set on.
- `demo_online_01` (00024, 75-164 s), the only held-out whole encounter with a breath: the one whale in view carries
  numbers 1, 2, 3 and 4. (a) The depth lens-out blocks 115.3-115.9 s although the lens is under water and the whale
  fills a third of the frame; the track ends by shape jump when the view returns, and the whale is visible without
  an outline for about 2 s before the breath. (b) The 1 -> 2 link that the separate span run (108-118 s) makes is not
  made in the stitched encounter. (c) After the breath (lens out about 117.3-125.2 s) number 3 is the fluke and 4 a
  pale body part, apart from each other; whether they are the same whale as number 1 I cannot tell either (dark
  flank before, pale parts after). My frames: `demo_online_01_review_breath.jpg`. Sent back: (a) and (b) as faults,
  (c) to be reported with the stored evidence, no rule change.
- `demo_link_01` (calf with fish, online): the calf is number 1 for 3 s, then number 3 to the end; one change of
  number on the same animal, no number moves to another animal on the sheet.
- hq spans changed with WP11b (675 s and 147 s instead of 758 s and 189 s), so a run on all hq spans is now about
  1.8 + 0.4 GPU h at the measured rates. Not run.

## 2026-11-23 03:13  06_clips_e2e (WP11b): box-classifier veto + sand / dark labels + 30 s activity guard in `uwid scan`: tier-hq frames graded low / whale-free 8/44 -> 0/37 and 1/11 -> 0/10, no labelled whale lost; clips 161 -> 100 (86.5 -> 94.3 min) and 61 -> 27 (16.6 -> 17.6 min)
Both deployments re-scored from the cache plus one new GPU stage (picture classes: 31,546 + 10,740 frames, 0.13-0.14
GPU-h per video hour, 3 locked passes each with `timeout 900`). Nothing tuned on the test deployments; WP10's
classifier was fitted on 1,054 labelled objects incl. the reference objects of both deployments and the 24 jacks:
the label scores are NOT a clean test. Truth: AI by-eye labels.
- Veto (rule-A whale box >= 300 px stays a whale only if the classifier says whale; fish -> fish object; other ->
  dropped, also at breaths): whale windows 3,002 -> 2,897 / 389 -> 248; hq windows 723 -> 647 / 174 -> 132, of them
  under 1 m 53 -> 27 / 71 -> 41; WP10's breath boxes (not fitted on) in hq / ok windows 26 -> 1 and 43 -> 0; the 24
  near jacks (fitted on) in ok / hq 12 -> 0. Windows the veto made inactive (active without the classifier, not
  with it; corrected after review): 59 on 2025-02-03 (42 still inside a kept clip, 17 no longer kept) and 71 on
  CC-09-85 (10 inside, 61 no longer kept); the flag `vetoed_other_only` (78 / 71) also counted 19 windows the shallow
  rule had already emptied. 60 flagged windows looked at by eye: 3 held a real whale (2 = the whale beside the
  floating tag in 00012, which loses its whale label on 3 windows at 14:39:37-40 but stays inside its clip; 1
  possible fin at an edge).
- Guard (user: a clip ends only after 30 s without activity, never across air / flight / a lens-out > 20 s; 2 s
  buffer): alone 161 -> 94 clips, 86.5 -> 99.7 min; 61 -> 35, 16.6 -> 23.1 min (the orchestrator's cross-check: ~93 /
  99.9 and ~35 / 23.1). Both: 100 clips 94.3 min (median 23 s, longest 607 s, 19 under 10 s, 1,646 s quiet inside);
  27 clips 17.6 min (median 20 s, longest 149 s, 8 under 10 s, 609 s quiet).
- Labels (before -> default): whale frames kept 219 -> 220/221, 27 -> 29/31; activity (air frames now excluded)
  258 -> 264/279, 36 -> 38/46; empty kept 53 -> 72/542, 31 -> 20/292; hq / ok frames in tier ok+ 92/93 and 13/20
  (unchanged); tier ok+ frames graded low / free 44/136 -> 36/128, 13/26 -> 1/14; tier hq graded low / free 8/44 ->
  0/37, 1/11 -> 0/10.
- Dark: 208 windows on CC-09-85 (8 of 11 labelled dark frames, 0 false), 0 on 2025-02-03; sand 242 (in-sample, one
  recording) and 3. Both out of hq spans. Seafloor not filled (user withdrew WP10's class); WP9's seafloor sheets and
  seafloor-span numbers are void; `evaluate.py --dep d2` marks them withdrawn.
- Evaluation fix: faint whales on frames labelled air (the drone's approach) no longer count as activity.
Still wrong (known error, not tuned on): the whale beside the floating tag (00012, 14:39:37-40) loses its whale label
on 3 windows (classifier: fish / other) but stays inside its clip; 41 / 27 hq windows under 1 m not checked one by one
by me (the orchestrator checked them: real near whales but 1 doubtful); large-fish evidence is one school the classifier was fitted on; sand unchecked by the user; no
seafloor detector. Outputs: `src/uwid/pipeline.py` (`picture_classes`, `attach_box_classes`, `fill_window_classes`,
`clip_intervals`, `hard_breaks`, `mark_continuing`), `clips.py` (columns), `cli.py` (scan flags only). After review:
sheets draw only what the decision uses (fish boxes >= the fish threshold or classifier-made, "(clf)" from
`boxclf_class`), descriptions fixed (fish box count wording, clip length in seconds incl. a part window at a file's
end); 79 tests. Tests
`tests/test_scan_wp9.py` (guard 29 / 31 s, file ends, hard breaks, veto, merge, labels), `experiments/06_clips_e2e/`
(`tables.md` section 13, `wp11b_measures.py`, `evaluate.py`, `sheets.py veto_dropped`, `jobs/run_wp11b.sh`,
`jobs/rescore_wp11b.sh`, `results/wp11b_measures.json`, `eval*.json`, `veto_dropped_sample.json`), cache
`underwater_cache/scan/<deployment>/cache/pclass/`, sheets `scan_veto_dropped_01..05`, `scan_hq_*`, `scan_d2_hq_*`,
`scan_hq_band_*`, `scan_d2_hq_band_*`, `scan_midsize_01..04`. 77 tests pass.

Reviewer's note (orchestrator, 2026-11-23 03:20; checked against `underwater_cache/scan/<deployment>/`):
- Window counts, minutes and clip counts of all five runs per deployment agree with the cache.
- "78 windows no longer active" on 2025-02-03 is too high: 19 of them were already inactive before the veto (the
  shallow rule had removed the box). The veto itself made 59 windows inactive there and 71 on CC-09-85.
- Of those, 42 of 59 and 10 of 71 are still inside a kept clip (as quiet seconds under the 30 s guard); 17 and 61 are
  no longer kept. The whale beside the floating tag (00012, 14:39:37 / :39 / :40) loses its whale label on three
  windows but the footage is kept: the clip runs through.
- The hq windows still under 1 m, not checked by the agent, looked at by me (AI by-eye): 2025-02-03 26 of 27 show a
  real near whale, 1 doubtful (00006 11:22:32, the surface from below); CC-09-85 all 41 sit in the two near-whale
  passes at 16:01 and 16:07-16:08, of which I saw the top 12 (all real). Sheet `scan_hq_under1m_review_01.jpg`.
- `scan_veto_dropped_*`: cyan "fish" boxes are drawn on windows whose fish decision is False (prompt boxes under the
  fish threshold), and "(clf)" on a fish box is guessed from scores, not read from the class. The sheet must draw
  the decision only. Sent back to the agent.
- A box the classifier calls fish does keep its window active (43 and 65 windows, all active): checked in the code
  and the cache.
- 03:25: the agent's fixes checked. `wp11b_measures.json` now gives 59 (42 inside a clip, 17 not kept) and 71 (10 /
  61); the re-made `scan_veto_dropped_03.jpg` shows the whale at 14:39:39 as "other 0.67 778px (clf)" with no fish
  boxes under the threshold; clip texts no longer say "up to 0 boxes" and the clip at a file's end reads 4.4 s;
  minutes and clip counts unchanged (94.3 / 100, 17.6 / 27); 79 tests pass.

## 2026-11-23 01:24  09_seafloor_sand (WP12, search only): the real seafloor is in ONE recording, 00017: 94 s of visible bottom in 5 episodes at the side of the frame while the whale scrapes along it (292-327, 405-422, 465-487, 510-524, 527-533 s), sediment 128 s; none found in 00010, 00011, 00012, 00023 (2 unsure bands); sheets for the user, nothing fitted
AI by-eye look by the WP10 agent at 1 s frames (1280 px) of sensor- and picture-chosen candidates (60 s around dark
windows, deep flat stretches, every deep stretch of 00011 / 00023); 00024 has no candidate. No accelerometer in the 1 Hz
table, so motion contact was not used. The old `visible` spans (00017 842-1390, 00023 0-306) show the back with haze:
moved to `seafloor_spans_withdrawn.csv`; new 1 s spans prepared in `seafloor_spans.csv` (00017, 00023), not yet
confirmed by the user. Sand labels: the big clouds hold; 405-422 s (bottom) and 645-675 s (haze) are wrong.
Mishap: rendering my sheets with the requested name pattern deleted the orchestrator's own
`seafloor_candidates_00017_01.jpg` / `_02.jpg` (same names); mine replace them.
Outputs: `experiments/09_seafloor_sand/candidates.py`, `candidate_calls.csv`, tables.md section WP12; frames in
`underwater_cache/seafloor/cand/`; sheets `review_sheets/underwater/seafloor_candidates_best_01.jpg`,
`seafloor_candidates_000{10,11,12,17,23}_NN.jpg`.

## 2026-11-23 01:10  02_reference_labels + 09_seafloor_sand (user review): the seafloor labels are wrong and are withdrawn; WP10's seafloor numbers are void
The user looked at `seafloor_01..05.jpg`: "Seafloor_01 is a whale back, so is 02, 03 is just hazy water, 04 is a
mixture of hazy water, a crowded whale scene, and reflection off surface from below, 05 is also just a whale back.
Seafloor can be found in a couple of the middle long videos from the Test subset day. Largely after/before moments
where whale screen blacks out from scratching along seafloor and kicking up ab unch of sediment."
- Withdrawn: the labeller's `visible` spans (00017 842-1390 s, 00023 0-306 s, "sand plane fading into haze"), the
  `seafloor` column and masks built on them, and with them WP10 part 2's seafloor results (541/548, 305/306 and the
  false-call counts): the model learned the haze band above the back. WP9's `scan_d2_seafloor_*` sheets rest on the
  same spans. `dark` (brightness rule) stands. `sand` was fitted against labels the user has not checked.
- My own look (main thread, 33 frames of 00017 at 960 px, not labels): the bottom is the rough, mottled surface at
  the side of the frame while the whale scrapes along it, tilted like a wall because the whale is rolled: about
  304-326 s (upper right), 405-415 s (right), 478-484 s (left), 532 s (upper right), with sediment clouds at
  333-335, 420-428, 532, 612, 750 s and black frames between. That is inside the labeller's `unsure` span
  (296-842 s), which it sampled at 10 s steps. Sheets for the user: `seafloor_candidates_00017_01..02.jpg`.
- How it went wrong: a new class was labelled by eye from faint, contrast-stretched frames and a model was trained
  on it before the user saw one example. Rule from now on: the first sheet of any new class goes to the user before
  anything is fitted on it.
- Next (WP12): dense search around every dark / sand span and every bottom-depth stretch of the second deployment,
  candidate sheets for the user, new spans and masks, re-fit and re-score with the old result kept only as a warning.

## 2026-11-23 00:15  07_tracks_demo/pure (WP8c): purity guards end a track number instead of letting it move: 0 moved numbers on the 50 gap runs (old 18, 12 of them whale numbers) at 1.95 tracks per whale (old 1.11); 7 of 9 by-eye moves fixed; the two joins in clip 01 still move
The user: a number moving to another whale is worse than new numbers. `src/uwid/tracks.py` `PurityConfig` /
`purify`: a number ends at a blocked view longer than 1 s (4 s if it was the only whale-sized track), after 0.5 s
unseen, on an outline jump (IoU with the previous frame < 0.45 or area x2), when it newly covers >= 35 % of another
whale-sized track, on appearance drift (masked DINOv3 ViT-B/16 feature, cosine distance > 0.10 from the track's
running mean), when a fish-sized start grows x1.5; no number while the region is >= 30 % bubbles / surface / air;
fish-like first second (< 0.3 of the width, > 1.5 body lengths/s) gets no whale number; no outline on blocked
frames. Applied to every cached WP8 / WP8b run (cut and renumber); new GPU: 3 backward runs + patch features.
- Truth: gap cut runs against the uncut run (moved = one displayed track on two animals the uncut run shows apart);
  9 moves / joins I saw by eye in WP8 / WP8b (AI by-eye).
- old -> loose / medium / strict: moved 18 -> 10 / 6 / 0 (whale numbers 12 -> 7 / 4 / 0); tracks per uncut whale
  1.11 -> 1.40 / 1.64 / 1.95; uncut whale frames outlined 0.742 -> 0.735 / 0.722 / 0.685; by-eye moves fixed
  1/9 -> 6/9 / 6/9 / 7/9; whale numbers on 19 demo / real runs 108 -> 134 / 160 / 209. Shipped default: strict.
- Work done by: appearance drift (alone 18 -> 9), outline jumps (-> 12), blocked-view limit (-> 15), absence (-> 15);
  swallowing, grow-from-fish-size and the start screen alone change little; together they reach 0. Clean single
  whales (clips 03, 04, 06) keep one number; the calf across a 4.2 s breath now gets a new number; fish fragment.
- Not caught: clip 01 joins at 25.6-26.0 and 29.8-30.4 s (no abrupt change, the swallowed whale had no track).
  Forward / backward runs flagged 3 tracks in the dev breath and nothing in clips 01 / 07; not shipped.
- Fish vs whale in the first second: fish 2.75-8.35 body lengths/s, whales 0.07-1.01 (set looking at test piece
  00006: exploratory). Sand (2025-01-02, 00017): no sand span has a whale before and after (10 spans), so no tracker
  test; the blocked rule marks 78 % of sand frames but also 60 % of open water via the `air` class, and with a
  sensor log it would not see sand at all.
Caveats: guards applied after the fact; "moved" on gap runs depends on the uncut run's own errors; thresholds partly
set on the clips they are scored on (dev clip and test 00006); by-eye checks one viewing. Outputs:
`review_sheets/underwater/demo_pure_0{1..5}_*.mp4/.jpg`; `experiments/07_tracks_demo/pure/` (runs, run_gpu,
evaluate, fb, sand, render, jobs/run_pure.sh, results/*.csv|json); `tables.md` sections C1-C4.

Reviewer's check (main thread, 2026-11-23 00:30): the table agrees with `pure/results/trade.csv`; sheets 01 and 05
looked at. "0 moved" is an in-sample number: strict was picked among three settings on the same 50 gap runs, and
those runs come from two recordings (dev clip, 00006 / 00007). It says the guards can reach zero there, not what
they do on new footage. Needed before the number is trusted: the policy run online (a new number starts from the
next detector box) on stretches it was not set on, second tag included. The calf in clip 05 stays `small` after
the fish's number ends on it (about 0.12-0.17 of the width): correct by the size rule, but a calf beside its
mother is a whale the user may want numbered; open.

## 2026-11-22 23:20  09_seafloor_sand (WP10 part 2): seafloor found in 541/548 and 305/306 windows of the other recording (0/282 and 25/981 false; same day, tag and bottom); false seafloor 22/1,736 and 6/821 windows on 2025-02-03, 1/91 on the dev clip; back called seafloor < 0.5 % of back pixels; sand does not carry beyond its one recording; dark = grey < 20, 0 false outside 00017
**WITHDRAWN (seafloor part): trained and scored on wrong seafloor labels (the `visible` spans show the back with haze above it); see the entry of 2026-11-23 01:10. The dark rule and the decode fix stand.**
Truth: AI by-eye labels (WP3b masks with seafloor on 15 frames, `seafloor_spans.csv`, `blocked_spans.csv`).
Seafloor exists on ONE deployment (2025-01-02, CC-09-85), in two recordings of the tagged whale resting on the same
sandy bottom: every seafloor number is leave-one-recording-out inside that deployment. Linear patch classifier on
frozen DINOv3 (1280 px, `uwid.patchseg`), classes water / whale / back / fish / bubbles / surface / air / seafloor /
gear; heads fitted on dev + 2025-02-03 + the other 2025-01-02 recordings' masks (test17 / test23), on dev + 2025-01-02
(test on 2025-02-03), on 2025-02-03 + 2025-01-02 (test on the dev clip). Windows = 2 frames at 0.25 / 0.75 s, 1280 px,
air breaths left out; window threshold on the seafloor patch share chosen on the training recording's windows.
- Seafloor per window: 00017 541/548 found, 0/282 false, unsure bottom-contact span 49/546 called (threshold 0.15);
  00023 305/306, 25/981 false (0.075; the resting view after the bottom faded). Other 2025-01-02 recordings 0-4 false
  of 220-1,499. Per pixel on held-out masks: 96.9 % / 79.5 % of seafloor found, IoU 0.87 / 0.78.
- False seafloor where none exists: 2025-02-03 00006 dense 22/1,736 windows (68 at a fixed 0.05), 2025-02-03 labelled
  6/821, dev clip 1/91. On sheets: a dark haze band just above the back in green water, painted where the training
  masks had their seafloor band.
- The back: back called seafloor 0.4 % (00017, 00023), 0.2 % (2025-02-03), 0.03 % (dev) of back pixels; seafloor
  called back 2.2 % / 1.7 %. Froth and bubbles called seafloor 0 % on the held-out 2025-01-02 masks (1.8 % on dev).
- Sand (00017 only; frame embedding + logistic regression): leave-one-100-s-block-out inside 00017 164/190 found,
  35/956 false. Not carried: left-out 00024 103/386, 2025-02-03 00006 728/1,736 and labelled 581/821 called sand
  (fitted on 2025-01-02 only). A low-dimensional variant was worse. Verdict: do not keep windows on `sand` alone yet.
- Dark (no model; every frame's mean grey < 20): 193/215 dark windows of 00017, 15 false (13 inside sand murk);
  0 windows dark anywhere else (6,666 windows: the other five 2025-01-02 recordings, 2025-02-03 dense and labelled, dev; darkest mean grey 50). Partly circular in 00017
  (the labeller placed dark ends with the same brightness measure).
- Delivered: `uwid.patchseg.window_classes(video)` -> per 1 s window seafloor, sand, dark, shares; rules in
  `patchseg.WINDOW_RULES` (seafloor share 0.15, dark grey 20, sand p 0.5); head `underwater_cache/seafloor/heads/
  head_all.pt` (all 74 mask frames), sand model `sand_all.npz` (2025-01-02 + other sources' windows as negatives,
  unscored). Cost from the run timings: 57 ms per 1280 px frame, about 0.11 GPU-h per video hour at 2 frames per
  window, plus 0.2-0.3 CPU-h of 4K decoding unless the scan's own decoded frames are reused.
- Incident: `cost.py` (a separate timing run) hung 1 h 43 min holding the GPU lock (no `__main__` guard; the spawned
  decode child died at start-up and the parent waited on its pipe); ended by the orchestrator, not repeated.
  `window_classes` now decodes through `patchseg._iter_decoded` (read timeout, child liveness check, raises) and logs
  each step; locked commands are wrapped in `timeout 900`.
Caveats: one deployment with seafloor, two static resting views, coarse band masks; sand in one recording; dark
spans placed with brightness. Outputs: `experiments/09_seafloor_sand/` (`decode.py`, `gpu.py`, `score.py`,
`sheets.py`, `cost.py`, `tables.md`, `results/*.json`), cache `underwater_cache/seafloor/`, sheets
`review_sheets/underwater/seafloor_01..05.jpg`.

Reviewer's check (main thread, 2026-11-23 01:10): counts agree with `results/seafloor.json`, `sand.json`,
`dark.json`. On `seafloor_01.jpg` the class is a band between the back and the open water; at sheet size I cannot
see a bottom in those frames, only haze. The held-out recording shares the day, the tag, the bottom and the resting
view, so this shows the class repeats within that situation, not that it finds seafloor in general. The user is
asked to confirm on the sheets that the band is what they mean by seafloor. Sand is a label for 00017 only.

## 2026-11-22 23:15  08_whale_fish_boxes (WP10 part 1): as a veto after the scan's rule, a linear box classifier on DINOv3 crops keeps 1-2 of 24 large near jacks as whales (the scan keeps 24/24) and loses 0 of 18 large whales on 2025-01-02 and removes none of the 163 whales the scan keeps on 2025-02-03; it calls 0 of 69 shallow blow / surface boxes whale (the scan keeps all 69)
Truth: AI by-eye labels: the reference labels of the three sources (dev 335, 2025-02-03 560, 2025-01-02 75 labelled
objects; ambiguous ones counted and left out), plus 84 extra 300-800 px boxes and 138 shallow breath boxes labelled
by eye by the WP10 agent from the scan's detector cache (test boxes). Objects = OWLv2 patch boxes >= 0.2 (scan
prompts) grouped at IoU 0.5. Model `uwid.boxclf`: tight + 2x context crop embeddings (DINOv3 ViT-B/16), geometry,
prompt scores; logistic regression; leave one source out (C chosen inside the training sources).
- Baseline (scan: whale >= 0.25 and >= 300 px): on 2025-01-02 all 4 reference fish and 24/24 extra jacks of 300-800
  px are whales; rule B (whale beats fish) 2/4 and 20/24. On 2025-02-03 no fish of 300 px or more exists in any label
  (all 24 "fishy" extra boxes there are whales).
- Classifier as veto (whale only if the rule and the classifier both say whale), large boxes: 2025-01-02 trained on
  dev + 2025-02-03: fish 4 -> 0/4, jacks 24 -> 2/24, whales 18 -> 16/21 kept (references), 18/18 (extra), other 12 ->
  4/27; trained on 2025-02-03 alone: jacks 24 -> 1/24. 2025-02-03 trained on dev + 2025-01-02: whales 163/163 of 168
  kept, 35/35 extra, other 7 -> 1/42. Trained on 2025-01-02 alone (75 boxes, 7 fish): 148/163 and 30/35 whales kept:
  too few examples. Dev clip: 61 -> 58 of 76 whales, other 11 -> 4 of 84.
- Feature sets on the extra jacks (dev + 2025-02-03 -> 2025-01-02): geometry + scores 11/24 still whale, tight crop
  3/24, tight + context 2/24. Step 4 (the box 0.5 s later) adds nothing on top of the crop features.
- Breath (orchestrator request): 138 boxes from hq / ok windows under 1 m (2025-02-03 27 whale / 26 other; 2025-01-02
  34 whale / 43 other / 3 fish; 7 unsure). The classifier not fitted on any of them calls 0/26 and 0/43 blow / surface
  boxes whale and loses 3/27 and 0/34 real shallow whales; re-fitted with the other deployment's breath boxes: 0/26
  and 4/43 blows kept, 3/27 and 0/34 whales lost, 300-800 px fish unchanged (2/24, 0/4). The re-fit is not kept.
- Below 300 px the classifier calls 18 of 108 small 2025-02-03 fish whale: keep the size rule there.
Delivered: `uwid.boxclf.classify(rows, frame)` -> class + probability per detector row; weights
`underwater_cache/boxclf/boxclf_v1.npz` (all reference + extra boxes; unscored), `boxclf_v1_with_breath.npz` beside it.
Caveats: the large-fish result rests on one deployment (2025-01-02, 28 fish >= 300 px incl. extras) and on my own
extra labels; reference truth for frames without masks uses a size-matched anchor box (2 truth errors seen on
sheets); the dev clip's whale truth comes from masks only. Outputs: `experiments/08_whale_fish_boxes/` (scripts,
`extra_labels.json`, `breath_labels.json`, `tables.md`, `results/`), `src/uwid/boxclf.py`, `tests/test_boxclf.py`,
cache `underwater_cache/boxclf/`, sheets `review_sheets/underwater/boxclf_fixed_01..05.jpg`,
`boxclf_wrong_01..05.jpg`, `boxclf_breath_01..02.jpg`.

Reviewer's check (main thread, 2026-11-23 01:10): breath counts agree with `results/breath.json` (blow / surface
boxes called whale 0/26 and 0/43 without any blow box in training; real whales under 1 m lost 3/27 and 0/34).
`boxclf_breath_01.jpg` and `boxclf_fixed_01.jpg` looked at: the by-eye truth on those two sheets is right as far as
I can see (blow, surface from below, mirrored back; jacks with forked tails). All 24 extra jacks come from one
recording (2025-01-02 00023), probably one school, so "2 of 24 kept" is one situation. 70 tests pass. Next: wire the
veto and the window classes into the scan (WP11b) and start tracks only from boxes the classifier calls whale.

## 2026-11-22 22:45  06_clips_e2e (WP9): `uwid scan` on a tag it was not built on (CC-09-85, 4K): clock now read on 8,480/8,480 frames, offset estimated (-0.04 s; 53 of 64 events), activity kept, tiers, floating time scanned, back filter; kept 16.6 of 140.8 min, no hq / ok whale frame lost
Second deployment `TestSubset 2025-1-2 CATS -85` (tag CC-09-85, 9 `.mov`, 4K, 140.8 min), nothing fitted on it
except its own clock glyphs and clock offset (measured from its frames / events by design) and the back filter (from
its kept windows: NOT clean). Truth: WP3b's AI by-eye labels (420 frames + spans); my own by-eye counts are AI too.
- What broke first: `-HRA-` names, the clock layout (DD/MM/YY + ms, 4K, proportional font: 0 frames read), mapping by
  file name (whole seconds), offset check "LOOKS WRONG" (40/108 events), 4K sheets with unscaled boxes, and the
  tagged whale's back (lower half of the frame) taken for a whale in 125 of 432 whale windows (122 of 247 hq).
- Fixed: `uwid.names` (extra part, sensor names); `uwid.timecode` finds a clock box by itself and learns digits from
  the video's file name + video time (`find_layout`, `bootstrap_from_name`, `glyphs_for`): 100 % of 8,480 sampled
  frames, monotonic, rms 13 ms (the clock's own jitter), slope within 17 ppm; per-file time link through each clock;
  offset check without surface brushes and on the exit / entry midpoint (this tag's lens: exits -0.29, entries
  +0.20 s): picture minus depth per recording -0.12 / +0.11 / -0.46 (2 events) / -0.08 / +0.01 / -0.26 s (5) (was
  -0.73..+0.73); default `--clock-offset tag` (known tag -> its offset, else estimated; stops when < 10 events).
- User decisions applied: keep anything detected (whale any size, fish >= 0.5); tiers hq / ok (>= 400 px, chosen on
  the dev clip) / low / small / fish as labels; `advanced_ok` + `hq_spans`; a description per clip (`clips.txt`,
  clip index); `--min-tier` optional; floating time scanned under water; silhouette flag (weak); sheets draw the
  decision (one box per object, scaled to 4K); empty `seafloor` / `sand` / `dark` columns for WP10.
- CC-09-85 (sensitive): kept 16.6 min (first run 15.6); boat recordings 0; frames with another whale 27/31 (hq 2/2,
  ok 18/18, low 7/11); activity 36/51 (5 lost where the labels say underwater but the sensors say "in air", 00010
  0-80 s); fish 7/7; empty kept 31/292; tier hq frames graded low / free 1/11. Dropped-window check (100, seed
  20261122): 0 large, 0 small whales, 2 fish, 7 black, 5 sand / murk, 16 in seafloor spans. Seafloor / sand / dark
  spans: the detector fires 0 times on sand (203 windows) or dark (215), on the bottom 0 of 24 looked at (20 real
  whales, 1 jack called whale); picture water = underwater throughout.
- 2025-02-03 re-scored: 66.7 -> 86.6 min, clear whale 199 -> 219/221, activity 215 -> 259/281, empty 28 -> 53/542;
  hq / ok frames in tier ok+ 92/93; tier hq frames graded low / free 8/44; silhouette 13/34 found, 27/187 false;
  00012 float: a large whale beside the floating tag now kept (4.0 -> 4.7 min).
- Still wrong: at breaths (depth < 1 m) the blow, the surface and the back mirrored in it are boxed as whale and
  tiered hq (71 of 174 hq windows on CC-09-85; 5 of the top 12); near jacks 360-524 px tiered ok; box size does not
  separate ok from low (36 of 128 low frames tiered ok+); the 00010 sensor / label disagreement.
- 4K: decode-bound (13 vs 23 frames/s), sizes are shares of the frame width, so the rules hold.
Outputs: `experiments/06_clips_e2e/` (`tables.md` sections 11-12, `clock_report.py`, `tier_rule_dev.py`,
`silhouette_rule_dev.py`, `float_check.py`, `evaluate.py --dep d2`, `sheets.py` new commands, `jobs/run_wp9*.sh`,
`results/clock_CC-09-85.json`, `eval_d2.json`, `dropcheck_d2_*`, `midsize.json`, `hq_bands_d*.json`), code
`src/uwid/{names,timecode,pipeline,clips,cli}.py`, tests `tests/test_names.py`, `test_timecode_dmy.py`,
`test_scan_wp9.py`; cache `underwater_cache/scan/<deployment>/` (pre-WP9 copies in `scan/_before_wp9/`, 00012 float
runs in `scan/_float_check/`), glyphs `underwater_cache/timecode/glyphs_DDMMYY_3840x2160.json`; sheets
`review_sheets/underwater/scan_d2_*.jpg` (timelines 00005 / 00017 / 00023, clips_middle / late, hq_top / border /
rejected, hq_band_*, dropcheck_01..17, sand, seafloor, seafloor_fired, float), `scan_hq_*` (re-drawn),
`scan_hq_band_*`, `scan_midsize_01..04`.

Reviewer's check (main thread, 2026-11-22 23:00; from `summary.json`, the window tables and the sheets):
- Kept time, tier counts, clock and offset numbers agree with the cache. 66 tests pass. No footage-derived file in
  the repo.
- The "00010 sensor / label disagreement" is not one: the labels call 0-80 s `air` (drone flight) with
  `faint_whales` 1-2 seen from above through the surface; sensors and the picture model say air too. The evaluation
  counted faint whales in air frames as activity. So 5 of the 15 lost activity frames are the drone's approach,
  dropped by design. The tag-state table has no drone flight and no drop for this tag (`drops.csv` empty): the flight
  is called "tag in air".
- Breaths tiered hq, by depth of the hq windows: CC-09-85 71 of 174 under 1 m (and 79 deeper than 30 m);
  2025-02-03 53 of 723 under 1 m (438 deeper than 15 m). The shallow rule only removes boxes over 0.6 of the frame;
  the blow boxes cover 0.26-0.38. A flat "no hq under 1 m" would cost real footage: of labelled frames under 1 m
  with a graded whale, 2 of 2 (2025-02-03) and 5 of 10 `ok` frames (CC-09-85) sit in tier hq today. So the blow has
  to be recognised as an object (box classifier class `other`), not cut by depth.
- Time link on this tag: the midpoint hides a wide exit / entry spread (00010: exits -0.70 s, entries +0.46 s), so
  the link is good to about 0.3 s per event, against 0.1 s on CC-11-96. Enough for 1 s windows.

## 2026-11-22 20:45  02_reference_labels (WP3b part B): second deployment CC-09-85 labelled (420 frames, 29 masks): seafloor only while the whale is on the bottom (00017, 00023; 854 s visible, all sand / sediment, faint); 00017 has 418 s of view blocked by sediment or a covered lens with no whale seen around it
AI by-eye labels by Claude, test deployment (scoring only, nothing fitted). Reduced plan from the orchestrator; no
helper agents. Input: the tag's 4K `.mov` read with `sample_mov.py` (uwid.video), frames saved at 1920 x 1080.
- Frame labels: 60 evenly spaced frames per on-whale recording (00010, 00011, 00012, 00017, 00023, 00024), 20 per
  boat recording (00005, 00006, 00009); all columns incl. size / quality; new `view` value `sand`. 420 frames: air 52
  (boat 60 frames are air; 7 drone), underwater 357, unusable 9 (black, 00017); other whale on 31 frames (00023 13,
  00024 15), quality hq 2 / ok 18 / low 11; fish few on 7 frames (8 jacks, all with forked tail, 120-560 px).
- Seafloor (`seafloor_spans.csv`): visible 00017 842-1390 s (sand plane fading into haze, whale resting at 47.5 m)
  and 00023 0-306 s (bottom fading into haze, 41.7 m); unsure 00017 296-842 s (bottom contact: sediment plumes,
  turbid water, black lens). None in 00010, 00011 (dives to about 50 m, open water only), 00012, 00024. No reef or
  rock. Frames: seafloor visible 35, unsure 25. Hard negatives: froth and brown water sheets that look like sand.
- Masks: 29 frames (21 about seafloor, 14 with another whale, 6 with fish), SAM 2.1 from my prompts, seafloor as a
  coarse band with an unsure band above.
- Blocked views (`blocked_spans.csv`, user request): 44 spans; 00017 sand 203 s + dark 215 s within 332-840 s, no
  other whale within 10 s of any of those pieces; 26 breaths (5-11 s) + the drone flight; another whale on both sides
  of a breath twice (00024, probably the same animal), after only once (00023). Sheets `reference_d2_<piece>_blocked`.
- Events, one window (00023 400-580 s, 0.2 s steps): 5 exits / 5 entries; the 1 s depth table agrees to about 1 s
  at every end but merges a double breath (2 s under at 445-447 s) into one surfacing.
Caveats: AI by-eye; the seafloor is faint (seen best with contrast stretching), its spans have 2-10 s ends; the
sand / dark split inside 00017 is to about 5 s (10 s steps + frame brightness); breath spans were found from the
sensor's surfaced intervals then checked by eye, so short underwater bubble blocks are not listed; WP7 reads these
files. Outputs: `experiments/02_reference_labels/labels/CC-09-85-*/`, README part B, `tables.md`, sheets
`/home/sat3737/NOAA_whales/review_sheets/underwater/reference_d2_*` and `reference_fish_candidates_d2_01..03.jpg`,
cache `/home/sat3737/NOAA_whales/underwater_cache/reference/CC-09-85-*/` (DONE in each).

## 2026-11-22 20:31  06_clips_e2e (user review): the user judged the windows at the hq threshold; two flagged, both traced to the sheet's drawing
The user looked at `scan_hq_border_01.jpg` (first deployment, 12 windows at hq 0.60) and at the labeller's
`reference_quality_examples` sheets. The user's words: "007 12:12:30.2 misses the large whale, and 006 11:20:53.1
calls 3 fish whale. The rest are looking good. The ref quality examples are all high quality whale sections as we had
hoped". This is the first human judgement of the quality tier: 10 of 12 border windows accepted as they are.
- 00007 12:12:30.2 (window 1772, frame 53135): checked by the orchestrator against the cached boxes. The whale box
  (993 x 1051 px) has whale score 0.96; the window is kept and `hq_whale`. The same box scores "small fish" 0.32 and
  "fish" 0.23, and the sheet drew the blue fish box over the red whale box. Not a miss by the tool.
- 00006 11:20:53.1 (window 482, frame 14470): the sheet drew four fish-sized "whale" boxes in red (287 x 195 px at
  0.54, 184 px at 0.34, 81 px at 0.25, 66 px at 0.20). The pipeline classed all four `small` (`small_score` 0.535);
  the window is kept for the large box (989 x 1017 px, 0.60). Not fish counted as whales by the tool, but the 287 px
  fish is 13 px under the size limit with whale 0.54 against fish 0.49, so a slightly nearer fish would pass, and the
  "whale beats fish" rule would not catch it. The tracking stage starts tracks from such boxes (WP8, clip 05).
- Consequences: sheets must draw the tool's decision, one box per object (WP9 is changing `sheets.py` and re-making
  the hq sheets); sheets of hq bands below 0.6 for the user to say where quality stops; a count and a sheet of kept
  windows that rest on a 300-500 px box. Open: which of the ten quality-example sheets the user looked at (01-03 are
  `hq`, 04-07 `ok`, 07-10 `low`).

## 2026-11-22 20:25  07_tracks_demo/gap (WP8b): cutting blocked views out of the tracker's eyes removes all outlines on bubbles / air and kept the calf across 4/4 breaths (3/4 straight through); a clean jump keeps 100 % of whales at 0.5 s, 72 % at 8 s (lone whale 31/32, 4+ whales 6/10 at 8 s)
The user's idea: detect bubble curtains and above-water moments and hide them from the tracker, so it meets a jump
instead of a blocked view. Tracker: OWLv2 boxes + SAM 3 tracker (WP8). `src/uwid/tracks.py` `run_tracks` gained
`hide` (frames not shown, consecutive tracker counter) and `gate` (frames shown, their memory entries deleted).
- Blocked-view rule, fixed before looking at tracker results (`gap/blocked.py`): lens out of water from the depth
  sensor (+-0.2 s) or, without a log, >= 25 % `air` patches; or >= 50 % of non-back patches `bubbles` (WP5b patch
  classifier; on the dev clip fitted on that clip, so only a tool there); runs < 0.2 s dropped, gaps < 0.3 s filled.
- Step 1, clean curve (identity truth = uncut run, no eye): 5 stretches (dev group 24-48 s, demo clips 03-06),
  g = 0.5/1/2/4/8 s at 2 positions, 50 cut runs, 105 whale cases: kept 18/18, 17/19, 15/18, 15/20, 13/18; lone
  whale 31/32 over all gaps; 4+ whales 10/10 -> 6/10 at 8 s with 3/22 swaps at 4-8 s; large 37/40 vs small 41/53.
- Real cases (variants a straight, b cut, c memory gated): dev bubble curtain (4.7 s episode), dev breath (8.7 s),
  4 calf breaths on 2025-02-03 (3.2-4.2 s; the only 4 of 66 surfacings with whales before and after). Outlines on
  blocked frames: a 324/795, c 431/795, b 0. Calf kept: a 3/4 (00007 1466 s: new number + a bogus track on the sea
  surface), b 4/4, c 4/4. Dev cases: no identity gain shown (re-attached numbers cannot be checked by eye; in the
  dev breath straight-through's extra "kept" numbers are stale outlines on wrong pieces, and #1 jumps from an
  exhalation cloud to a whale in all variants); b gives cleaner tracks after. Over-blocking seen once (whale visible
  through bubble streaks, 1 s hidden).
Verdict: use b; it helps a little (no outlines on blocked views, the one lost calf kept) and costs less GPU (5.7-10.3
vs 7.2-15.6 GPU s per video s), but cannot keep identities across multi-second blocks in a moving group.
Caveats: by-eye judgements by me (one viewing); cut runs had 2 s of history; group cases come from the development
clip only; 4 breath cases. Outputs: `review_sheets/underwater/demo_gap_0{1..6}_*.mp4/.jpg`, `demo_gap_curve.png`;
`experiments/07_tracks_demo/gap/` (cases, blocked, synthetic, real, run_gpu, jobs/run_gap.sh, results/*.json/csv),
`tables.md` sections B1-B3; runs under `underwater_cache/tracks/<stem>/{gap_*,real_*}`.

## 2026-11-21 07:40  07_tracks_demo (WP8): stage 2 = OWLv2 boxes start SAM 3 tracker tracks on stage-1 stretches; 7 outline videos (+2 SAM 2.1 re-renders): clean single-whale clips, one id for the calf through 20 s, failures shown in the bubble-curtain and hazy-calf clips; ~10 GPU s per video s
Chain: stage 1 = WP5a's cached OWLv2 run ("whale", every 15th frame, window = both frames >= 0.25); its runs of
whale windows >= 8 s were listed (`stretches.py`) and 7 stretches of 12-20 s picked from them (dev clip 24-36 s and
61-76 s; 2025-02-03 `.mov` pieces 00004, 00005, 00006 x2, 00007). Stage 2 (`src/uwid/tracks.py`): every frame, the
back = object 0 from a fixed bottom-25 % box; a stage-1 whale box (>= 0.10 of the width) starts a track unless a
track already explains it; cleaning (lies on the back, duplicates, fish-sized tracks drawn grey `sN small`). No
fitting; rules set once before the runs except the fish-sized-track rule (0.15 of the width, 90th pct), added after
clip 05. Truth: WP3 reference masks (AI by-eye labels) on 12 frames inside the stretches (40 whale instances); identity
and errors judged by eye by me on dense sheets (one viewing, no identity truth).
- Starting tracks, clips 01 + 05: (ii) OWLv2 boxes + SAM 3 tracker: 16/17 dev reference whales matched, median IoU
  0.80, calf 0.95, 13 GPU s/s; (i) SAM 3 video with text "whale": 16/17, 0.71, calf 0.98, ignores most fish, but
  thin masks hug the frame edges, a flipper becomes its own whale, 9-15 GPU s/s; (iii) OWLv2 + SAM 2.1 video:
  13/17, 0.75, calf 0.43, and on clip 05 the calf returning from behind the back is split and gets a NEW id; 8 GPU
  s/s. Chosen: (ii) (starts where stage 1 saw whales, no edge strips, kept the calf's id where SAM 2.1 lost it);
  its weakness is stage 1's fish-as-whale boxes.
- Per clip (ii): 03 one whale passing, 04 very near whale on the back, 06 far hazy whale approaching: 1 id each,
  nothing clearly wrong. 01 competitive group: 6 ids, 0 swaps, a corner fluke lost after 0.4 s, 2 two-whale joins
  (25.6-26.0 s, 29.8-30.4 s). 05 calf: id 1 for all 20 s incl. behind the back's hump (old id kept); 8 of 10
  tracks start on fish, 2 of them numbered as whales. 02 bubble curtain: 12 ids for ~6-7 whales, outlines on bubbles
  64-66 s, 2 ids re-attach after the curtain, the rest new; back IoU drops to 0.56-0.86. 07 hazy calf: FAILS: the
  one track starts on a fish and moves onto the calf (1 swap), the calf never gets a whale id (drawn grey small) and
  the back's outline swallows it at 824-826 s.
- Reference overlap by grade (clips 01, 02, 05, unmatched = 0): clear 13 median 0.90, hazy 21 0.77, faint 6 0.29.
  Back IoU 0.93-0.98 (dev, 4 frames), 0.56-0.86 in bubbles (7), 0.70 (test, 1).
- Cost (shared RTX 4000 Ada, every frame, model load not counted): 7.4-15.6 GPU s per video s (mean 10.1 over 118 s),
  1.5-3.1 GB peak (tracker history pruned to 24 frames). A 30-min 2025-02-03 piece: stage-1 whale windows 13-55 %
  (mean 24.8 %) -> about 1.2 GPU h (0.5-4.2), i.e. ~2.5 GPU h per video h, 8x stage 1; only windows with a whale box
  >= 0.3 of the width (8.9 %) -> 0.45 GPU h per piece.
Caveats: one viewing by eye; 12 reference frames, 5 of them in bubbles, only 1 on the test deployment inside a
stretch; the dev clip is the development video; transformers here lacks `kernels`, so SAM 3 video's hole filling is
off. Outputs: videos + sheets `/home/sat3737/NOAA_whales/review_sheets/underwater/demo_tracks_0{1..7}_*.mp4/.jpg`,
`demo_tracks_01_*_SAM2.1_same_start.mp4`, `demo_tracks_05_*_SAM2.1_same_start.mp4`; tracks
`underwater_cache/tracks/<stem>/<t0>-<t1>_<method>/`; code `src/uwid/tracks.py`, `tests/test_tracks.py`,
`experiments/07_tracks_demo/` (`stretches.py`, `clips.py`, `render.py`, `judge.py`, `report.py`, `jobs/run_all.sh`,
`tables.md`, `results/`).

## 2026-11-21 04:30  06_clips_e2e (WP7): `uwid scan` keeps 66.7 of 267 min of the 2025-02-03 deployment (25 %); every labelled large / ok / hq whale frame kept (103/103 size >= 0.3, 93/93 ok+hq), 199/221 clear-whale frames, 32/570 whale-free; no large whale in 150 dropped windows checked by eye
New command `uwid scan DIR` (`src/uwid/pipeline.py`, tests `tests/test_pipeline.py`): sensors first (tag state and
lens out of water from `tagdata`, attaching drop and release handled), sensor time = burned-in clock - 2.03 s with an
offset check against the picture's exits / entries, OWLv2 "whale" on 2 frames per 1 s window only on on-whale
underwater windows, presets `sensitive` (either frame) / `strict` (both), size limit 300 px (user's decision: smaller
= fish or far whale, reason `small`), whole-frame boxes kept at score >= 0.6 below 1 m depth, pad 2 s / merge < 5 s,
one drop reason per second, quality tier `hq_whale` per window and clip, clips via `uwid.clips`. Nothing fitted on test
video; the 1 m depth limit and the near-whale rule came from test failures (not clean). Truth: WP3 / WP3b AI by-eye labels.
- 2025-02-03 `.mov` 00004-00012 (9 x 29.7 min): sensitive keeps 66.7 min in 154 stretches (strict 58.2); dropped:
  no whale 183.3, small 9.4, tag floating 4.3, surfaced 1.8, drone flight 1.5, tag in air 0.1 min. Labelled 00004-00010
  (221 clear-whale / 570 whale-free frames): sensitive 199/221 and 32/570, strict 190/221 and 12/570; by WP3b size
  >= 0.3 103/103 both, 0.2 76/78 (strict 72/78), 0.1 20/40 (15/40); quality hq 11/11, ok 82/82, low 106/128 (97/128).
  The 22 lost frames (sensitive): 17 dropped as small, 5 as no whale; by eye none large, 3 borderline 240-300 px.
- Size limit 200 / 300 / 450 px: sensitive 213 / 199 / 152 of 221 frames, 75.1 / 66.7 / 48.6 min kept.
- Dropped-window check (AI by-eye, seed 20261120): 150 of 10,953 "no whale" windows: 146 empty, 1 fish, 2 small
  whales, 1 faint whale part, 0 large; 30 of 564 "small" windows: 20 small / far / cut whales, 9 fish, 1 ray-like.
- Options (once, not clean): near-whale rule adds nothing to the keep once the surface-mirror windows are excluded
  (before the 1 m limit it added 1.1 min, 20 of 22 shallow whole-frame windows were the tagged whale's mirrored back);
  fish veto: -1 whale frame, -7 whale-free (sensitive), -1.2 min. Fish score written, never used to keep.
- Clock offset check: 156/156 depth events paired, picture - depth +0.04 s (ok, implied 2.07 s); deliberately wrong
  offsets 1.53 / 3.03 / 12.03 s all flagged and all recover 2.07 s.
- 00012: release + rise and the 25-min float dropped as "tag floating"; but whales seen next to the floating tag are
  dropped too (2 of 18 float frames looked at). Dev clip (no log, picture water state): 145/145 whale frames kept,
  1.52 of 1.59 min; 4/180 underwater frames called surfaced.
- hq_whale (first guess, boxes not masks, scales from the dev clip): 717 windows / 11.9 min, hq stretches 23.0 min,
  95 of 154 clips; best 40 at 0.1-4.1 m depth; windows of WP3b's 11 `hq` frames all hq_whale, 25/82 ok, 1/128 low.
- Clips (sensitive, copy): 154 clips, 0.75 GB vs 2.16 GB source, 3 decoded: frame counts, lengths, first / last frames
  (NCC 1.0) and clock text match the index.
- Cost: detector 23 frames/s with the OWLv2 processor on the GPU (3.4 with it on the CPU; 120 frames: max score change 0.03, 0 decisions
  changed) = ~0.08 GPU-h per video-h; picture water ~0.04 GPU-h; whole chain ~9 min wall per video hour without the
  queue; the shared GPU lock queue dominated real time. Sensors first scanned 96.8 % of windows here (on-whale pieces).
- Broke and fixed: threaded HEVC decoding deadlocks next to torchvision (one thread in-process, child process for
  model stages); the near-whale rule counted the surface mirror of the tagged whale (1 m depth limit).
- Second deployment `TestSubset 2025-1-2 CATS -85` (tag CC-09-85, 9 files, 141 min of 4K, nothing fitted): the
  burned-in clock has another format (DD/MM/YY, ms) and is not read; the camera splits sessions into 4 GB files, so
  the log's camera-time pieces did not match file numbers and mapped deck video onto whale time (fixed: camera time
  only when it agrees with the file name within 3 s; all files now map by name); `-HRA` names not parsed by
  `uwid.names`. Sensors: 49.4 min deck, 0.6 floating, 90.8 on the whale (no drone flight). Offset check by file name:
  per-file -0.73 to +0.73 s, pooled -0.38 s over 34 events -> flagged (the names' whole-second start times). Detector
  run still queued for the shared GPU at 05:45 (`jobs/run_cats85.sh`, log `underwater_cache/scan/logs/run_cats85.log`).
- Outputs: `experiments/06_clips_e2e/` (`tables.md`, `evaluate.py`, `sheets.py`, `hq_scales.py`, `offset_demo.py`,
  `release_sheet.py`, `gpu_preprocess_check.py`, `jobs/`, `results/*.json`), cache `underwater_cache/scan/<deployment>/`
  (per preset: `windows/*.csv`, `intervals.json`, `dropped.csv`, `summary.json`; `offset_check.json`, `cache/`),
  clips `underwater_cache/clips/2025-2-3 Mini-mother 11-96/`, sheets `review_sheets/underwater/scan_*.jpg`
  (timeline_00004 / 00012 / dev, dropcheck_01..30, dropcheck_found_01..04, lost_sensitive_01..04, hq_top / border /
  rejected_01..04, wholeframe_sensitive_01..03, release_00012, clips_01..02).

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
2026-11-22: column `silhouette` added to every frames.csv (user: keep large silhouettes marked, grades unchanged): yes on dev 29,
2025-02-03 34, 2025-01-02 2 frames, all graded `ok`; 18 of the 27 large (size >= 0.5) `ok` frames; none of the hq / low frames.

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

