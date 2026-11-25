# 12_export (WP19): web export bundle `uwid-web-export` v1

Format: `docs/EXPORT_FORMAT.md` (copied into the bundle as `README.md`). Bundle: `underwater_cache/export/web_v1/`.
First full run 2026-11-24 21:34-21:40 (`uwid export`, all scanned deployments, CPU only, nice 10, 8 threads; 5.3
min), with the tracks as they were before WP18's final change (outlines will change slightly; re-run after WP18).

## Counts and sizes

| deployment (id) | recordings | windows.csv | kept stretches | hq spans | tracked spans hq / extra | whale numbers |
|---|---|---|---|---|---|---|
| 2025-12-20_CG (dev clip) | 1 | 1 | 1 | 10 | 0 / 0 | 0 |
| 2025-02-03_CC-11-96 | 9 | 9 | 100 | 118 | 118 / 16 | 144 (hq 131, extra 13) |
| 2025-01-02_CC-09-85 | 9 | 9 | 27 | 17 | 17 / 1 | 19 (hq 18, extra 1) |

| file type | files | size |
|---|---|---|
| clips `.mp4` (H.264 1280 x 720) | 152 | 112 MB |
| posters `.jpg` | 152 | 6.8 MB |
| track `.json` | 152 | 31 MB (largest 1.1 MB: 00023 252-271, 4K, 563 frames) |
| `windows/*.csv` | 18 | 3.4 MB |
| clip `.check.json`, `index.json`, `stretches.json`, `README.md` | 152 + 5 | < 1 MB |
| **bundle** | | **154 MB** (budget 1.5 GB) |

Outlines in the track JSON: 27,331 outlined (number, frame) pairs, 29,755 polygons, 1,517,734 points.

## Checker (`check_bundle.py`)

- Part (i), `/usr/bin/python3 -I` (3.12.3, standard library only): 3 deployments, 19 recordings, 19 windows files,
  152 spans / track files / clips; **0 problems** (fields, paths, frame lists, polygons and boxes in 0..1, CSV
  columns).
- Part (ii), `.venv`: 152 clips decoded: frame count and size equal to the track JSON in **152 / 152** (0 padded
  frames). Polygon accuracy, 20 random (span, number, frame) picks, seed 20261124, exported polygons rasterised at
  1280 x 720 against `tr.mask` of the cleaned tracks: **IoU min 0.9862, median 0.9939** (cost of the 1.5 px
  simplification and of dropping holes).
- Review picture `review_sheets/underwater/export_check_01.jpg`: 12 random frames (seed 20261125) drawn only from the
  bundle (clip frame + JSON polygons); outlines sit on the whales in all 12 (my look, one viewing).

## Media facts (summary; full tables in EXPORT_FORMAT.md)

| files | codec / tag | size | browser as is |
|---|---|---|---|
| 2025-02-03 `.mov` (9) | HEVC `hev1` | 1920 x 1080 | no |
| 2025-02-03 `.m4v` (9) | H.264 `avc1` (yuvj420p) | 1920 x 1080 | yes |
| 2025-01-02 `.mov` (9) | HEVC `hvc1` | 3840 x 2160 | Safari only |
| dev clip `.m4v` (1) | H.264 `avc1` | 1920 x 1080 | yes |
| scan clips `underwater_cache/clips/` (158) | HEVC `hev1` 155, H.264 3 | 1920 x 1080 | 3 of 158 |
| review videos (40) | H.264 `avc1` | 1280 x 720 / 1920 x 540 | yes |
| export clips (152) | H.264 `avc1` yuv420p | 1280 x 720 | yes |

## Final run (2026-11-24 evening, after WP18)

**Dev clip tracked** (`uwid tracks 2025-12-20_CG`, default settings as of WP18, nothing tuned; one locked piece
21:44:45-21:52:46, 416 GPU s of tracking for 45.5 s tracked; `experiments/12_export/run_dev_tracks.sh`). Before it,
`trackstage` could not find the dev clip's scan files (its file name has a space, the scan stores it with `_`), so
the tracker would have had no start boxes: fixed with `trackstage.scan_key` (file lookup only, no rule changed).
The old dev run `0024.00-0036.00_online` (WP11a, WP5a start boxes) was moved aside as `*_pre_wp19`. No sensor log on
the dev clip: blocked frames come from the air-share rule. The own-back template was not used on any dev span
("fewer than 5 frames in the block"), so the back is the tracker's whole back object there, as before WP18.

| deployment | spans | video s | numbers (before -> after links) | numbers >= 3 s | blocked frames | links made | links not made (why) | endings (reason) | starts refused (reason) | GPU tracking min |
|---|---|---|---|---|---|---|---|---|---|---|
| dev clip 2025-12-20 | 10 | 45.5 | 46 -> 45 | 16 (median life 1.07 s; 21 under 1 s) | 100 | 1 | not lone 82 | appearance drift 12, shape jump 4, area jump 4, swallowed another track 2, re-found after 1 | surface / bubbles / air 11, box classifier other 2, fish 1 | 6.9 |

What it shows: with many whales in view the purity rule ends numbers early and nothing may join them (82 of 83 link
candidates were "not lone"), so the dev clip has many short numbers: 45 numbers in 45.5 s of tracking, the
25 s span alone 14.

**Bundle after the final run** (JSON / CSV rewritten for all with the WP18 clean-up; clips of the 152 earlier spans
kept, 10 dev-clip clips made; paths without spaces):

| deployment | recordings | windows.csv | kept stretches | hq spans | tracked spans hq / extra | whale numbers |
|---|---|---|---|---|---|---|
| 2025-12-20_CG (dev clip) | 1 | 1 | 1 | 10 | 10 / 0 | 45 |
| 2025-02-03_CC-11-96 | 9 | 9 | 100 | 118 | 118 / 16 | 144 |
| 2025-01-02_CC-09-85 | 9 | 9 | 27 | 17 | 17 / 1 | 19 |

Sizes: clips 162 files 128 MB, posters 7.4 MB, track JSON 162 files 34 MB, windows CSV 19 files 3.5 MB; bundle
**173 MB**. Group `hq` is now "what `uwid tracks <deployment>` runs": on the two test deployments it is exactly the
135 WP13 spans (checked: 17 = 17 and 118 = 118, no folder differs). `extra` comes from
`underwater_cache/tracks/extra_stretches.csv` (17 rows, its `note` carried into index and track files).

Checker: (i) standard library, 0 problems over 3 deployments, 19 recordings, 19 windows files, 162 track files /
clips (31,189 outlined frames, 34,036 polygons, 1,681,625 points); (ii) 162 / 162 clips decode to the JSON's frame
count and 1280 x 720; polygons vs masks, 20 picks, seed 20261124 (the same seed draws other picks now that the
span list has the dev clip): **IoU min 0.9793, median 0.9921**; the lowest is a small outline (8,180 px at
1280 x 720: 00008 5-16 s), where the 1.5 px simplification costs most. Review picture
`review_sheets/underwater/export_check_02.jpg` (seed 20261125, first 3 tiles from the dev clip).
