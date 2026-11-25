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
