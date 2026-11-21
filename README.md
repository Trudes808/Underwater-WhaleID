# Underwater-WhaleID

Underwater whale video from camera tags attached to whales. Phase 1 (see `PLAN.md`): detect whether the camera is
underwater, when it surfaces, and whether a whale or fish is in frame, then cut the footage worth keeping into clips.
Working rules (where data may be written, shared GPU / CPU / disk): `CLAUDE.md`. Results: `experiments/LOG.md`.

## Setup

There is no `ensurepip`, no system ffmpeg and no sudo on the host. ffmpeg comes from the `imageio-ffmpeg` package.

```bash
~/.local/bin/virtualenv -p python3.12 .venv
.venv/bin/pip install --no-cache-dir -r requirements.txt   # torch 2.6.0 / torchvision 0.21.0 cu124, transformers 5.20.1, PyAV ...
.venv/bin/pip install -e .
.venv/bin/python -m pytest -q                             # synthetic videos only, about 15 s
```

`.venv/READY` marks a working environment for other agents.

## Where data lives

| what | where | set by |
|---|---|---|
| footage and sensor files (read only) | `/home/sat3737/NOAA_whales/underwater_video/` | `footage_dir` in `configs/paths.toml`, env `UWID_FOOTAGE_DIR` |
| derived data: inventory, clips, clock glyphs and readings | `/home/sat3737/NOAA_whales/underwater_cache/<topic>/` | `cache_dir`, env `UWID_CACHE_DIR` |
| contact sheets and other images to look at | `/home/sat3737/NOAA_whales/review_sheets/underwater/` | `sheets_dir`, env `UWID_SHEETS_DIR` |

Nothing made from the footage goes into this repo. The tools refuse to write inside the footage folder.

## Commands

```bash
uwid probe FILE [--decode] [--keyframes]   # metadata, real frame count, frame rate, keyframe spacing, edit lists
uwid inventory [DIR]                       # every video under DIR (default: footage_dir), see below
uwid timecode FILE [--fps 2]               # read the burned-in clock, fit clock against presentation time
uwid timecode "<footage>/2025-12-20 CG.m4v" --bootstrap   # (re)learn the clock glyphs from the development clip
uwid sheet FILE [-n 40]                    # contact sheet of evenly spaced frames
uwid clips --intervals FILE.json [--mode copy|reencode|auto] [--max-lead 2] [--pad 0] [--merge-gap 0] [--dry-run]
uwid scan DIR [--preset sensitive|strict] [--cut] [--mode copy] [--clock-offset 2.03|auto]   # folder -> kept clips
```

- **scan** (WP7, `uwid.pipeline`) takes a deployment folder (sensor log + `Video/`), a folder of deployments or one
  video. Per recording and per 1 s window of video time it decides: tag state (`air` / `flight` / `floating` /
  `whale`, from the sensor log), water state (under water / crossing / surfaced, from depth with the lens lags; from
  the picture model `uwid.waterstate` when there is no log), and, only on windows on the whale and under water, the
  OWLv2 "whale" score of two frames (+0.25 s, +0.75 s). Presets: `sensitive` keeps a window when either frame has a
  large whale box >= 0.25, `strict` when both do. A box shorter than `--min-whale-px` (300 px of 1920) is `small`
  (fish or far whale) and keeps nothing; a box covering most of the frame counts when its score is >= 0.6 (very near
  whales). Whale windows are padded by 2 s and stretches closer than 5 s (after padding) merged. Every other second
  gets one reason: tag in air, drone flight, tag floating, surfaced, underwater with no whale, small (fish or far
  whale), unreadable. A quality tier `hq_whale` (large share of the frame, detector score, brightness, contrast,
  sharpness, no bubbles, shallow depth; first guess, see `experiments/06_clips_e2e/tables.md`) is written per window
  and per clip. Sensor time = burned-in clock - `--clock-offset` (2.03 s measured on tag CC-11-96); the **offset
  check** compares the picture's exits / entries with the depth's and prints the median difference: if it says
  `OFFSET LOOKS WRONG`, re-run with the offset it suggests or `--clock-offset auto`.
  Outputs `<cache>/scan/<deployment>/<preset>/`: `windows/<recording>.csv` (one row per second), `intervals.json`
  (`uwid-intervals/1`, keep intervals carry `tier`; `hq` lists the hq stretches), `dropped.csv`, `summary.json`;
  `offset_check.json` one level up; with `--cut` the clips go to `<cache>/clips/<deployment>/` with
  `clips_index.csv` (column `tier`). GPU stages take the shared lock themselves (at most 12 min at a time) and a
  re-run reuses the clock fits, picture water states and detector boxes in `<cache>/scan/<deployment>/cache/`, so
  changing the preset or the size limit costs seconds. Source footage is only read.

- **inventory** pairs the `.mov` / `.m4v` forms of each recording, lists the sensor files of each deployment folder,
  probes and decodes every video once, and writes `<cache>/inventory/inventory.json`, `inventory.md` and one
  contact sheet per video (`<sheets>/inventory_<stem>_<ext>.jpg`, 40 tiles with time and clock). Files modified in
  the last 60 s (still being copied) are skipped and listed. A re-run reuses unchanged entries.
- **timecode** writes `<cache>/timecode/<stem>_clock.csv` (per frame: index, time, clock text, residual, flag
  `unreadable` / `outlier`) and a JSON summary (share read, monotonic, slope, largest residual, offset to the start
  time in the file name).
- **clips** writes clips to `<cache>/clips/<source stem>/` and appends to `<cache>/clips/clips_index.csv` and
  `dropped.csv`. Source footage is only read.

### Intervals file (input of `uwid clips`)

The detection work fills this file; times are presentation times in seconds as `uwid.video` reports them (the frame
with index `i` in `uwid.video.frame_table(path).times[i]`), an interval holds the frames with `start <= t < end`.

```json
{
  "format": "uwid-intervals/1",
  "videos": [
    {
      "source": "2025-2-3 Mini-mother 11-96/Video/CC-11-96-20250203-101236-000-00004.mov",
      "keep": [{"start": 100.5, "end": 112.0, "reason": "whale in view"}],
      "drop": [{"start": 0.0, "end": 100.5, "reason": "no whale"}, {"start": 112.0, "end": 1781.1, "reason": "camera in air"}]
    }
  ]
}
```

`source` is absolute or relative to the footage folder. Extra keys on an interval (scores, labels) are kept as JSON
in the `extra` column. `drop` intervals go to `dropped.csv` as given; keep intervals that cannot be cut are listed
there too with `NOT CUT (...)`.

Modes: `copy` does not re-encode the video (the sound is re-encoded to AAC so it is trimmed exactly); the clip starts
on the keyframe at or before the wanted start, so nothing wanted is lost, and the index gives the real first and last
frame. `reencode` is frame exact (x264 CRF 17, AAC sound). `auto` copies when the keyframe before the start is at most
`--max-lead` seconds early, else re-encodes. `--pad` widens every keep interval, `--merge-gap` joins intervals closer
than that.

## Package layout (`src/uwid/`)

| module | what |
|---|---|
| `paths.py` | footage / cache / sheet folders from `configs/paths.toml` and environment variables |
| `video.py` | `probe`, `frame_table` (shown frames from demuxing, edit lists respected), `decode_check`, `iter_frames(path, fps=2.0)`, `VideoReader.frame_at(t)` (keyframe seek + decode forward), `keyframes`, MP4 box reader (edit lists, fragments, truncation) |
| `audio.py` | `read_audio(path, start, end)`: mono float32 at the native rate |
| `timecode.py` | burned-in clock reader: glyph templates learned from the development clip (stored in the cache), `read_clock(image)`, `read_video_clock(path, fps)` with a robust line fit |
| `names.py` | tag file names (`CC-11-96-20250203-101236-000-00004`: tag, local start time, sequence number), date hints |
| `sheets.py` | contact sheets: grid, time labels, captions, coloured mask overlays |
| `inventory.py` | footage folder inventory |
| `clips.py` | clip cutter with the bundled ffmpeg |
| `cli.py` | the `uwid` command |
| `tagdata.py` | tag sensor data (WP1); `derive_tables` makes states / drops / surfacings for a new deployment |
| `waterstate.py`, `eval_state.py` | water state from the picture (WP4) and its scoring |
| `detect.py`, `eval_presence.py`, `patchseg.py` | text-prompted detectors and window scoring (WP5a), patch classifier (WP5b) |
| `pipeline.py` | `uwid scan`: discovery, sensor timeline, clock fit and offset check, detector cache, keep rule, quality tier (WP7) |

## Notes on the footage (2026-11-20)

- Times: everything is in presentation time after the container's edit list. The development clip's edit list hides
  161 frames at the start (clock 11:29:31.3 to 11:29:36.5) and 2 at the end, so 2,857 of its 3,020 frames are shown.
- The tag's own `.mov` (HEVC, fragmented) is the better input: complete, its timestamps follow the burned-in clock to
  within 17 ms over 30 min, keyframes every second. The `.m4v` conversion has the same frames but loses the last
  ~41, runs on a constant 29.95 frames/s timeline that drifts 1.4 s from the clock over 30 min, and has its
  brightness squeezed (`m4v ≈ 0.83 × mov + 17`).
- The clock reader needs the glyph file `<cache>/timecode/glyphs.json`; it is made from the development clip on
  first use. Months it has not seen (only `Dec` so far) are taken from the date in the file or folder name.
- Decoding the tag's HEVC `.mov` with several threads deadlocks once torchvision is loaded in the same process
  (PyAV 19 + torchvision 0.21, found 2026-11-20). `uwid.video` then decodes with one thread (about 200 frames/s at
  1080p, enough for the scanner); scripts that import torchvision first get this automatically.
