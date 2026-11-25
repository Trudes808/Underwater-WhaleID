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
uwid timecode FILE [--fps 2]               # read the burned-in clock (layout chosen or learned), fit against time
uwid timecode "<footage>/2025-12-20 CG.m4v" --bootstrap   # (re)learn the clock glyphs from the development clip
uwid sheet FILE [-n 40]                    # contact sheet of evenly spaced frames
uwid clips --intervals FILE.json [--mode copy|reencode|auto] [--max-lead 2] [--pad 0] [--merge-gap 0] [--dry-run]
uwid scan DIR [--preset sensitive|strict] [--cut [--min-tier ok]] [--mode copy] [--clock-offset tag|auto|SECONDS]
uwid tracks DEPLOYMENT [--preset sensitive] [--only REC:T0-T1 ...] [--max-seconds 30] [--render] [--lock-minutes 12] [--no-own-back-template]
uwid export [DEPLOYMENT ...] [--force] [--no-clips] [--out DIR]
```

- **tracks** (WP11a-WP18; `uwid.trackstage`) needs the GPU: run it as `flock /var/tmp/sat3737-parts-cache/gpu_train.lock
  timeout 900 nice -n 10 uwid tracks ...`; it skips finished spans, so a cut piece is simply run again.
- **export** (WP19; `uwid.export`) needs no GPU (it reads finished scan and tracker runs) and writes the web bundle
  `underwater_cache/export/web_v1/`; the contract is `docs/EXPORT_FORMAT.md` (copied into the bundle as `README.md`).

- **scan** (WP7, WP9; `uwid.pipeline`) takes a deployment folder (sensor log + `Video/`), a folder of deployments or
  one video. Per recording and per 1 s window of video time it decides: tag state (`air` / `flight` / `floating` /
  `whale`, from the sensor log), water state (under water / crossing / surfaced, from depth with the lens lags; from
  the picture model `uwid.waterstate` when there is no log, and on `floating` time, where the sensors cannot say where
  the lens looks), and, on windows on the whale or floating and under water, the OWLv2 scores of two frames (+0.25 s,
  +0.75 s) for "whale" and three fish prompts. **Keep rule (user, 2026-11-22): every window with anything detected is
  kept**: a whale box of any size >= 0.25 or a fish box >= 0.5 (`--fish-threshold`); `sensitive` (default) = either
  frame, `strict` = both. `--whale-only` keeps only large whales (the rule before 2026-11-22). A "whale" box of 300 px
  or more stays a whale only if the box classifier `uwid.boxclf` (WP10) says whale: "fish" makes it a fish, "other"
  (blow, surface, the back mirrored in it, sediment) drops it (`--no-boxclf` to switch off; without its weights the
  size rule decides alone). **Clips (user, 2026-11-23)**: a clip starts at the first active window and ends only after
  30 s with nothing detected (`--guard`), never across tag in air / flight or the lens out of the water for more than
  20 s (a breath does not end it), with a 2 s buffer each side (`--pad`); back-to-back files keep a clip each, marked
  `continues_in` / `continues_from`. Every window and kept stretch gets a **tier**, a label and not a
  filter: `hq` (= `hq_whale`: large share of the frame, detector score, brightness, contrast, sharpness, no bubbles,
  shallow; the only footage meant for 3D / identification, flag `advanced_ok` with its `hq_spans`), `ok` (whale box
  >= 400 px of a 1920 px frame, `--ok-min-px`, chosen on the dev clip), `low` (whale box 300-400 px), `small` (a
  "whale" box under 300 px, `--min-whale-px`: fish or far whale), `fish`, `none`. Each kept stretch also carries a
  one- or two-sentence description built from its window columns (clock times, depth range, tier seconds, largest
  whale size, boxes at once, fish, a breath of the tagged whale, floating, sand / dark seconds, and how many seconds
  have no detection). Dropped seconds get one reason: tag in
  air, drone flight, tag floating, surfaced, nothing detected, unreadable. Window labels from WP10's models on the scanned
  frames: `dark` (lens covered: every frame's mean grey < 20) and `sand` (stirred-up sediment; fitted in one
  recording); both are left out of the hq spans and change no keep decision. `seafloor` stays an empty column: the
  first model for it was withdrawn on 2026-11-23 (it marked whale backs and haze). These GPU stages cache per frame in
  `<cache>/scan/<deployment>/cache/pclass/`; `--gpu-budget-min` stops them for a resumable re-run. Two more labels: `silhouette` (a large whale box,
  dark inside, brighter around: a whale seen from below against the light; weak, see tables.md) and the tagged
  whale's own back is never a whale (a box across the full width resting on the bottom edge, `back_filter`; on
  CC-09-85 the back fills the lower half of the frame).
  **Clock and time link.** Sensor time = burned-in clock - clock offset, the clock read per file (two layouts known,
  found by themselves: `20Dec25 11:29:36.600` on CC-11-96, `02/01/25 15:45:02.475` in 4K on CC-09-85; a new layout is
  located and learned from the deployment's own frames and file names, see `timecode` below). `--clock-offset tag`
  (default) uses the measured offset of a known tag (CC-11-96: 2.03 s); for any other tag it is estimated from the
  picture's exits / entries against the depth's (the midpoint of the exit and entry medians, so a tag whose lens sits
  differently does not bias it) and **the scan stops with an error when fewer than 10 events (3 exits, 3 entries)
  match**; then give `--clock-offset SECONDS`. `offset_check.json` holds the estimate, the per-recording medians and
  the check after the offset is applied.
  Outputs `<cache>/scan/<deployment>/<preset>/`: `windows/<recording>.csv` (one row per second), `intervals.json`
  (`uwid-intervals/1`; keep intervals carry `tier`, `advanced_ok`, `hq_spans`, `description`; `hq` lists the hq
  stretches), `clips.txt` (the descriptions, readable), `dropped.csv`, `summary.json` (minutes per tier);
  `offset_check.json` one level up; with `--cut` every kept stretch is cut to `<cache>/clips/<deployment>/` with
  `clips_index.csv` (columns `tier`, `advanced_ok`, `hq_spans`, `description`); `--min-tier ok` cuts only the
  better ones (the others stay listed). GPU stages take the shared lock themselves (at most 12 min at a time) and a
  re-run reuses the clock fits, picture water states and detector boxes in `<cache>/scan/<deployment>/cache/`, so
  changing the preset, the tiers or the size limit costs seconds. Source footage is only read.
  **4K input** (CC-09-85, 3840x2160 HEVC): the detector decodes to 1920x1080 in a child process (all threads,
  about 130 frames/s sequential, 0.15-0.2 s per random frame), so boxes, the size limits (pixels of a 1920 px wide
  frame, i.e. shares of the frame width) and the quality measures mean the same as at 1080p; OWLv2 sees 960x960 either
  way. Decoding, not the model, sets the detector's speed at 4K.

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
| `timecode.py` | burned-in clock reader: two layouts (`DDMonYY`, `DD/MM/YY` with ms); `glyphs_for(path)` picks the stored glyph set that reads a video or finds the text box (`find_layout`) and learns the digits from the video's own frames and file name (`bootstrap_from_name`); `read_clock(image)`, `read_video_clock(path, fps / every_s)` with a robust line fit |
| `names.py` | tag file names (`CC-11-96-20250203-101236-000-00004`, `CC-09-85-HRA-20250102-111224-000-00005`: tag, optional extra part, local start time, sequence number; sequence numbers can have gaps), sensor file names, date hints |
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
- The clock reader needs glyph files in `<cache>/timecode/`: `glyphs.json` is made from the development clip on first
  use (months it has not seen are taken from the date in the file or folder name); `glyphs_DDMMYY_3840x2160.json`
  was learned from CC-09-85's first recording and its file name (WP9). A video none of them reads gets its own,
  learned the same way, when its file name follows the tag convention.
- Tag CC-09-85 (4K): file names carry an extra part (`-HRA-`) and whole seconds (the clock runs -0.5 to +0.6 s from
  the name); the camera splits a recording session into 4 GB files, so sequence numbers have gaps and do not match
  the sensor log's camera pieces; its burned-in clock is about 0.1 s behind the sensor log (CC-11-96: 2.03 s ahead).
- Decoding the tag's HEVC `.mov` with several threads deadlocks once torchvision is loaded in the same process
  (PyAV 19 + torchvision 0.21, found 2026-11-20). `uwid.video` then decodes with one thread (about 200 frames/s at
  1080p, enough for the scanner); scripts that import torchvision first get this automatically.
