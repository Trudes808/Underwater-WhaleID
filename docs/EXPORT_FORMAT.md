# Underwater-WhaleID web export (`uwid-web-export`, version 1)

Plain files (JSON, CSV, H.264 MP4, JPEG) with the results of the camera-tag video analysis, made for a web page
that draws the tracker's whale outlines over the playing video. Nothing here needs this repo's Python environment:
JSON and CSV read with any standard library, the MP4s play in any browser.

Made by `uwid export` (code `src/uwid/export.py`); checked by `experiments/12_export/check_bundle.py`. This file is
copied into every bundle as `README.md`.

## Regenerate

```
cd Underwater-WhaleID
nice -n 10 .venv/bin/python -m uwid.cli export                      # every scanned deployment
nice -n 10 .venv/bin/python -m uwid.cli export "<deployment>"       # one deployment (others stay in index.json)
            ... --force       # also re-make existing clips / posters (default: kept; JSON and CSV always rewritten)
            ... --no-clips    # JSON / CSV only
            ... --out <dir>   # default underwater_cache/export/web_v1
/usr/bin/python3 -I experiments/12_export/check_bundle.py structure  # standard library check
.venv/bin/python experiments/12_export/check_bundle.py media         # clips + polygon accuracy + review picture
```

A full run (all three deployments, clips included) takes about 6 minutes on an idle machine and needs no GPU (it
only reads finished tracker runs; tracking itself is `uwid tracks`).

## Conventions

- **Time**: seconds of the source recording's own presentation clock (`t = 0` = first shown frame), as in every
  table of this project. `clock` columns / fields are the burned-in clock of the video (read by OCR and fitted
  linearly; local time of the tag, `HH:MM:SS.s`), empty when it could not be read. Known clock facts: on 2025-02-03
  the burned-in clock is the tag's sensor time + 2.03 s (on 2025-01-02 it agrees with the sensor time to 0.04 s);
  on the dev clip (2025-12-20) the burned-in clock reads about two hours later than the field notes' time for that
  group: an open question with the user, not resolved here, so do not convert dev-clip clock times to field-note
  times.
- **Frames**: `frames.index` are 0-based indices among the shown frames of the source recording.
- **Coordinates**: fractions of the frame width / height, origin top left, x to the right, y down, so the same
  numbers fit any display size (multiply x by the drawn width, y by the drawn height). Polygons are lists of
  `[x, y]` points, closed implicitly (last point joins the first). Coordinates have 4 decimals.
- **Outlines**: the tracker's masks are kept at 1280 x 720 for every source size. Each mask is turned into the outer
  outlines of all its pieces (`polygons`: one polygon per separate piece; holes are not kept) and simplified with
  Douglas-Peucker at `simplify_px` = 1.5 px of the 1280 x 720 mask. Measured cost: see the checker's IoU in
  `experiments/12_export/tables.md`.
- **Number**: a whale number is **per tracked span, not per animal**. Numbers start at 1 in every span in order of
  first appearance. The tracker follows a purity rule: a number must never move from one animal to another, so when
  in doubt it ends a number (see `endings`) and the next sighting gets a new number; the same animal can therefore
  carry several numbers in one span. Two numbers are joined only by the lone-link rule (one whale in view, short
  gap, size and appearance agree; see `links`). Numbers are not identities across spans, recordings or days.
- **Model output vs sensor**: fields marked *sensor* come from the tag's depth / state log; *model* from the image
  models (OWLv2 detector, picture water model, patch classifier); *rule* from the scan's rules on both.
- Booleans in CSV are `1` / `0`; empty = not available.

## Layout

```
web_v1/
  README.md                                   this document
  index.json                                  entry point
  <deployment id>/                            e.g. 2025-02-03_CC-11-96 (date of the recordings + tag serial)
    stretches.json                            the scan's kept / dropped stretches per recording
    windows/<recording>.csv                   the scan's per-second table of one recording
    tracks/<group>/<recording>/<t0>-<t1>.json one tracked span: frames, numbers, outlines, events
    clips/<group>/<recording>/<t0>-<t1>.mp4   the same span as a plain video, frame j = track frame j
    clips/<group>/<recording>/<t0>-<t1>.jpg   poster (the frame with the largest outlines)
    clips/<group>/<recording>/<t0>-<t1>.check.json  frame count of the clip as decoded after writing
```

`<recording>` = recording file name without extension (e.g. `CC-11-96-20250203-104242-000-00005`); `<t0>-<t1>` =
the span asked for, in seconds with 2 decimals and leading zeros (`0332.00-0348.00`). All paths inside the JSON files
are relative to the bundle folder.

### Groups of tracked spans

- `hq`: exactly what `uwid tracks <deployment>` runs on that deployment: every hq span of a kept stretch with
  `advanced_ok` in the scan, at most 30 s around its centre, tracked from its first detector start box (only
  finished runs). On the two test deployments this is the same 135 spans as the all-hq-spans run (WP13); the dev
  clip adds its 10.
- `extra`: further stretches tracked on request, listed in `underwater_cache/tracks/extra_stretches.csv` (deployment,
  recording, t0, t1, folder, `note`); the `note` (why the stretch was tracked, e.g. "calf stretch (user-confirmed
  calf)") is carried into the index and the track file. The same recording time can appear in both groups.

## `index.json`

| field | meaning |
|---|---|
| `format`, `version`, `kind` | `"uwid-web-export"`, `1`, `"index"` |
| `created`, `command` | local time of the run (ISO), the command that made it |
| `readme`, `coordinates` | `"README.md"`, the coordinate convention in words |
| `deployments[]` | one entry per deployment, below |

Per deployment: `id` (folder name), `name` (the footage folder name), `tag` (tag serial, e.g. `CC-11-96`), `date`
(`YYYY-MM-DD` from the recording names), `preset` (`"sensitive"`: the scan setting used), `sensor_log` (whether a
depth / state log exists), `stretches` (path), `counts` (`recordings`, `kept_stretches`, `hq_spans`,
`tracked_spans` per group, `numbers` = sum of numbers over its spans), `recordings[]`, `spans[]`.

Per recording: `recording` (name), `file`, `source` (path under `underwater_video/`; the original, read-only; HEVC or
H.264, see Media facts), `duration_s`, `n_frames`, `fps`, `width`, `height`, `codec`, `codec_tag`, `container`,
`first_frame_clock` (burned-in clock at t = 0, or null), `clock_fit` (`ok`, `kind`, `rms_s` of the clock fit),
`other_forms` (other files of the same recording, e.g. a `.m4v` next to the `.mov`; the scan used `file`),
`windows` (path of its `windows.csv`, or null).

Per span: `group`, `note` (extra group: why it was tracked; null for hq), `recording`, `t0`, `t1` (span asked for), `first_t`, `last_t` (times of the first / last tracked
frame), `n_frames`, `numbers` (count of whale numbers), `links_made`, `track`, `clip`, `poster` (paths).

## `windows/<recording>.csv` (one row per second of the recording)

| column | kind | meaning |
|---|---|---|
| `time_s`, `end_s` | time | window start / end (s) |
| `clock` | clock | burned-in clock at `time_s`, `HH:MM:SS.s`, empty if unread |
| `depth_m` | sensor | tag depth (m, positive down; about 0 at the surface) |
| `tag_state` | sensor | `whale` (on the whale), `flight` (drone flight before the deployment), `floating`, `air` |
| `water_state` | rule | `underwater`, `crossing`, `surfaced`, `air`, `waterline` (from depth, the tag state or the picture) |
| `water_source` | rule | what `water_state` came from (`depth`, `tag state`, `picture`) |
| `water_picture`, `p_air_picture` | model | the picture water model: `underwater`, `crossing`, `air`, `unknown` (not run), and its probability of air |
| `scanned` | rule | the detector looked at this window (only under water, tag on the whale) |
| `whale`, `whale_score`, `n_whale` | model | a whale decision box in the window, its score, how many whale boxes |
| `whale_px` | model | longest side of the largest whale box, px of a 1920 px wide frame (>= 300 = whale-sized) |
| `small`, `small_score` | model | only small / far whale-like shapes (fish-sized) |
| `fish`, `fish_score`, `n_fish` | model | fish seen, score, number of fish boxes |
| `active` | rule | something was detected (whale, small or fish) |
| `tier` | rule | `hq`, `ok`, `low` (whale picture quality), `small`, `fish`, `none` |
| `kept` | rule | the window belongs to a kept stretch |
| `why` | rule | why kept or not: `whale`, `small`, `fish`, `padding` (kept next to activity), `nothing detected`, `tag in air`, `tag floating`, `drone flight`, `surfaced`, `unreadable` |
| `seafloor_share` | model | share of the picture the seafloor model calls bottom (not reliable on 2025-01-02, see LOG) |
| `sand`, `dark` | model | sand / dark-water flags of the scan |
| `silhouette` | model | a large whale in silhouette against the light |
| `bubbles` | model | share of bubble patches |
| `hq_score` | model | the picture-quality score behind the `hq` tier |
| `whale_box_x0..y1` | model | the main whale box, fractions of the frame |

## `stretches.json`

`{format, version, kind: "stretches", deployment, preset, scan_created, recordings: {<recording>: {...}}}`; per
recording: `duration_s`, `kept[]`, `dropped[]`, `hq_clips[]`. A kept stretch: `start`, `end` (s), `reason`, `tier`
(best), `advanced_ok` (good enough for 3D / identification), `hq_whale`, `hq_spans` (`[[start, end], ...]`, the
spans the tracker ran on), `description` (the scan's text: clock times, depth, what was seen), `active_s`, `quiet_s`,
`whale_windows`, `hq_windows`, `tier_windows`, `silhouette`, `silhouette_windows`, `sand_s`, `dark_s`,
`max_whale_score`, `max_fish_score`. A dropped stretch: `start`, `end`, `reason` (`tag in air`, `drone flight`,
`surfaced`, `nothing detected`, ...). `hq_clips`: `start`, `end` of the hq clips the scan cut.

## `tracks/<group>/<recording>/<t0>-<t1>.json`

| field | meaning |
|---|---|
| `format`, `version`, `kind` | `"uwid-web-export"`, `1`, `"track"` |
| `group`, `note`, `deployment`, `recording` | as in the index |
| `span` | `{t0, t1}` asked for; `tracked` = `{t0, t1}` times of the first / last tracked frame |
| `fps`, `n_frames` | frame rate of the source, number of tracked frames |
| `frames` | `{index: [...], t: [...]}`: source frame index and time (s) of track frame j = 0..n_frames-1 |
| `clip` | `{path, poster, width: 1280, height: 720, fps, n_frames, correspondence, check}`: `check` = `{decoded_frames, size, padded, ok}` measured after writing |
| `coordinates`, `simplify_px`, `mask_size`, `back_every` | see Conventions; `mask_size` = [1280, 720]; the back every `back_every` frames |
| `numbers[]` | whale numbers (below) |
| `small[]` | fish-sized tracks (label `s1`, `s2`, ...), same structure; shown as "small" in the review videos |
| `back[]` | the tagged whale's own back: `{j, polygons}` for j = 0, `back_every`, 2 x `back_every`, ... |
| `blocked[]` | `{j0, j1, t0, t1}`: frames j0 <= j < j1 hidden from the tracker (lens out of the water where the picture agrees, or bubbles); no outlines there |
| `endings[]` | `{j, t, number, reason}`: a number ended by a purity guard (e.g. `blocked view 4.5 s`, `appearance drift 0.12`, `area jump`, `swallowed another track`) |
| `links[]` | every lone-link candidate, made or not: `made`, `why` (`linked`, `not lone`, `gap too long`, `checks contradict`), `number_a`, `number_b`, `object_a`, `object_b`, `t_end` (s, end of a), `gap_s`, `breath` (gap crosses a lens-out), `limit_s`, `others` (raw object ids of other whale-sized tracks in view), `others_numbers` (their whale numbers, same order; null for a track without a whale number), `lone`, `detector_before` / `_during` / `_after` (frames with an unexplained detector box), `size_ratio`, `appearance_distance`, `checks_ok` |
| `refused_starts[]` | `{j, t, reason, box}`: detector boxes the tracker did not start a number from (`start on bubbles / surface / air`, `box classifier: fish`, `box classifier: other`) |
| `purity`, `method`, `start_source`, `blocked_rule`, `own_back_template` | settings of the run |

A number: `number` (int; `"sN"` for small), `kind` (`whale` / `small`), `objects` (raw tracker object ids joined in
it), `first_j`, `last_j`, `first_t`, `last_t`, `n_frames` (frames with an outline), `frames[]` = `{j, box, polygons}`
(`box` = `[x0, y0, x1, y1]`, fractions; only frames where the number is outlined).

## Clips

`clips/.../<t0>-<t1>.mp4`: no overlay, H.264 (`avc1`), yuv420p, 1280 x 720, constant frame rate = the track's `fps`
(30000/1001 or 30), `+faststart`, exactly `n_frames` frames: **frame j of the file (0-based, in decode order) is
track frame j**, i.e. source frame `frames.index[j]` at `frames.t[j]`. In a browser, frame j is shown from
`currentTime ≈ j / fps` (use `requestVideoFrameCallback` and its `mediaTime` for exact frame matching: `j =
round(mediaTime * fps)`). Frames were decoded exactly as the tracker decoded them and scaled to 1280 x 720
(`INTER_AREA`). Poster `.jpg`: the frame with the largest outlines, 1280 x 720.

Not in the bundle on purpose: GPS positions, photo file names or keys, sensor tables beyond `windows.csv`, raw masks
(npz / parquet), anything from the photo-link work.

## Media facts (measured with PyAV from the file headers, 2026-11-24)

Measured by `experiments/12_export/media_facts.py` (codec, codec tag, size, pixel format, average frame rate,
container from each file's headers). "Browser plays as is": H.264 8-bit 4:2:0 in an MP4 / MOV container plays in
every current browser; HEVC plays only in Safari and only with the `hvc1` tag (`hev1` files are refused even there
unless re-tagged); this is from the codec tags, not from trying each browser. The `._*` files next to the
2025-02-03 recordings are macOS resource forks (AppleDouble), not videos, and are left out. Frame rates of the
2025-02-03 recordings are not exactly 30000/1001 (29.95-29.98 average): the export clips are constant 30000/1001 and
`frames.t` in each track JSON gives the true source time of every frame.

Summary: the source recordings of 2025-02-03 (`.mov`, HEVC `hev1`, 1920 x 1080) and of 2025-01-02 (`.mov`, HEVC
`hvc1`, 3840 x 2160) do not play in Chrome / Firefox; the `.m4v` copies of 2025-02-03 (9 of them) and the dev clip
are H.264 and would. **Caution with the `.m4v` copies of 2025-02-03**: this project found dropped frames, time
drift of up to 1.5 s per file and washed-out levels in them (PLAN.md, "Design that follows"). Every time, frame index
and outline in this bundle refers to the `.mov` (`index.json` `file`); a page that plays the `.m4v` of a whole
recording must not expect frame-exact agreement with `windows.csv` or the tracks. For frame-exact drawing use the
bundle's clips. The scan's cut clips under `underwater_cache/clips/` are mostly HEVC `hev1` (155 of 158) and
do not play in a browser. The review videos and this bundle's clips are H.264 yuv420p and play everywhere.

#### source recordings: 28 files

| ext | codec | tag | size | pixel format | fps | container | files | browser plays as is | where |
|---|---|---|---|---|---|---|---|---|---|
| .mov | hevc | hev1 | 1920x1080 | yuvj420p | 29.973 | mov | 9 | no (HEVC hev1: even Safari wants hvc1; needs a re-tag or re-encode) | 2025-2-3 Mini-mother 11-96 |
| .mov | hevc | hvc1 | 3840x2160 | yuvj420p | 30.0 | mov | 9 | Safari only (HEVC hvc1) | TestSubset 2025-1-2 CATS -85 |
| .m4v | h264 | avc1 | 1920x1080 | yuvj420p | 29.975 | mov | 4 | yes (H.264 8-bit 4:2:0) | 2025-2-3 Mini-mother 11-96 |
| .m4v | h264 | avc1 | 1920x1080 | yuvj420p | 29.95 | mov | 2 | yes (H.264 8-bit 4:2:0) | 2025-2-3 Mini-mother 11-96 |
| .m4v | h264 | avc1 | 1920x1080 | yuvj420p | 29.963 | mov | 2 | yes (H.264 8-bit 4:2:0) | 2025-2-3 Mini-mother 11-96 |
| .m4v | h264 | avc1 | 1920x1080 | yuvj420p | 29.974 | mov | 1 | yes (H.264 8-bit 4:2:0) | 2025-12-20 CG.m4v |
| .m4v | h264 | avc1 | 1920x1080 | yuvj420p | 29.951 | mov | 1 | yes (H.264 8-bit 4:2:0) | 2025-2-3 Mini-mother 11-96 |

#### cut clips (underwater_cache/clips): 158 files

| ext | codec | tag | size | pixel format | fps | container | files | browser plays as is | where |
|---|---|---|---|---|---|---|---|---|---|
| .mp4 | hevc | hev1 | 1920x1080 | yuvj420p | 29.973 | mov | 124 | no (HEVC hev1: even Safari wants hvc1; needs a re-tag or re-encode) |  |
| .mp4 | hevc | hev1 | 1920x1080 | yuvj420p | 29.974 | mov | 14 | no (HEVC hev1: even Safari wants hvc1; needs a re-tag or re-encode) |  |
| .mp4 | hevc | hev1 | 1920x1080 | yuvj420p | 29.975 | mov | 6 | no (HEVC hev1: even Safari wants hvc1; needs a re-tag or re-encode) |  |
| .mp4 | hevc | hev1 | 1920x1080 | yuvj420p | 29.971 | mov | 5 | no (HEVC hev1: even Safari wants hvc1; needs a re-tag or re-encode) |  |
| .mp4 | hevc | hev1 | 1920x1080 | yuvj420p | 29.972 | mov | 3 | no (HEVC hev1: even Safari wants hvc1; needs a re-tag or re-encode) |  |
| .mp4 | h264 | avc1 | 1920x1080 | yuvj420p | 29.974 | mov | 3 | yes (H.264 8-bit 4:2:0) |  |
| .mp4 | hevc | hev1 | 1920x1080 | yuvj420p | 29.976 | mov | 2 | no (HEVC hev1: even Safari wants hvc1; needs a re-tag or re-encode) |  |
| .mp4 | hevc | hev1 | 1920x1080 | yuvj420p | 29.978 | mov | 1 | no (HEVC hev1: even Safari wants hvc1; needs a re-tag or re-encode) |  |

#### review videos (review_sheets/underwater): 40 files

| ext | codec | tag | size | pixel format | fps | container | files | browser plays as is | where |
|---|---|---|---|---|---|---|---|---|---|
| .mp4 | h264 | avc1 | 1280x720 | yuv420p | 29.97 | mov | 22 | yes (H.264 8-bit 4:2:0) |  |
| .mp4 | h264 | avc1 | 1280x720 | yuv420p | 30.0 | mov | 6 | yes (H.264 8-bit 4:2:0) |  |
| .mp4 | h264 | avc1 | 1280x720 | yuv420p | 29.974 | mov | 5 | yes (H.264 8-bit 4:2:0) |  |
| .mp4 | h264 | avc1 | 1920x540 | yuv420p | 29.97 | mov | 4 | yes (H.264 8-bit 4:2:0) |  |
| .mp4 | h264 | avc1 | 1920x540 | yuv420p | 29.974 | mov | 2 | yes (H.264 8-bit 4:2:0) |  |
| .mp4 | h264 | avc1 | 1280x720 | yuvj420p | 29.971 | mov | 1 | yes (H.264 8-bit 4:2:0) |  |

#### export clips (export/web_v1): 152 files

| ext | codec | tag | size | pixel format | fps | container | files | browser plays as is | where |
|---|---|---|---|---|---|---|---|---|---|
| .mp4 | h264 | avc1 | 1280x720 | yuv420p | 29.97 | mov | 134 | yes (H.264 8-bit 4:2:0) |  |
| .mp4 | h264 | avc1 | 1280x720 | yuv420p | 30.0 | mov | 18 | yes (H.264 8-bit 4:2:0) |  |
