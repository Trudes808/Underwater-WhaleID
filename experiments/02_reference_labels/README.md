# 02_reference_labels: AI by-eye reference labels (WP3)

Labels that detectors in this project are scored against. They were made by an AI agent looking at frames (Claude,
2026-11-20): they are **AI by-eye labels, not ground truth**. Where I could not tell, the label says so (`faint`,
`unsure`, a note). Spot checks by people: the sheets listed at the end.

## Files

| what | where |
|---|---|
| frame labels, dev clip (191 frames, every 15th decoded frame = 2 per second) | `labels/2025-12-20_CG/frames.csv` |
| surfacing events, dev clip | `labels/2025-12-20_CG/events.json` |
| mask prompts (rebuild the masks) | `labels/2025-12-20_CG/prompts.json` |
| test recordings 2025-02-03 (part 2) | `labels/<video stem>/frames.csv`, `events.json`, `prompts.json`, `notes.md` |
| masks (class PNG + instance PNG + meta JSON per frame) | `/home/sat3737/NOAA_whales/underwater_cache/reference/<video>/masks/` |
| frames the labels were made on (full-size JPEG) | `/home/sat3737/NOAA_whales/underwater_cache/reference/<video>/frames/` |
| check sheets for people | `/home/sat3737/NOAA_whales/review_sheets/underwater/reference_dev_*.jpg`, `reference_test_*.jpg` |
| counts | `tables.md` |

Code: `frametool.py` (decode in order, dump frames, viewing sheets, crops, frame strips), `masktool.py` (SAM 2.1
from my prompts, check overlays; adapted from `Dinov3-WhaleID/experiments/test_day/pixtruth_draw.py`), `sheets.py`
(check sheets).

## Frame index and time

Frame index = position in decode order from the first frame (OpenCV, reading in order; no seeking). The dev clip
decodes 2,857 frames (the container lists 3,020; the rest is hidden by an edit list). Frame times are uniform at
33.362 ms (29.974 frames/s from the container), so `time_s = frame / 29.9742`. `clock_est` in `frames.csv` is the
burned-in clock at frame 0 (11:29:36.6) plus `time_s`; it was not read off each frame.

## Definitions as applied

**state**
- `air`: the scene at the centre of the frame is above the water (sky, horizon, the back in air), even through
  droplets or a thin film on the lens.
- `crossing`: the lens is in the surface foam or covered by a sheet of water, so neither side is recognisable at the
  frame centre. Starts at the first frame where the centre stops showing the side the camera came from; ends at the
  last frame before the centre shows the other side.
- `underwater`: the centre shows the water column, including bubble froth under water.
- `unusable`: black, covered, corrupt. Not used on the dev clip.

Boundary cases met: f0-f20 have the lens just above the surface with splash and water films: labelled `air`
(the above-water scene is recognisable). f405 shows splash above the surface seen through the lens from below:
`underwater` (the lens is still under water). Frames of froth just after an entry (f30, f495) are `underwater` /
`bubbles`.

**view** (one value per frame, the strongest; underwater frames only use clear / bubbles here)
- `bubbles`: about half the frame or more is hidden by bubbles (bubble curtain, exhalation cloud, entry froth). At
  about half I chose `bubbles` and wrote "borderline" in the note (f75, f90, f1950, f2070, f2085). Bubble patches of
  a quarter to a third of the frame stay `clear` with a note (f570, f1560, f1905).
- `glare`: the sun or its reflection washing out half the frame or more. Never reached on the dev clip: the sun disc
  is in view in several frames (f1890, f2340, f2565) but covers far less than half.
- `dark`: not met on the dev clip.
- `clear`: everything else. Air frames are `clear` unless the lens is covered.

**other_whales** (not counting the tagged whale): whales I would call a whale without seeing the neighbouring
frames: a recognisable body, fluke, flipper or head, sharp or hazy, whole or cut by the frame edge. A lone
flipper or fluke counts as one whale when it cannot belong to a whale already counted.
**faint_whales**: dark or pale shapes in the haze or behind bubbles that could be a whale but that I could not
confirm in the frame itself.
Exception: the huge dark pectoral fin very near the lens at f2175-f2190 is not counted: it most likely belongs to the
near whale already counted under it (its base is at the right edge next to that whale), possibly to the tagged
whale itself.

**back** (renamed from `carrier`, user decision 2026-11-20): the tagged whale's own body. `back` = the textured
curved surface fixed at the bottom of the frame (skin with white scars, the low dorsal hump at the lower right,
from which exhalation bubble clouds rise at f345, f1200, f1755, f1905, f2610). I read it as the tagged whale's back
just behind the blowhole / near the dorsal hump; nothing on the dev clip contradicts that. `none` when it is not
identifiable (f0, f15, crossing frames, entry froth f495-f525, f2850). Never counted in `other_whales`.

**seafloor** (added on user request 2026-11-20; none / visible / unsure): **the dev clip shows no seafloor at all.**
The camera looks up and forward through open blue water; the only textured surface low in the frame is the back,
which is labelled `back`, not seafloor. All 191 frames: `none`.

**fish** (none / few / school): checked at full size at many moments (f240, f1485, f1500, f1800, f2340, f2445 and the
sheets). **The dev clip has no fish** (all 191 frames `none`). Correction 2026-11-20: I first labelled a group of 6-8
small dark shapes, 15-20 px long, at f1480-f1515 (upper left) as fish (`few` on f1485, f1500, f1515); the user looked
at the sheet and judged them not fish. They are now `fish` none with `unsure: small dark shapes upper left` in
`other`, and in the masks they are painted `unsure` (255). Rule since then: a fish call needs shape that can be seen
(body and tail, a school turning together, a remora on a whale); specks of 15-20 px are `unsure`, never `fish`.
Other things checked and found not to be fish: dark specks just
under the surface at f1785-f1830 (wave facets), silver blobs at f2445 (exhalation bubbles), a pale patch at f240
(foam). No remoras seen on any whale at the sizes available.

**other**: tag antenna (thin rod with a dark tip, upper left) in the air frames f450-f480; a white object on the
back near the bottom at f2820 (tag part?); land on the horizon at f2850; a thin pale arc low in the frame in many
underwater frames (f810, f870, f900, f2340 noted; probably the tag antenna seen through water or a reflection in the
housing; not labelled on every frame). No dolphin, shark, turtle, diver or boat.

## Events (`events.json`)

Rule: **the event frame is the first frame whose centre (960, 540) shows the scene on the other side of the
surface** (exit: sky / horizon / the back in air, even through droplets; entry: the water column, including froth).
`crossing_first` .. `crossing_last` are the frames in between where the centre shows neither side (foam or a sheet
of water). Every frame around each transition was stepped through (strips of single frames). The dev clip starts in
air and ends in air.

| event | frame | time (s) | crossing frames |
|---|---|---|---|
| entry | 24 | 0.801 | 21-23 |
| exit | 437 | 14.579 | 417-436 (13.9-14.5 s: foam, then a draining water sheet) |
| entry | 492 | 16.414 | 491 |
| exit | 2806 | 93.614 | 2796-2805 (water sheet, sun visible through it) |

Uncertainty: about 1 frame for the entries; for the exits the event frame is within about 1 frame but where the
crossing "starts" in foam is a judgement call of about 3 frames. For scoring at 0.5 s tolerance this does not matter.

## Masks (`prompts.json` -> cache `masks/`)

31 frames chosen to cover: near clear whales (f630, f720, f810, f900, f1545, f1665, f2175, f2190), several whales
(up to 6 instances: f900), far / hazy / faint whales (f1065, f1110, f1290, f2250, f2340, f2565), whales cut by the
frame edge (f570, f630, f1665, f1890, f2565), whales behind bubbles (f570, f585, f1485, f2070), the surface seen from
below with sun glare (f1890, f1950, f2565), small dark shapes that are not fish (f1485, f1500: painted `unsure`), and frames with no other whale (f60 entry
froth, f390 just under the surface, f450 and f2820 in air, f1995 and f2700 bubble curtain).

Per frame: `<frame>_class.png` (0 water, 1 whale, 2 back, 3 fish, 4 bubbles, 5 surface, 6 air, 7 seafloor,
255 unsure), `<frame>_inst.png` (instance id per whale / fish; fish ids would start at 11; the dev clip has no fish instances) and `<frame>_meta.json`
(instance -> class, grade, note). Whale grades: `clear` (outline sharp), `hazy` (outline visible but soft),
`faint` (shape I am not sure of). Bubbles and surface are coarse hand polygons; whales, back and fish from SAM 2.1
(box + points) except where SAM failed and I drew a polygon (f630 whale 4, f810 whale 2, f1950 whale 1) or thick
lines (the 13 small dark shapes on f1485 / f1500, 5-7 px, class `unsure` since the 2026-11-20 correction).

Every mask was checked on an overlay at 1280 px width with a coordinate grid, failures redone. Known weaknesses that
remain: SAM outlines of hazy whales sometimes carry speckle into the haze or bubbles around them (f720 whale 3,
f900 whale 5, f1290 whale 3, f2100 whale 3); far faint whales are rough blobs; the back is masked only where its
texture is visible (where bubbles cover it, the pixels are `bubbles`). Masks do not include every far whale counted
in `frames.csv` (I outlined those I could place).

## Uncertain labels on the dev clip (191 frames)

- 9 bubbles/clear borderline (about half the frame): f75, f90, f585-f615 region edges, f1950, f2070, f2085.
- Whale counts: `faint_whales` > 0 on 103 frames; on those the total is uncertain by that number. Counts on frames
  with 4-5 whales may be off by one where two hazy whales overlap.
- f240 centre: a pale streak near the surface (whale belly / flipper or foam) counted faint.
- f2055-f2070: a large dark region behind the bubble curtain may be a very near whale (counted faint, masked
  `unsure` on f2070).
- f2175-f2190: the owner of the huge pectoral fin (near whale, most likely; possibly the tagged whale).
- f1485-f1515 small dark shapes: first called fish, judged not fish by the user (2026-11-20); now `unsure`.
- Event crossing start in foam: about 3 frames.

## Rebuild

```
PY=/home/sat3737/NOAA_whales/Dinov3-WhaleID/.venv/bin/python   # or .venv/bin/python once .venv/READY exists
D=experiments/02_reference_labels; C=/home/sat3737/NOAA_whales/underwater_cache/reference; V="/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
$PY $D/frametool.py dump "$V" $C/2025-12-20_CG/frames --step 15
flock /var/tmp/sat3737-parts-cache/gpu_train.lock $PY $D/masktool.py run --frames $C/2025-12-20_CG/frames \
    --prompts $D/labels/2025-12-20_CG/prompts.json --out $C/2025-12-20_CG/masks --cache $C/samcache
$PY $D/sheets.py frames --labels $D/labels/2025-12-20_CG/frames.csv --frames $C/2025-12-20_CG/frames --out /home/sat3737/NOAA_whales/review_sheets/underwater/reference_dev_frames
$PY $D/sheets.py masks --masks $C/2025-12-20_CG/masks --frames $C/2025-12-20_CG/frames --out /home/sat3737/NOAA_whales/review_sheets/underwater/reference_dev_masks
```
SAM needs about 3 GB of GPU memory and takes a few seconds per frame; results are cached per prompt in
`$C/samcache`. `frames.csv` and `events.json` are hand-written (not generated).

## Part 2: test recordings of 2025-02-03 (tag CC-11-96 on "Mini-mother 11-96")

Test videos: labels for scoring only, nothing fitted on them. Seven 30-minute `.m4v` pieces that were completely
copied (size unchanged) when sampled: 00004 to 00010. Pieces 00011 onward were still being copied and are not
labelled. Per piece (`labels/<stem>/`): `frames.csv` (120 frames), `events.json` (two 180 s windows stepped through),
`notes.md` (what the piece shows, every fish call, uncertain cases), `fish_calls.json` (box per counted fish),
`prompts.json` where the piece has mask frames. How the labels were made, with the boundary rules agreed after the
first pieces and the user's fish rule: `LABELLING_GUIDE_PART2.md`. Counts: `tables.md`.

Who looked: I labelled 00004 (and the drone drop) myself; 00005-00010 were labelled by helper agents (Claude, same
model) working from the guide, each looking at all 120 frames at 960 px or more with full-size crops; all fish calls
were re-checked under the user's stricter rule (00004 by a separate helper). I read every helper's report but did
not re-label their frames; the check sheets are the way to spot-check them.

Sampling: 120 frames at `round(offset + k * N / 120)`, offset uniform in [0, N/120) from
`numpy.random.default_rng(seed)`, seeds 20250207 ... 20250213 for pieces 00004 ... 00010 (20250203 + piece number);
windows: two non-overlapping 180 s starts uniform in [0, duration - 180) from `default_rng(1000 + piece number)`.
Piece 00004's first window (1.4-181.4 s) already contained the drone drop, so it is the densely labelled drop window.
All seeds and frame times: `underwater_cache/reference/<stem>/index.json`.

Rebuild the frames: `PY sample_test.py VIDEO <cache>/<stem> --n 120 --seed S --windows auto --wseed W`; masks:
`PY masktool.py run --frames <cache>/<stem>/sample --prompts labels/<stem>/prompts.json --out <cache>/<stem>/masks
--cache <cache>/samcache` (run here on the CPU, `CUDA_VISIBLE_DEVICES=""`, because the GPU lock was busy); sheets:
`sheets.py frames|masks`, `fish_sheets.py`.

What the test recordings show that the dev clip does not (details and frame numbers in each `notes.md`):
- The drone drop (00004): tag under the drone f0-f2722, release f2723 (90.9 s), fall, hit on the whale f2747
  (91.7 s), carried under f2796 (93.4 s), settled about f2830. Hit, not miss: after the impact a dark body stays
  fixed at the bottom edge and the camera is carried down in one smooth dive.
- The camera looks up through open blue water; the back is usually only a thin band at the bottom edge. Long empty
  stretches (most frames have no other whale: 56-107 of 120 per piece).
- One other whale (almost certainly the calf), often very near, diving past the mother, rising over the back's
  horizon (the hard back-vs-other-whale case; rule: count it only if it moves relative to the back).
- Fish: jack-like fish with forked tails, singly and in loose groups; one school (00006 f13991, at least 11). Many
  more tail-less specks / spindles are `unsure` in `other`.
- Regular surfacings, each about 2.3-3.3 s in air, often in bouts of several breaths a few seconds to a minute apart.
- No seafloor in any of the 840 sampled frames or 14 windows (looked for on purpose); entry froth looks like sand
  and is the likely false positive. No boat, people, deck, divers, dolphins, sharks or turtles seen.

Masks: 14 frames over the pieces (drone parts in air as `gear`, whale silhouettes with fish, a very near whale with a
fish against it, a fluke against the sun, a school around the calf's flipper, a whale's head rising over the back,
fish alone in open water, a frame of tail-less specks (only `back`), entry froth that looks like sand, an air frame).
Fish masks are SAM inside the boxes of `fish_calls.json`.

## Size and quality of the other whale (WP3b part A, 2026-11-20)

Why: the user cares about whales that are large and well seen ("We really only care about whale when its large and
is high quality footage of whale in frame"), and wants a tag for the best footage (`hq_whale`: shallow, clear water,
good light, whale in a large part of the frame). Five columns were added to every `frames.csv` (existing columns
unchanged); they are filled on frames with `other_whales` >= 1 and empty otherwise. Graded by me (Claude) on all 366
such frames (dev 145, 2025-02-03 test pieces 221), on sheets of 6 frames at 640 px with tick marks every 0.1 of the
frame width and height, with the existing label note printed. AI by-eye labels, and **a first proposal: the user's
eye defines "high quality"**. Code: `quality.py` (`view` labelling sheets, `merge`, `review` sheets, `stats`).

| column | meaning |
|---|---|
| `whale_size` | longest visible extent of the LARGEST other whale (tip to tip in any direction, attached flippers and fluke included; a vertical extent is converted to frame-width units, the full frame height = 0.56) / frame width, 0.1 steps, 1.0 = fills the frame |
| `whale_frame_share` | rough share of the frame area it covers, 0.05 steps; `0.05` means "5 % or less" |
| `whale_cut` | `yes` when the frame edge cuts it |
| `quality` | `hq` / `ok` / `low`, rule below |
| `quality_note` | why, in a few words; `close call` and `near hq` mark the borderline grades |

Check of the size numbers against the outlines of the mask frames (31 frames with both): longest extent of the
largest whale mask vs my number: mean absolute difference 0.046, 26 of 31 within 0.1, no bias (-0.015). The area
share is rougher: my numbers run above the mask areas (0.05 written where masks say 0.01-0.07, 0.10 where they say
0.05-0.07), use it only as a coarse order.

**Rule as applied**
- `hq`: size >= 0.5 AND outline sharp AND surface detail visible (fluke trailing edge serrations or underside
  pattern, pigment, scars, tubercles, throat grooves, skin texture) AND lit (not a uniform dark silhouette against
  the surface or the sun) AND a recognisable whole part in view (fluke, head, most of the body: not only a sliver at
  the edge) AND not more than about a third hidden by bubbles.
- `ok`: clearly a whale, size >= 0.3 with a recognisable shape, failing one or more `hq` conditions: backlit
  silhouette, underside in deep shadow, haze, partly out of the frame, partly in bubbles, or simply 0.3-0.4 and
  mid-distance.
- `low`: size <= 0.2 at any sharpness (heads rising at the back's horizon, small sharp silhouettes, fluke tips), or
  any size when hazy / faint (size 0.3 in haze), only a sliver at the frame edge, or mostly hidden by bubbles.

Borderline cases met (all marked in `quality_note`):
- **Large backlit silhouettes** (size 0.5-0.9, outline sharp, fluke edge or knobbed flipper visible, but the
  underside dark against the surface light): graded `ok`, note `near hq` (dev f660-f690, 00004 f8049, 00005 f9974,
  f10419, f40651, 00006 f5092 (fills 0.9 of the frame), f12656, f14881, 00007 f49004, f50782, 00008 f44025, 00009
  f8652, f15767). This is the main thing to settle with the user: a tag camera looks up, so most near whales are
  seen from below against the light. 18 test and 9 dev frames are `ok` at size >= 0.5; if the user counts them as
  `hq`, `hq` roughly doubles.
- Near whales whose underside is in shadow but whose fluke edge / scars are sharp (dev f705-f900, 00007 f4104):
  graded `hq`, f705 and f4104 noted close call.
- Near whale beside the back, cut by the edge and partly in bubbles, skin detail visible (dev f2100-f2235): `ok`.
- Huge knobbed pectoral fin of uncertain owner on dev f2175 / f2190: measured with the fin (it most likely belongs to
  the near whale under it): f2190 `hq` (fin lit and sharp), f2175 `ok` (dark).
- 00004 f15161: very near knobbed flipper and body, possibly the tagged whale's own flipper (see its notes): `hq`,
  close call.
- 00007 f38335: counted as one whale whose body runs along the top edge (size 0.9); if it is two whales, 0.6.
- Clear but small heads with tubercles at the back's horizon (size 0.1-0.2) are `low` by size alone.
- Sizes on the 0.25 line were rounded by eye; about 10 frames could be 0.2 or 0.3 (and so `low` or `ok`), marked
  close call.

Sheets for the user (39 frames, 12-14 per grade with the close calls):
`/home/sat3737/NOAA_whales/review_sheets/underwater/reference_quality_examples_01..10.jpg` (sheets 01-03 `hq`,
04-07 `ok`, 07-10 `low`).
