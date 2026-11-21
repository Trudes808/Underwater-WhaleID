# How the test-recording labels are made (WP3 part 2)

Brief used for every 2025-02-03 recording (CC-11-96 tag on the mother "Mini-mother 11-96"). The labels are AI by-eye
labels for scoring only; nothing is fitted on these videos. Definitions follow `README.md` (dev clip); this file
adds what the tag recordings need.

## Inputs (already extracted; do not re-decode unless needed)

`/home/sat3737/NOAA_whales/underwater_cache/reference/<stem>/`
- `sample/NNNNNN.jpg`: 120 full-size frames, evenly spaced with a seeded random offset (`index.json`: seed, spacing,
  decoder times). File name = frame index in decode order.
- `win_SSSS/NNNNNN.jpg`: every 15th frame (2 per second, 960 px) of two 180 s windows (`index.json`: windows, wseed).

Tools (`PY=/home/sat3737/NOAA_whales/Underwater-WhaleID/.venv/bin/python`, `T=experiments/02_reference_labels/frametool.py`):
`$PY $T sheet DIR --out x.jpg f1 ... f6` (6 frames, 960 px each), `$PY $T crop DIR f x0 y0 x1 y1 [grid] --out x.jpg`
(full-size crop, upscaled, with a coordinate grid), `$PY $T strip VIDEO a b --out x.jpg --tw 480 --cols 4` (every frame
a..b, decoded from the start: slow late in a 30 min file, about 1-2 min), `$PY $T grid DIR --out prefix --tw 128 --cols 15`
(all window frames as thumbnails, 180 per image). Scratch images go to the session scratchpad, never into the repo.

## Frame labels (`labels/<stem>/frames.csv`, one row per sampled frame)

Write one line per frame `frame|state|view|other_whales|faint_whales|back|seafloor|fish|other|note` to a scratch text
file, then `$PY experiments/02_reference_labels/build_frames_csv.py LINES index.json labels/<stem>/frames.csv` (it
checks value sets and that all 120 frames are present).

- `state`: air / crossing / underwater / unusable. Air = the above-water scene at the frame centre (also on the drone,
  on deck, in someone's hands: user decision, deck tests are `air`). Crossing = lens in foam or under a water sheet so
  neither side is recognisable at the centre. Unusable = black, covered, corrupt.
- `view`: clear / bubbles (about half the frame or more hidden by bubbles) / dark (too dark to see a whale-sized shape
  in most of the frame) / glare (half or more washed out by sun). Borderline -> pick one, write "borderline" in note.
- `other_whales`: whales recognisable as a whale or whale part in the frame itself, hazy or cut by the edge, NOT the
  tagged whale. A lone fluke / flipper counts as one whale unless it belongs to a whale already counted.
  `faint_whales`: shapes that could be a whale but you cannot confirm.
- `back` (the tagged whale's own body, user decision): none / back / other part named (e.g. `flipper`). In these
  recordings the back is often only a dark curved band at the very bottom edge, fixed in the frame; label it `back`
  and say "faint band" in the note when that is all there is. Never count it as another whale.
- `seafloor`: none / visible / unsure (a dark band in the haze that may be the bottom). LOOK FOR IT ON PURPOSE (a mother
  with a calf often rests over sand or reef). Tell it from the back: the back is fixed in the frame and curved, moves
  with the camera; the seafloor has perspective, sand ripples / reef texture / rocks, fades into haze with distance.
  If none is seen in the whole recording, say so plainly.
- `fish`: none / few (up to about 10) / school (more than about 10). **Correction from the user 2026-11-20:** the
  15-20 px dark shapes on the dev clip that I called fish are NOT fish. A fish call needs shape you can actually see
  at full size: a body AND a tail (fins, eye, forked tail), or a school visibly turning together, or a remora on a
  whale. Specks, dashes and spindles without a visible tail (in practice anything under about 40 px long) are never
  `fish`: label `fish` none and write `unsure: N dark specks / spindles, <where>, <size> px` in `other`. Check at full
  size (crop) and follow a candidate over neighbouring frames. For EVERY frame labelled few / school, add an entry to
  `labels/<stem>/fish_calls.json`: `[{"frame": N, "boxes": [[x0, y0, x1, y1], ...], "count": n, "note": "..."}]`
  (full-resolution pixel boxes around the fish you count, one box per fish or one per tight group); the fish
  candidate sheets for the user are made from it.
- `other`: anything else alive or man-made: drone, tag parts, boat hull, divers / snorkellers, dolphin, shark, turtle,
  seabed objects. `note`: what you saw, and why a label is uncertain.

Boundary cases agreed after the first four recordings:
- Just under the surface, the sky seen through Snell's window with the back's brown reflection around it is still
  `underwater`. The half-out phase of an exit (circular waterline rim across the dome, brownish band at the centre)
  is `crossing`. In air, waves washing over the lens with sky / back visible through them stay `air`.
- A whale part seen exactly at the back's horizon counts as another whale only if it moves relative to the back in
  neighbouring frames; otherwise `faint_whales` with a note (it may be the tagged whale's own fin or fluke).
- Things fixed on the back (barnacle patch, remora, tag part) are part of `back`; name them in `other`.
- Entry froth (beige streaks for 5-10 frames after an entry) is `bubbles`, never seafloor.

## Events in the two windows (`labels/<stem>/events.json`)

Look at every window frame on the thumbnail grids. For every change between underwater and air, step through every
frame around it with `strip` (full frames, 480 px wide is enough) and apply the rule: event frame = first frame whose
frame centre shows the other side of the surface (exit: sky / horizon / the back in air, even through droplets;
entry: the water column, including froth). `crossing_first` / `crossing_last` = frames in between. Write the schema of
`labels/CC-11-96-20250203-101236-000-00004/events.json` (windows with start/end/seed/how, events list with type, frame,
time_s, crossing_first, crossing_last, note). A window without events is written with `"events": "none: ..."`.

## Also

- `labels/<stem>/notes.md`: what this recording shows that the dev clip does not (other animals, seabed, people, boat,
  tag in air, long empty stretches), with frame references; every fish sighting.
- Mask candidates: list 4-6 frames that would make good pixel-mask frames (other whales, fish, anything a detector
  could confuse with a whale, seafloor, seafloor vs back, whale lying near the bottom, sand vs reef, bottom fading into
  haze) with one line each on why.

## Second deployment, 2025-01-02 tag CC-09-85 (WP3b part B, added 2026-11-20)

Same brief as above, with these changes.

- Input is the tag's `.mov` (the `.m4v` conversions drop frames and drift). Frames were read with `sample_mov.py`
  (PyAV through `uwid.video`): frame index = position among the shown frames, as `uwid scan` counts them. Step
  through frames with `$PY sample_mov.py strip VIDEO a b --out x.jpg --tw 480 --cols 4` (it seeks, so it is fast);
  `frametool.py strip` must NOT be used on the `.mov` (different decoder). Seeds: sample `20250102 + piece number`,
  windows `2000 + piece number`; `index.json` has them.
- Lines have 15 fields: the ten above, then `whale_size|whale_frame_share|whale_cut|quality|quality_note` on frames
  with `other_whales` >= 1 (empty fields otherwise). Rule: `README.md`, "Size and quality": size = longest visible
  extent of the largest other whale / frame width in 0.1 steps (a vertical extent in frame-width units: full height
  = 0.56); `hq` = size >= 0.5, sharp, surface detail visible, lit (not a dark silhouette against the light), a whole
  part in view, little bubble cover; `ok` = a clear whale >= 0.3 failing an `hq` condition (silhouette, haze, cut,
  bubbles); `low` = <= 0.2, or hazy / only a sliver / mostly in bubbles. Write `close call` or `near hq` in the note
  when unsure. Look with a grid (`frametool.py sheet DIR --grid 192 ...` draws lines every 0.1 of the width).
- Boat time (tag on deck, in hands, on the drone): `state` = `air`, `back` = none. In `other` describe what is in view
  in plain words (`deck`, `people` (count only), `hands`, `sky`, `water from above`, `drone`, `boat rail`, `tag
  case`). NEVER write a person's name, face description, clothing logo, boat name or registration, or anything else
  that identifies a person. Lens covered / black / pocket: `unusable`.
- SEAFLOOR is the class nobody has labelled yet. `visible` = bottom you can recognise (sand ripples, reef, rocks,
  coral heads, the bottom's perspective fading away); `unsure` = a dark or pale band in the haze low in the frame that
  could be the bottom; `none` otherwise. Always write the kind in `other`: `seafloor: sand`, `seafloor: reef/rock`,
  `seafloor: sand+reef`, `seafloor: fading into haze`. Tell it from the back: the back is fixed in the frame, curved,
  brown-grey with scars, and moves with the camera; the bottom has perspective (texture gets finer with distance),
  changes between frames as the whale swims, and fades into blue. Entry froth (beige streaks for a few frames after
  an entry) is `bubbles`, never seafloor. Note frames where the bottom and the back touch, where another whale is near
  or on the bottom, sand against reef, and the bottom fading into haze: these are the mask candidates.
- Fish: unchanged (body AND tail visible at full size; specks are `unsure` in `other`). Every frame labelled
  few / school gets a `fish_calls.json` entry with boxes.
