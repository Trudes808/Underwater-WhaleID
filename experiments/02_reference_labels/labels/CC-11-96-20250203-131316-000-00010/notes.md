# CC-11-96-20250203-131316-000-00010: notes (AI by-eye labels, WP3 part 2)

Test video, scoring only. Recording 7 of the 2025-02-03 "Mini-mother 11-96" tag, 53,395 frames, 29.975 fps, about
1781 s. Labelled by Claude on 2026-11-20 from the 120 sampled frames (sheets of 6 at 960 px, full-size crops of every
speck or unclear shape, frames 1 s before / after decoded for 13 candidates) and the two 180 s windows at 2 frames per
second (152.1-332.1 s, 858.0-1038.0 s), with every frame stepped through around the four surface transitions in
window 1. These are AI by-eye labels, not ground truth.

## What this recording shows

- **The tag stays on the mother the whole recording.** No detachment, no floating tag, no boat, deck, people,
  drone, divers, other animals besides whales and one fish. Underwater the camera looks up through blue water with
  the sun disc near the top; the back is a pale curved band at the bottom edge, usually faint (`back` on 119 of 120
  frames; `none` only on the froth frame f10459).
- **Long empty stretches**: most of the recording is empty blue water (107 of 120 sampled frames have no other
  whale). Sun disc in almost every frame, never half the frame (`view` stays `clear`).
- **The calf**, mostly far and silhouetted, in 13 sampled frames (+7 faint). A pattern specific to this recording: a
  dark shape appears on the back's horizon, stands there for several seconds almost without moving, grows, and then
  becomes the whole calf swimming up past the mother (window 1, 279-302 s: f8370-f9045; window 2, 931-972 s:
  f27908-f29140; sample f35376 -> f35821 -> f36266 -> f36711 -> f37156, 1180-1240 s). Applying the horizon rule
  (count only if it moves in neighbouring frames), I counted f1560, f8679, f28257, f36711 and f40271 (they move or
  grow into the calf within seconds) and left f35376, f35821, f36266 and f45610 as `faint` (fixed within +-1 s).
  Whole or near calf: f9124, f9569 (head with tubercles cut by the left edge, just under the surface), f23807 (head
  down, flippers raised), f28702, f31372, f37156, f37601, f46500.
- **Surfacings**: two in window 1 (exit f9382 / entry f9452, 313.0-315.3 s, about 2.3 s in air; exit f9780 / entry
  f9867, 326.3-329.2 s, about 2.9 s in air), between them the camera stays just under the surface (Snell's window,
  brown reflection of the back). A third just before 349 s (f10459 is entry froth) and one near 1611 s (f48280 is in
  air; f46945-f48725, 1566-1626 s, sit just under the surface); neither stepped through. Window 2: no surfacing.
- **In air** (f48280, window-1 surfacings): sky, sun, tilted sea horizon with whitewater, drops and water sheets on the
  lens, wet dark back at the bottom. **Yellow-green objects at the left / right frame edges** in air (f48280,
  f9386-f9390, f9790): most likely the tag's own float or housing; not boat or people.
- **Seafloor: none seen in the whole recording.** I looked for it on purpose on every sampled frame and window grid,
  and contrast-stretched the lower halves of f2004, f12239, f13128, f22918, f42496, f49615, f50505, f53175: the only
  surface low in the frame is the pale curved band, fixed in the frame, without perspective, ripples, rocks or reef:
  the back. The entry froth at f10459 is bubbles, not sand. All 120 frames `seafloor = none`.
- Artefacts: pink horizontal streak (f48725) and pink lens flare over the back (f46945); colour shifts to pale blue
  around 1640-1685 s (f49170-f50505).

## Fish (strict rule: body AND tail visible in the frame itself)

**No sampled frame has a fish**: all 120 frames `fish = none`, `fish_calls.json` is empty. A dark-blob finder (blobs
30 px or longer, darker than the blurred background) was run over all 120 frames and every hit was looked at; the
candidates below are `unsure` in `other`:

- f5119: 1 dark round blob ~25 px, no tail.
- f17578: 3 blurred dark blobs 30-50 px in a chain under the sun disc, moving together down-left over f17578-f17608;
  no tails (maybe fish, maybe far debris; not a school by the rule).
- f38936: 2 blurred blobs 20-40 px above the back's horizon. f41606: 1 round blob ~40 px. f42941: 1 spindle ~20 px.
- Many frames: specks under 15 px (particles), not noted on every row.

One clear fish outside the sample: **f9903 (330.4 s, window 1, seen while stepping through entry 2)**: a dark fish
~200 px long with its tail fin, passing through the entry froth at about x 1350-1570, y 370-560 (also f9902, smaller).
It is not a sampled frame, so it is not in `frames.csv` / `fish_calls.json`. No remoras seen.

## Mask candidates

- **f8679**: calf rising beyond the back's horizon with a raised tubercled flipper, silhouetted next to the low sun
  disc: back vs other whale at the horizon, a whale shape a detector could split or merge with the back.
- **f28257**: a whale head standing on the back's horizon that barely moves for seconds (the horizon case); window 2
  shows it is the calf.
- **f9569**: very near whale head with tubercles cut by the left edge, just under the surface, surface underside and
  brownish back band: large partial instance near the surface.
- **f45610**: long dark body lying just above the back's horizon (calf or the mother's own flipper): ambiguous
  instance, worth an `unsure` mask.
- **f48280**: in air with tilted sea horizon, whitewater, wet back and yellow-green tag parts: air / back / surface
  classes, nothing that is a whale.
- **f17578**: three small blurred blobs moving together under the sun: fish vs specks (`unsure`).

## Uncertain labels (120 sampled frames)

- Horizon shapes: f28257 counted as a whale (borderline: fixed within +-1 s, but becomes the calf within 6 s in
  window 2); f35376, f35821, f36266, f45610 left `faint` (probably the calf, possibly the mother's own body / flipper).
- Faint shapes cut by the edges: f1115 (top right, hazy tip), f29147 (beside the sun; the calf is there 1 s earlier),
  f47835 (skin with bubbles in the top-left corner).
- f8679: counted as the calf (window 1 shows it grow into the calf); with the camera rolled, a raised flipper could in
  principle be the mother's own; I think not.
- f48280 `back`: the dark band at the bottom in air is read as the wet back (unsure).
- f10459 `back = none` (froth).
- Events: event frames +-1-2 frames; the start of the exit crossing (Snell's window vs half-out rim) is a judgement
  of about +-5 frames. Water sheets over the lens in air (f9386-f9395, f9786-f9808) kept as `air`.
