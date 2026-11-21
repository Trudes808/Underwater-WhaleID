# CC-11-96-20250203-121302-000-00008: notes (AI by-eye labels, test video)

Recording: 53,393 decoded frames (29.97 frames/s, about 29.7 min), tag on the mother "Mini-mother 11-96", already on
the whale at the start (no drone drop in this file). Labelled: 120 sampled frames (`frames.csv`), two 180 s windows
(267.8-447.8 s, 1301.9-1481.9 s) scanned at 2 frames/s, every frame stepped through around the two surfacings in
window 2 (`events.json`). Labels by Claude, 2026-11-20; nothing fitted on this video.

## What the recording shows

- **Camera view**: looks up / forward through open blue water; the tagged whale's back is a curved band at the bottom
  edge in almost every frame (often only a faint dark band, f7540-f33347; clearer near the surface). The camera is
  tilted in long stretches (sun glow in the top-right corner, dark lower-left corner).
- **Long empty stretches**: about 70 % of the sampled frames show only blue water (e.g. f5760-f13769, f15104-f26673,
  f28452-f33347, f47585-f50699). Window 1 is empty blue throughout.
- **Revision 2026-11-20 (stricter fish rule and boundary cases from LABELLING_GUIDE_PART2.md)**: f28007 now counts
  5 fish with body and tail (boxes in `fish_calls.json`), the fainter ones and all specks moved to `other` as
  `unsure: ...`; f33792 (bump at the back horizon, no movement relative to the back) -> faint 1; f42690 / f43135
  (lower-left shape fixed relative to the camera) -> faint instead of counted; things fixed on the back named in
  `other` as part of `back`; exit crossings now start at the half-out phase (f39538, f40048).
- **Other whales** (22 of 120 sampled frames with a confirmed whale, 7 more with a faint candidate). Most likely one
  animal, the calf, coming and going and often very near: fluke over the back horizon (f300-f480), rising close to the
  back (f840-f990), seen from below with flukes and flippers (f1756, f2201, f27118, f27562, f44025), head with
  tubercles very near (f39131, f41801, f53369). Window 2 f43524-f44169: a whale swims a full circle above the camera.
- **A whale rising over the back horizon** is a recurring hard case: a pale knobby head tip shows just beyond the
  back's outline (f9243-f9303, f14190-f14790) and could be mistaken for part of the back; the strip f14100-f14790
  shows it is another whale's head that then rises past the camera (f14760).
- **Lower-left whale in window 2** (f42234-f43089, sample f42246 / f42690 / f43135): a hazy dark tapered head with
  white tubercle spots and a pale knobby flipper stays at the lower left for about 30 s, nearly fixed relative to the
  camera. Labelled faint (not counted): it may be the calf beside / below the mother or the tagged whale's own
  flipper.
- **Pale knobby flipper** at the left just after the second entry (f40164-f40187) and a pale green shape at the left
  at f40021: owner unknown (calf or the tagged whale's own pectoral flipper).
- **Surfacings**: air in sampled frames f4425 (about 148 s), f39576 (inside window 2) and f47140 (about 1573 s).
  Window 2 holds two short surfacings (exit f39558 / entry f39638, exit f40077 / entry f40158, each about 2.5 s in
  air). In both exits the lens rises through a thin layer of water flowing over the back, so the exit frame is a
  judgement call of about 2-3 frames; waves then wash over the lens in air (f39565-f39571, f40078-f40092) without a
  water-column view (not counted as entry / exit). A splash rises ahead at f40116-f40124 (blow or another animal).
- **Tag in air**: a yellow-green tag part at the left edge in air frames (f4425, f39563-f39566, f40079-f40084).
- **Camera artefacts**: pink / red horizontal streaks over the image just under and at the surface
  (f39497-f39555, f40045-f40076); not an object.
- **Seafloor: none.** Looked for on purpose in every sampled frame and both windows: no sand, reef, rocks or bottom
  fading into haze anywhere; the only textured surface low in the frame is the back (fixed in the frame, curved).
- **No boat, people, divers, dolphins, sharks or turtles seen.**

## Fish (every sighting)

- **f28007 (934.4 s)**: loose group of jack-like fish, right half of the frame. 5 show body, dorsal fin and forked
  tail at full size (60-100 px; boxes in `fish_calls.json`); up to 5 fainter spindles (35-70 px, no tail seen) lower
  in the group are listed as unsure in `other`. Followed f27960-f28060 (every 4th-20th frame): they swim across as a
  group. Labelled `few`.
- Not fish, checked: dark dashes just under the surface f5305-f5320 (wave facets that change shape frame to frame),
  pale flecks under the surface f37320-f37383 (light facets), small dark object with a pale spot on the back edge
  f28850-f29342 (fixed in the frame, so attached: barnacle, remora or tag part, unsure), dark bump on the back
  horizon f33700-f33890 (fixed for 6 s: part of the tagged whale), white blob on the back f36016 (bubble / mark).
- Unresolved specks, labelled `none` with `unsure: ...` in `other`: dark 5-10 px dots in empty water (f10210, f18664, f21778, f22223,
  f25783, f26228, f36461): particles or very distant fish; a pale 25-30 px spindle fleck drifting slowly at
  f38680-f38698 (small fish far off or a foam fleck).

## Mask candidates

- f14659: another whale's head with tubercles rising over the back horizon, right next to the back (whale vs back).
- f28007: the fish group (9-10 fish, 80-130 px), empty water otherwise.
- f2201: clear whale silhouette from below with fluke and both flippers, back band at the bottom.
- f42690: hazy whale with pale flipper at the lower left in dark blue water (hard, owner uncertain).
- f39131: very near whale head at the right edge just under the surface with surface texture above.
- f33792 or f28897: things fixed on the back (bump / small dark object) that a detector could take for a whale or
  fish.

## Uncertain labels

- Whale counts: faint candidates f33792, f40021, f42246, f42690, f43135, f45360, f52034 (any could be a real whale).
- f28007: 5 fish counted; the true number in the group is probably 9-10.
- f866, f17774, f34237, f34682, f51589: hazy whales, counted.
- f37796: bubble froth over about 40 % of the frame, labelled `clear` (borderline).
- f47140: the brown at the bottom read as the back in air (unsure).
- Exit frames f39558 and f40077: about 2-3 frames.
- Specks and the pale fleck listed above: possibly distant fish.
