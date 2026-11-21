# CC-11-96-20250203-114256-000-00007: notes (AI by-eye labels, test video)

Recording: 53,346 decoded frames, 29.963 frames/s, about 29.7 min. 120 sampled frames labelled (`frames.csv`), two 180 s
windows looked at on thumbnail grids (`events.json`). Labels by Claude, 2026-11-20: AI by-eye labels, not ground truth.

## What the recording shows

- Starts under water: the first sample (f103, 3.4 s) already shows the steady view up and forward through blue water
  with the back at the bottom. No drone drop, no deck scene, no tag in someone's hands in this file.
- The tagged whale's back is a dark curved band with a low hump and a few pale scars at the bottom edge in every
  underwater sample except f10328 (entry froth). Near the surface the back shows texture and light dapples (f9883,
  f40113, f41446, f42336, f44114, f50782-f51671). A small pale mark at the lower right edge (about x 1820, y 1040)
  stays fixed from f12106 to f33445: a scar on the back or a tag part, not a fish.
- Long empty stretches: f10773-f17885 (6 min) and f19219-f37446 with only three whale sightings (f27665, f33889,
  f36556). The mother seems to rest at depth while a second whale (most likely the calf) comes and goes.
- Other whales: 36 of 120 samples show one or two (34 with one, 2 with two); 4 more have a faint one. Many are sharp silhouettes from
  below against the sun (f4104 belly with both flippers, f7216, f8105, f46781, f47226, f47670, f49004 fluke), several
  are very near and cut by the frame edge (f6772 head, f38335 fluke + flipper, f41002 / f41446 lone pectoral fin,
  f45447, f45892, f46337, f53005), several are heads rising just above the back's horizon (f33889, f37890, f43225;
  f38779 stays fixed relative to the back and is only faint).
- Six surfacings (breaths) in window 2, 1332-1483 s, each 2.4-3.1 s in air; one more around 341-345 s (sample f10328
  is its entry froth). In air the frame shows sky, sea, distant land on the horizon, clouds, the wet brown back at
  the bottom, blow spray / whitewater in front, water sheets washing over the lens, and a yellow tag part at the frame
  edge for a few frames at most exits (f39918, f40389-f40393, f41528-f41531, f42427-f42430, f43929-f43932,
  f44358-f44360). No boat, people, drone or other animals seen in the air frames looked at.
- Just under the surface before each exit the centre shows the brown total reflection of the back below a bright
  sky window (e.g. f39886-f39903). With a rippled window edge this is underwater; once a smooth circular waterline
  rim crosses the dome with the brown band at the centre it is the half-out phase, labelled crossing (f39904-f39915
  and the other exits in `events.json`). A detector that treats brown = air would fire early.
- Entry froth: tan / white bubble froth for about 0.5-1 s after each entry (f39987-f40005, f41602-f41630), then bubble
  streaks and white specks for a few seconds (f44114 at 3.5 s after an entry).
- No seafloor in the whole recording. Looked for on purpose on contrast-stretched lower halves of 16 samples spread
  over the file and on every sheet: the only textured surface low in the frame is the back (fixed, curved, moves with
  the camera, scars). All 120 frames `none`.
- No dolphins, sharks, turtles, divers, boats or seabed objects seen.

## Fish candidates (all checked at full size)

Revised 2026-11-20 under the user's fish rule (a fish call needs a visible body AND tail, a school visibly turning
together, or a remora on a whale; specks and spindles under about 40 px are never fish). None of the candidates
shows a tail at full size, so **no frame is labelled fish**; all are `fish` none with `unsure: ...` in `other`.
`fish_calls.json` is therefore an empty list.

| frame | time (s) | count | size (full-res px) | where / what | label (first pass -> now) |
|---|---|---|---|---|---|
| 4549 | 151.8 | ~15-20 | 10-17 | dark spindles upper centre-right near the sun glow; over f4519-f4579 they drift as one group but no visible turning, no tails | school -> none + unsure |
| 4993 | 166.6 | ~15-20 | 10-17 | dark spindles upper right, same kind of group | school -> none + unsure |
| 5438 | 181.5 | ~10 | 10-15 | dark spindles top centre, loose group | few -> none + unsure |
| 6327 | 211.2 | 1-2 | 5-10 | dark specks top right (f6297-f6357) | none + unsure |
| 49004 | 1635.5 | 6-8 | 5-10 | dark specks next to the sun disc, absent at f48999 and f49009 | none + unsure |

Not fish: pale rising specks f10773 (small bubbles), white streaks f44114 (bubbles / particles after an entry),
the fixed pale mark lower right (f12106-f33445), surface dapples f42780 / f51671. No remoras seen.

## Changes after the 2026-11-20 boundary-case rules

- Fish: f4549, f4993 (school) and f5438 (few) -> none, with `unsure` notes in `other` (see the table above).
- Whale at the back's horizon: checked every sighting that touches the back's horizon on frames +-15 / 30 / 45
  around it (f18774, f27665, f33889, f36556, f37890, f38779; f43225 and f43669 on the window frames). All move
  relative to the back except the head at f38779, which stays almost fixed over f38734-f38824: other_whales 1 -> 0,
  faint_whales 0 -> 1.
- The fixed pale mark at the lower right (f12106-f31666) is now named in `other` as part of the back.
- Events: the half-out phase before each exit (smooth circular waterline rim across the dome, brown band at the
  centre) is now crossing; crossing_first moved earlier by 6-13 frames on all six exits, event frames unchanged.
- Entry froth rule: no sampled frame falls 5-10 frames after an entry (f10328 is about 18 frames after; froth under
  half the frame), so no view label changed.

## Mask candidates

- f4549: about 15-20 dark spindles near the sun glow (a fish-or-not test case: label them unsure, not fish).
- f36556: near whale, clear outline, body diagonal from the bottom edge with a raised knobbly pectoral fin.
- f38335: fluke upper left and long pectoral fin upper right of a whale passing just above, both cut by the edge
  (one or two instances? good test of instance splitting).
- f43225: near whale at the surface cut by the top right edge plus a hazy whale low on the back's horizon.
- f43669: whale with a long pale pectoral fin reaching down to the back's horizon, plus a dark whale part hanging
  from the top (other whale touching the back in the image; whale vs back boundary).
- f53005: very near scarred whale along the right edge with dorsal and pectoral fin (large, cut, close).

## Uncertain labels (of 120 frames)

- Whale counts: f6772 (left-edge shape, faint), f38335 (one whale or two), f43669 (two or one), f41891 and f42780
  (faint pale shapes), f53005 (whose body; counted as another whale).
- f41002 and f41446: lone pectoral fins counted as one whale each; owner out of frame.
- f38779: faint (head at the back's horizon, nearly fixed relative to the back); could be another whale holding
  station.
- Fish: the spindle groups at f4549, f4993 and f5438 are probably real small fish (they move as a group) but show
  no tails at the available resolution; labelled none + unsure.
- Exits: the start of the half-out (crossing) phase is uncertain by about 3 frames.
- f4549, f5438, f46781, f47226: the bottom edge is dark and the back band barely visible; labelled `back` by
  continuity.
- f10328: view `clear` with entry froth over about a quarter to a third of the frame (borderline bubbles).
- Events: entries at f42518 (about 3 frames) and f44006 (about 2 frames) are the least certain; the out-of-window
  surfacing near f10234 / f10310 was not stepped with the full rule.

## Things that may need a class rule

- Lone pectoral fins very near the lens (f41002, f41446) and parts of one whale on two sides of the frame (f38335,
  f43669) make the per-frame whale count depend on a guess about ownership. A rule such as "count connected
  whale parts; parts separated by open water count once if they plausibly belong to one animal" would help.
- Under the 2026-11-20 fish rule this recording has no fish at all, although three samples show groups of 10-20
  spindles that drift together; if detectors are later scored on "small schooling things", an `unsure` value in
  the `fish` column itself (instead of free text in `other`) would make them countable.
