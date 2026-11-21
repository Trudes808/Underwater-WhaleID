# CC-11-96-20250203-111250-000-00006: notes (AI by-eye labels, test video)

Recording 3 of the 2025-02-03 "Mini-mother 11-96" tag (follows 00004), 53,391 frames, 29.975 fps, about 1781 s.
Labelled by Claude on 2026-11-20 from 120 sampled frames (sheets of 6 at 960 px, full-size crops wherever there were
specks or unclear shapes) and two 180 s windows at 2 frames per second (539.5-719.5 s, 962.1-1142.1 s), with every
frame stepped through around the four surface transitions in window 1. These are AI by-eye labels, not ground truth.

## What this recording shows

- **Tag on the mother the whole time; no drone, boat, people, divers or other animals besides whales and fish.**
  Underwater the camera looks up and out through blue water; the tagged whale's back is a band at the bottom of every
  underwater frame, often only faint. A **knobbed hump** sits at the same place at bottom centre in many frames
  (f23779, f24224, f25114, f32678, f41576, f50475): fixed in the frame, so labelled `back`.
- **The other whale (almost certainly the calf) is in view in about half of the sampled frames** (64 / 120 counted,
  14 more faint): from below with flippers and throat (f1533, f5537, f5982, f10431, f12656, f48250), very near and
  filling most of the frame (f5092), head-on (f43801), whole body (f48695), fluke and tail stock from below near the
  surface (f13546, f16215, f26894, f51809, f53144), and diving down past the mother (window 2, 999-1044 s: fluke up,
  then the whole calf swimming away below the back's horizon). The calf swims close beside the camera during both
  surfacings in window 1 (f17356-f17416, f17515-f17581).
- **Whale parts at the back's horizon** (rule: count only if they move relative to the back). I decoded the frames
  1 s before and after f13101, f15326, f24224, f24669, f34457, f41576, f42021, f42911, f49585, f50030, f51365: in all
  but f24669 and f34457 the part moves (the calf diving past the mother: whole calf above the hump 1 s before f24224,
  fluke changes place in f41576 / f49585, f13101's dark triangle is the calf's fluke tip sinking), so they count as
  other whales; f30453, f31343, f33123, f34012 move in window 2.
- **Hazy rounded shapes sitting on the back's horizon at x 1000-1150** (f24669, f33568, f34457): they grow and shrink
  in place without moving across the back, so `faint`. In window 2 (1115-1128 s) the shape at that place grows and shrinks over a few seconds; it may be the
  calf far beyond the back, or part of the mother (fin / fluke edge). Unsure.
- **Two surfacings in window 1** (581.5-584.1 s and 627.1-630.0 s, each about 3 s in air) and two more air frames in
  the sample outside the windows (f37127 at 1238.6 s, f45581 at 1520.6 s; not stepped through). Exits pass through a
  phase where the dome is half out of the water: a sharp circular waterline rim with a brownish band (water film /
  grazing view of the surface) below the sky, lasting about 0.5 s (16-17 frames). On f45581 a dark back with a dorsal
  fin is at the left in air: it could be the mother with the camera rolled or the calf (counted faint).
- **Long stretches just under the surface** (surface underside across the top: f2423, f8207-f9097, f17105-f18440,
  f27784-f28228, f35792-f36682, f46025-f46470) and stretches of empty blue water (f2868-f3313, f28673-f29563,
  f37572-f40686 except f38907, f46915).
- **Sun disc** in many frames (f14881-f16215, f23779-f25559, f33123-f34457, f43356-f44246, f50475): always well under
  half the frame, so `view` stays `clear`.
- **Sensor artefacts**: red / magenta horizontal streaks across the frame just before both exits (f17386-f17417,
  f18792-f18796) and during air (f17485-f17497, f18827-f18829); not in sampled frames.
- **No seafloor anywhere.** I looked for it on purpose in every sampled frame and window grid, and cropped the darkest
  lower halves (f39352, f39797, f4647, f33568): no sand, reef, rocks or perspective texture. The only surface low in the
  frame is the mother's back (fixed, curved, moves with the camera). All 120 frames: `seafloor = none`.
- Other things checked and not labelled as animals: a pale brownish haze cloud just under the surface (f8652; also
  f17550 after the entry: froth residue or old exhalation, unsure), small rising bubbles / particles (f9542), pale
  ghost-like vertical blotches under the surface (f17550, f18440: bubble patches or light), a pale surface sparkle
  (f36682), water drops / a halo on the lens in air (f37127).

## Fish (re-checked 2026-11-20 under the stricter rule)

Rule (guide, user correction): a fish call needs a body AND a tail (fins / fork / eye) at full size, or a school
turning together. Every candidate of 33-320 px in the 55 frames first labelled few / school was proposed by a
dark-blob finder and judged on a full-resolution zoom tile; frames without candidates were cropped by hand.
Result: **25 frames keep fish (24 `few`, 1 `school`, 58 fish counted, boxes in `fish_calls.json`); 30 frames go to
`none`** with `unsure: N dark specks / spindles, where, size` in `other` (1088, 1978, 2423, 4203, 5092, 5537, 7317,
7762, 9987, 10876, 11766, 12211, 16660, 21110, 22444, 23779, 24224, 25559, 27339, 30008, 31343, 32233, 33568, 34902,
35347, 38462, 42021, 49585, 50030, 50920). Some of these are probably fish (blurred 60-120 px ovals near the calf:
f5092, f5537, f34902, f49585), but no tail is visible.

| frame | fish counted | size (px) | note |
|---|---|---|---|
| f198 | 1 | ~50 | upper left |
| f643 | 4 | 48-80 | 2 cut by the left edge |
| f1533 | 1 | ~50 | forked tail |
| f5982 | 3 | 90-200 | **pale silvery jack ~200 px against the calf's throat** + 2 |
| f11321 | 1 | ~55 | tail just visible |
| f13101 | 4 | 45-120 | jacks |
| **f13991** | **>=11 (school)** | 45-180 | jacks around the calf's flipper |
| f14436 | 5 | 55-120 | |
| f14881 | 1 | ~130 | forked tail |
| f15326 | 2 | 50-90 | |
| f19330 | 1 | ~90 | pale, eye visible, tail fin faint |
| f22000 | 2 | 70-130 | |
| f23334 | 1 | ~100 | |
| f24669 | 2 | 70-100 | |
| f25114 | 2 | 40-80 | |
| f26004 | 1 | ~75 | |
| f31788 | 1 | ~150 | forked tail |
| f32678 | 2 | 45-50 | |
| f33123 | 1 | ~110 | forked tail |
| f34012 | 1 | ~110 | just above the back |
| f41576 | 5 | 65-170 | around the fluke |
| f42911 | 2 | 70-110 | |
| f44691 | 2 | 60-100 | |
| f48695 | 1 | ~65 | |
| f51365 | 1 | ~270 | large jack, forked tail, eye |

All jack-like. No remoras seen.

## Mask candidates

- **f13991**: school of >=11 jacks with tails (45-180 px, more blurred) around the calf's long flipper and body: fish vs whale, many instances.
- **f5982**: calf's head and throat from below with a pale ~200 px jack right against it plus 2 fish with tails and blurred spindles: fish touching
  a whale.
- **f41576**: fluke rising just behind the mother's knobbed hump with 5 jacks: back vs other whale (the fluke moves, so it is the calf) at the horizon,
  the case a detector (and a person) could get wrong.
- **f33568**: hazy rounded shape on the back's horizon under a sun disc: faint far whale (or own body) vs back.
- **f5092**: a near whale filling most of the frame from below: very large instance cut by every edge.
- **f27339**: dark scene just under the surface with bubble froth over a third and pale tail-less silvery shapes: bubbles vs fish
  vs whale part.

## Uncertain labels (120 sampled frames)

- 14 frames with `faint_whales = 1`: f1978, f4647, f7762, f16660, f17105, f24669, f26004, f27339, f32233, f33568,
  f34457, f40241, f44691, f45581. The shapes on the back's horizon (f24669, f33568, f34457) may be the mother's own body.
- f13101 changed from faint to other whale after checking neighbours (calf's fluke tip).
- f35792: dark tapered whale part hanging from the top counted as a whale (which part is unclear).
- f27339: pale shapes not called fish (no tails); dark part at the left counted faint.
- Fish: f11321, f19330, f32678 (one of two), f41576 count tails that are only just visible; 30 frames with probable
  but tail-less fish are `none` + `unsure` in `other`.
- f18885 `back = none` (froth); f37127 and f45581 `back = none` (in air, the mother's body not identifiable).
- Events: event frames +-1 frame; where the exit "crossing" starts is a judgement of about +-5 frames (the rim / band
  phase); splash and water sheets over the lens in air (f17437-f17449, f18803-f18825) were kept as `air` because sky
  stays visible through them.

## Class definitions

Both points I raised (half-out dome phase = `crossing`; whale parts at the back's horizon count only if they move)
are now boundary cases in LABELLING_GUIDE_PART2.md and are applied here: the events already used the dome phase as
crossing (no sampled frame falls in it), and the horizon frames were re-checked against neighbouring frames.
