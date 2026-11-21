# CC-11-96-20250203-104242-000-00005: notes (AI by-eye labels, test video)

Recording 2 of the 2025-02-03 "Mini-mother 11-96" tag (follows 00004, before 00006), 53,352 frames, 29.950 fps,
about 1781 s. Labelled by Claude on 2026-11-20 from 120 sampled frames (sheets of 6 at 960 px, full-size crops wherever
there were specks or unclear shapes) and two 180 s windows at 2 frames per second (131.7-311.7 s, 1490.2-1670.2 s).
Every frame was stepped through around the six surfacings in window 2, and neighbouring frames (every 10th or
20th) were decoded around eight sampled frames to follow whales at the back's horizon. These are AI by-eye labels,
not ground truth.

## What this recording shows

- **Tag on the mother the whole time; no drone, boat, people, divers, seafloor or animals other than whales and
  fish.** The camera looks up and out through blue water with the sun glow at the top; the tagged whale's back is a
  band at the bottom of every underwater frame (faint in deep frames, textured with light ripples just under the
  surface: f13531, f49099).
- **Long empty stretches.** 93 of 120 sampled frames have no other whale, and two stretches are empty for minutes:
  f3305-f9085 (110-303 s, which includes all of window 1) and f22867-f31759 (764-1060 s). Window 1 shows open water
  for its full 180 s except one fish (below).
- **The other whale (almost certainly the calf)** is in 27 sampled frames, often rising from or lying along the
  back's horizon: rising vertically (f193, f9529, f17532, f45542), lying side-on along the horizon (f9974, f17977),
  fluke rising beyond the back and sinking away (f32204), seen from below against the sun (f2860, f19755, f38428,
  f40651, f46431), very near (f10419 fluke across half the frame, f11308 side view with dorsal fin, f13975, f14420,
  f47320, f49988, f50433). For f193, f9529 and f32204 I checked neighbouring frames: the shape moves relative to the
  back, so it is another whale, not the mother's own body.
- **Six surfacings in window 2** (1562.9-1646.0 s, about every 20-30 s, each 2-3.5 s in air), plus a surfacing near
  392-408 s seen in the sample (f11752, f12197 in air; not stepped through). Exits pass through the half-out phase
  (sharp circular waterline rim, brownish band at the centre) for 20-33 frames; the rim forms gradually, so the
  crossing start is a judgement of about +-5 frames. In surfacing 3, whitewater washed over the lens for 11 frames
  (f47537-f47547) right after the exit; kept as air (see events.json).
- **The calf in window 2**: a small shape on the back's horizon from about 1505 s grows into the calf's head rising
  (1510-1528 s), then the calf swims up past the camera in front of the sun, fluke up (1540-1560 s), just before
  surfacing 1. It is near the camera again under the surface after entry 2 (1576-1580 s, right edge) and at the end
  of the window (1669-1683 s, f49988-f50440, close with pale scars, a silvery fish swimming near it).
- **Sun disc** in many frames (f9974, f10419, f29981-f41096, f45987, f46431): always well under half the frame,
  `view` stays `clear`. Only f46876 is `bubbles` (entry froth).
- **Lens / dome spot**: a small dark spot fixed at about (240, 110-160) from f42430 to at least f45097; in window 2
  near the surface it shows as an orange oval at the same place (f47031-f47511). Not an animal.
- **No seafloor anywhere.** I looked for it on purpose in all 120 sampled frames and both window grids. The only
  textured surface low in the frame is the mother's back (fixed in the frame, curved, moves with the camera); no sand
  ripples, reef, rocks or perspective. All 120 frames: `seafloor = none`.
- Things checked and not labelled as animals: pale streaks / specks (rising bubbles or particles: f1527, f13086,
  f13975, f14420), a dark blurred mass in air at f12197 (most likely a water film on the lens; counted `faint`),
  water drops and film streaks on the lens in air (f11752, f12197, f48210, f48654).

## Fish (every sighting)

Jack-like spindle-shaped fish, single or in loose groups near the top of the frame; no school, no remoras seen.

| frame | count | size (px, full frame) | where / note |
|---|---|---|---|
| f17088 | 4 | 60-100 | near the top; body and tail visible on 3, one cut by the top edge |
| f19311 | 4 | 55-80 | upper right (two overlapping); 2-3 fainter 30-50 px spindles not counted; borderline for the 55 px ones |
| f20200 | 1 | ~200 | right, hazy; pale lateral stripe and eye, tail faint: borderline |
| f33093 | 1 | ~50 | upper left, deep body with a tail notch: borderline (near the 40 px limit) |
| window 1, f6658-f6763 (222-226 s) | 1 | ~50-60 at 960 px (~110 full) | one fish swimming across the top; not a sampled frame |
| window 2, f50200-f50280 (outside the window, neighbour frames) | 1-2 | | silvery fish near the calf at the end of window 2 |

`unsure` (specks / spindles without a visible tail, in `other`): f637, f16198, f18866, f19311 (extra), f19755 (3),
f20200 (2), f21534 (5 spindles 25-45 px, lower right), f49988 (1 pale 30 px spindle). Boxes for the four `few` frames
are in `fish_calls.json`.

## Mask candidates

- **f10419**: very near fluke spread across the right half, tail stock down to the bottom, sun disc with sparkle:
  a very large whale instance against glare.
- **f18421**: calf's head at the back's horizon plus a long dark flipper standing up at the lower left (counted
  faint; could be the mother's own pectoral fin): other whale vs own body at the horizon.
- **f17088**: four jack-like fish 60-100 px near the top in open water: clean fish instances, fish vs speck.
- **f11308**: just under the surface, very near calf with dorsal fin filling the left half, surface underside at the
  top, textured back at the bottom: whale vs surface vs back.
- **f14420**: very near whale along the right edge plus a faint translucent head-like shape on the back's horizon:
  edge-cut instance and a faint far shape.
- **f46876**: entry froth covering most of the frame (bubbles vs whale vs back, first frame after an entry).

## Uncertain labels (120 sampled frames)

- 8 frames with `faint_whales = 1`: f12197 (probably water on the lens), f14420, f18421 (flipper: calf or mother's
  own), f21978, f41985, f43319, f45097, f50877.
- Fish: f19311 count (55 px spindles), f20200 and f33093 borderline single fish; all four are in fish_calls.json
  for checking on the fish sheet.
- `back` in the four air frames (f11752, f12197, f48210, f48654) is the dark band at the bottom edge, probably the
  back, not clearly identifiable.
- f46876 is one frame after the entry event: `underwater` / `bubbles`, borderline `crossing`.
- Events: event frames +-1 frame (+-2 for entries 2 and 6, where foam turns into froth); crossing start of exits
  about +-5 frames. Surfacing 3 has 11 frames of whitewater over the lens just after the exit (kept as air).
