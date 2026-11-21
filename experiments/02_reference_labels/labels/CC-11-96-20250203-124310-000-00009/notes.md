# CC-11-96-20250203-124310-000-00009: notes (AI by-eye labels, WP3 part 2)

Test video, scoring only. 120 sampled frames labelled from full-size frames (sheets of 6 at 960 px, crops at full
size, neighbouring frames decoded where a candidate needed following). Both 180 s windows looked at as thumbnail grids,
every frame stepped through around each of the 8 surface crossings. Labels by Claude, 2026-11-20; not ground truth.

## What this recording shows

- The tag sits on the mother's back the whole recording; no detachment, no drone, no tag in air on its own. The view
  looks up / forward through blue water, the sun or its glow near the top centre. The back is a dark (early) or pale
  (after about 1150 s) curved band at the bottom edge, often barely visible; the dorsal hump with a row of knobs shows
  at f16212, f26885, f27329 (and f26869-f26905). Labelled `back` on every frame except the froth frame f32221.
- **Surfacings**: two breaths in window 1 (exit f4038 / entry f4125, exit f4322 / entry f4409, 135-147 s), two in
  window 2 (exit f19354 / entry f19434, exit f20200 / entry f20291, 646-677 s), one outside the windows (exit about
  f32124, entry f32221, 1072-1075 s; found because sample f32221 is entry froth). Between the two breaths of a pair the
  camera stays just under the surface: sky through the surface at the top, a brown reflection of the back at the centre.
  In air: sky, clouds, sea surface, a blow / whitewater in front of the lens (f4116-4122, f32205-32219), the wet back at
  the bottom, droplets and water sheets on the lens.
- **Another whale (most likely the calf, sometimes very near)** in 33 of 120 frames: silhouettes with flukes against the
  sun glow (f203-f1092, f6873-f7762, f16212), very near bodies cut by the frame edge (f5539, f8652, f9097, f10431 head
  with tubercles, f15767, f25995, f30442), a lone flipper tip (f21103), hazy whales (f9986, f18880, f29553, f34000).
  One clearly far small whale (about 130 px, fluke and flippers) at f26885. Faint shapes counted `faint_whales` on 6
  frames (f1981, f16656, f17101, f17546, f27329, f37113).
- **Seafloor: none seen in the whole recording.** Looked for it on purpose on every sample frame and at the bottom band
  with contrast-stretched crops (f5094, f13544, f24661, f37557, f45562): the band is curved, fixed in the frame and has
  no perspective, sand ripples or reef texture: it is the back. The beige streaked texture at f32221 (and f4125-4132,
  f4409-4413, f19434-19439, f20291-20295) is entry froth, not sand.
- Long empty stretches: 11320-13988 (377-467 s), 20659-25550 (689-852 s; f24661-f25550 with strong sensor grain, low
  light) and almost everything after f34889 (1164-1772 s: empty blue, sun disc at the same place, tiny particles; the
  camera barely moves, the mother is probably resting just below the surface).
- No boat, people, drone, divers, dolphins, sharks or turtles seen. A low dark bump on the horizon in the air frames
  around f32150-f32175 might be land or a boat far away (seen only at 320 px, unsure).

## Fish sightings (every one; strict rule of 2026-11-20: body AND tail / fins visible in the frame itself)

Counted (`fish` few, boxes in `fish_calls.json`):

| frame | time (s) | count | size (px, full frame) | what |
|---|---|---|---|---|
| f26885 (f26869-f26905) | 897 | 1 | ~45 body (~100 with fins) | jack-like; body and fins visible, tail motion-blurred in the frame itself (borderline), clear forked tail in f26869 |
| f40226 | 1342 | 2 | ~280 and ~140 | very near fish, mackerel / tuna-like (eye, open mouth, forked tail), and a smaller one with eye and forked tail |

Not counted, `unsure:` note in `other` (no tail visible in the sampled frame):
- f10431: several dark spindles 100-150 px just under the surface (f10420-f10440); could be wave shading.
- f26440: two fish-like shapes 35-55 px below the whale; jacks with forked tails show in f26453, but not in f26440 itself.
- f38891: one dark spindle ~60 px.
- f3760 dashes under the surface (wave shading), f10875 pale streaks in one frame only, specks / blobs of 8-15 px at
  f7318, f17991, f34444, f42894, f49564.
- The dark triangle above the whale at f33110 is its dorsal fin (moves with it f33097-f33125). No remoras seen.

## Mask candidates

- f26885: far small whale (~130 px) next to a jack-like fish (~100 px) of similar size, plus the back with the dorsal
  hump and knobs at the bottom: whale vs fish at small scale, back vs other whale.
- f40226: a very near fish ~270 px long, the largest non-whale object in the recording: a detector could call it a whale.
- f26440: near whale (body, flipper, fluke) with two fish below it.
- f10431: near whale head cut by the edge right under the surface, with the surface pattern and the unsure spindle
  shapes (fish vs wave shading).
- f33110: clear whale with dorsal fin, flippers and fluke in open water (the dorsal fin is not a fish).
- f32221: frame full of entry froth (beige streaks): looks like sand; should be bubbles, not seafloor.

## Uncertain labels

- f26885 fish `few`: tail blurred in the frame itself (fins visible); the only borderline fish call left.
- Faint whales on f1981, f16656, f17101, f17546 (barely visible), f27329, f37113.
- f32221: froth just after the entry, labelled underwater / bubbles; could be called crossing.
- Back on the late frames (after f34889) and on f13988, f23772, f24216: a faint pale band only.
- Exits (revised to the agreed boundary cases): Snell's window = underwater, half-out phase (rim / brown band at the
  centre) = crossing, then sky through a draining sheet = air. Exits moved 6-11 frames earlier (f4045->4038,
  f4330->4322, f19360->19354, f20211->20200); where the half-out phase ends is a judgement of about 3 frames.
  Entries are within about 2 frames.

## Possible changes to the class definitions

- Covered by the boundary cases agreed 2026-11-20 (Snell's window, half-out, froth = bubbles).
- A whole-body fish of 45-60 px with its tail motion-blurred (f26885, f38891) is the remaining grey zone of the fish
  rule; following it over neighbouring frames settles it, the single frame does not.
