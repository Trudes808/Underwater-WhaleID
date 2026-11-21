# CC-11-96-20250203-101236-000-00004: notes (AI by-eye labels, test video)

Sample: 120 frames, seed 20250207 (offset 46.8 frames, spacing 444.5 frames). Windows: drone drop (0-181.4 s, which
is also what random window 1 with wseed 1004 gave: 1.41-181.41 s) and random window 2, 208.4-388.4 s (no events).
Labelled by me (WP3); fish re-checked by a helper agent under the user's stricter fish rule (2026-11-20).

## What this recording shows that the dev clip does not

- **The drone drop** (events.json `drone_drop`): f0-f2722 the tag hangs under the drone (drone frame, landing gear
  and release fill the top third, fisheye view down to the sea and horizon; `other` = drone). Release f2723
  (90.92 s), fall f2723-f2746 (sky and horizon), hit f2747 (91.72 s, splash sheet and jolt), then the view is fixed
  to the mother's back at the waterline with whitewater and her blow in view; carried under at f2796 (93.35 s);
  view settled about f2830 (94.5 s). Hit, not miss: after the hit a dark body stays fixed at the bottom edge and the
  camera is carried down in one smooth dive; the back is then fixed at the bottom of every frame.
- **Camera view**: after the drop the camera looks up and forward through open blue water; the back is usually only
  a thin dark curved band at the bottom edge (much less of it than on the dev clip). Long empty stretches:
  f21829-f31609 and f32943-f52947 are almost all empty blue.
- **The other whale** (probably the calf) comes very near: head with tubercles (f5381, f6715, f7159), silhouette with
  both flippers (f8049), throat grooves and a flipper over the lens (f15606), head-down touching the back's horizon
  (f19607, f20051), head rising above the back (f45835).
- **Fish** (strict rule, boxes in fish_calls.json): jack-like fish with forked tails in 13 sampled frames, 1-8 per
  frame (f5381 7, f8049 8, f8938 5 larger 100-250 px around two whales, f15606 one jack swimming against the
  whale's throat). 22 further frames have tail-less spindles or specks, kept as `unsure` in `other`.
- **In air after the drop**: f14272 (476.5 s) shows the back in air with spray and a tag part blurred on the lens.
- **Seafloor**: none in the whole recording.
- Things checked and not fish: the round object at f9827 is a distant whale; dark streaks at f12938 are wave shading.
  f12938 also shows a smooth dark tapered shape hanging from the surface (whale part or boat hull: uncertain).
