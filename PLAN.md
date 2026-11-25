# Underwater-WhaleID: plan

Written 2026-11-20. Phase 1 is running. Working rules: `CLAUDE.md`. Results: `experiments/LOG.md`.

## Purpose

Take a directory of underwater whale video, find the parts worth keeping and cut them into clips, so that footage
above water, tests, failed drops and stretches with no whale can be deleted. Later: tell individual whales apart in
the footage, build 3D models of them, and link them to the animals labelled in the surface photos.

## Phase 1 (now): detection

The user's priority, 2026-11-20: detection and semantic segmentation first, clustering later. Four questions:

1. Can we detect a whale reliably in frame underwater?
2. Can we detect fish as well, and tell them apart from whales?
3. Can we detect that the camera is underwater?
4. Can we detect when the camera surfaces?

And for 3 and 4: how much comes from the sensor data recorded with the video ("metadata").

Phase 1 is done when each question has a measured answer, a first `uwid clips` command turns a folder of video into
kept clips plus a list of what was dropped and why, and we have a list of what is missing.

## Findings so far (2026-11-20, details in `experiments/LOG.md`)

All picture-side truth is AI by-eye labels. One development clip (2025-12-20), one test deployment (2025-02-03,
pieces 00004-00010), one public deployment (Zenodo 4902325) for water state.

| question | answer | evidence |
|---|---|---|
| camera underwater? surfacing? with a sensor log | Yes. Depth alone, to about 0.3 s. | 80 surfacings on 2025-02-03, 18 checked by eye; burned-in clock = sensor time + 2.03 s, steady over 2.5 h |
| the same from the picture alone | Near perfect on the tag it was trained on; on an unseen deployment most windows right but events are missed and false events appear. | same tag: 53/54 events, 1 false; Zenodo: 1,112/1,182 windows, 30/40 events within 0.5 s, 58 false (about half probably real wave wash); 2025-12-20: 2/4 events, 4 false |
| whale in frame underwater? | Yes with a text-prompted detector, settings fixed on the other clip. | OWLv2 "whale", both frames of a 1 s window: 192/221 whale windows, 7/570 false on 2025-02-03; single frame 203/221, 14/570 |
| fish, apart from whales? | Fish are found; fish and whales are swapped both ways. Open. | 29/45 fish frames, 14/795 false; about 1 in 4 confident fish boxes sits on a whale; 14 of 33 whale boxes on test mask frames sit on a fish |
| the tagged whale's back | Found as its own class by a small trained classifier and by a fixed box plus SAM 2.1; rarely called a whale. | trained on 31 frames of the other clip: 99.4 % of back pixels found on 2025-02-03, but 8.7 % of open water also called back |
| seafloor | Not testable yet: no labelled frame shows any. Models with a sand / seafloor class put it on the back. | SAM 3 "seafloor" fires on 617 of 821 underwater frames with no seafloor |
| tag not on a whale (deck, drone flight, floating) | From the sensors: yes on this deployment, a third of the recorded video. | 7,931 of 23,605 s; 3 drone drops found, 2 failed; rule for failed drops rests on 3 drops |

Design that follows:

- Sensors first. With a log, depth and motion sort deck / flight / floating / surfaced / underwater; picture models
  run only on on-whale, underwater time. Without a log, the picture water-state model stands in.
- Every deployment with a log labels its own video for the water-state model at no cost.
- Input is the tag's `.mov`, never the `.m4v` conversion (dropped frames, time drift up to 1.5 s per file, washed-out
  levels).
- Whale scanner: OWLv2 at two frames per second (about 0.3 GPU hours per video hour). SAM 3 (the user's access works)
  outlines whales tighter but is weaker and slow as a scanner on the test deployment.
- A classifier trained on the 2025-12-20 clip alone does not carry over for water and air; 10 to 20 mask frames from
  each new deployment do (fitted-on-test result: water IoU 0.54 to 0.95, back 0.21 to 0.72).
- Per-window decisions: for presence, agreement across the frames of a window halves false alarms; for masks,
  averaging over a window blurs moving whales, so masks stay per frame.

Still missing: seafloor footage; more fish and a whale-against-fish test on a second deployment; pieces 00001-00003
and 00013-00014 (deck, failed drops, floating); a second and third labelled deployment before any threshold is
trusted.

### Added 2026-11-22: the scan on a second tag (WP9, details in the LOG entry of 2026-11-22 22:45)

| question | answer | evidence |
|---|---|---|
| does `uwid scan` run on a tag it was not built on? | Not at first (names, clock layout, time link, 4K sheets, the back boxed as a whale); it does now. | CC-09-85: clock read on 8,480 of 8,480 sampled frames; offset estimated per file, good to about 0.3 s per surfacing (0.1 s on CC-11-96); back filter found on this deployment, so not a clean test |
| what is kept on the second tag | 16.6 of 140.8 min; boat recordings dropped whole (49.5 min in air). | frames with another whale 27/31 (hq + ok 20/20, low 7/11); fish 7/7; empty frames kept 31/292; 100 dropped windows by eye: 0 whales, 2 fish |
| seafloor, sand, dark lens | The whale detector does not fire on them. | sand 0 whale boxes in 203 windows, dark 0 in 215; over the bottom 20 of 24 detections are real whales, 1 a jack |
| seafloor (labels) | Withdrawn 2026-11-23: the user found the labelled "seafloor" to be a whale's back and haze. Real seafloor is in 00017 around the blackouts, a rough surface at the side of the frame while the whale scrapes along it. | user review of `seafloor_01..05.jpg`; my look at 33 frames (`seafloor_candidates_00017_01..02.jpg`); WP10's seafloor numbers are void, redo is WP12 |
| seafloor (redo on user-confirmed frames, 2026-11-23) | Per pixel yes inside the one recording that has it; as a per-second label not yet; not testable on another day or bottom. | held-out episodes of 00017: 5,227,026 of 5,407,044 seafloor px found, IoU 0.89, back + seafloor as one class IoU 0.96; windows 59/59 on three episodes, 0/35 on the faint one; false calls 62-103 of 291 on 00023's back with haze; dev whale pixels called seafloor 2.9 % (old head 0.4 %). Scan unchanged |
| the hq tier | Wrong at breaths: the tagged whale's blow, the surface and the mirrored back are boxed as a whale. | hq windows under 1 m: 71 of 174 on CC-09-85, 53 of 723 on 2025-02-03; a flat depth rule would cut real shallow whales (5 of 10 graded `ok` frames under 1 m are tiered hq) |
| ok against low, silhouette flag | Weak rules. | box size: 36 of 128 low frames tiered ok or better; silhouette 13/34 found, 27/187 false |

Keep rule since 2026-11-22 (user): anything detected is kept, tiers hq / ok / low / small / fish are labels, each kept
stretch has a text description and its hq spans (`advanced_ok`). 2025-02-03 under this rule: 86.6 of 267 min kept
(66.7 before), clear whale frames 219/221.

### Added 2026-11-23: classifier veto and the 30 s clip guard in the scan (WP11b, LOG entry of 2026-11-23 03:13)

| question | answer | evidence |
|---|---|---|
| does the box classifier fix the hq tier at breaths? | Yes on these two tags. | hq windows under 1 m 53 -> 27 (2025-02-03) and 71 -> 41 (CC-09-85); blow boxes it was not fitted on, in hq / ok windows: 26 -> 1 and 43 -> 0; the remaining shallow hq windows by eye (orchestrator): 26 of 27 real near whales, and all 41 inside the two near-whale passes at 16:01 and 16:07 |
| what does the veto cost? | 59 and 71 windows no longer active; 3 of 60 looked at held a real whale (whale beside the floating tag in 00012, one fin at an edge); that footage is still kept, only the label is lost. | `scan_veto_dropped_01..05.jpg`; labelled whale frames kept 219 -> 220 of 221 and 27 -> 29 of 31 |
| large fish | The 24 near jacks are now tier fish / small (were ok / hq 12). Not a clean test: the classifier was fitted on them. | one school, one recording (00023) |
| the user's clip rule (30 s without activity, 2 s buffer) | Fewer, longer clips; more quiet water inside them. | 2025-02-03: 161 -> 100 clips, 86.5 -> 94.3 min, clips under 10 s 48 -> 19, 1,646 of 5,658 kept seconds without a detection; CC-09-85: 61 -> 27 clips, 16.6 -> 17.6 min, 609 quiet seconds |
| dark lens, sand | Labels in the scan, no keep rule, left out of hq spans. Sand is fitted in one recording and unchecked by the user. | dark 208 windows on CC-09-85 (8 of 11 labelled, 0 false), 0 on 2025-02-03; sand 242 and 3 |

The classifier was fitted on 1,054 labelled boxes that include both deployments' reference boxes, so the label
scores in this table are in-sample for the classifier. A clean test needs a third deployment.

### Added 2026-11-23: the tracking stage (WP11a, LOG entry of 2026-11-23 03:34 with the reviewer's note)

| question | answer | evidence |
|---|---|---|
| do the purity guards work online? | Yes: `uwid tracks "<deployment>"` runs them on the scan's hq spans. | 8 held-out stretches (4 per tag, each with ONE other whale): 0 numbers moved, 98-100 % of whale frames outlined except one breath span (226 of 293) |
| numbers on blow / surface / mirrored back | None started. | 2 breath spans on CC-09-85: 13 of 13 starts refused |
| lone-whale links (user's easy cases) | Work on the stretches the rule was set on (after the breath in 00007, 4.4 s; calf with fish: number 1 for 3 s, then one number to the end). On held-out footage only short breaks were linked. | old lone gap runs 12 links, 12 right; held-out 3 links, 3 right, breaks 0.1-1.0 s |
| a lone whale across a breath, held-out | Not yet. In the one whole encounter with a breath (00024, 75-164 s) the whale is one number up to the breath since the two fixes of 2026-11-23 (was two), and two new numbers after it (a fluke and a pale body, in view together, so the break is "not lone"). | fixes: a depth lens-out blocks a frame only where the scan's picture label agrees (blocked 117 -> 30 frames in 97-120 s); detector boxes on blocked frames no longer count as a second animal. Candidate 1 -> 3 passes size (1.39) and appearance (0.249). Whether it is the same whale is asked of the user; no rule changed on this one case. 00024 is no longer held-out for the blocked rule |
| a lone whale across a breath, held-out test with frozen rules (WP11c, 2026-11-23) | Partly. Only 3 breath encounters were left to test on (all 2025-02-03): 2 links made (1 right, 1 probably right), 2 refused (1 rightly: the whale did not come back; 1 a miss: the whale came back cut by the frame edge), no wrong link, no number moved. | gaps 3.9-7.4 s; both refusals on appearance (0.358, 0.364 against 0.30). Both tags together hold 11 breaths with a good whale before and whale activity after, so the most to gain is about 11 number changes |
| groups on held-out footage | One stretch with two whales (00004 286-306 s): no number moved between them; each whale gets a second number after a break of 0.5-0.7 s, because no link is made while two whales are in view. | 5 candidates, all refused as not lone; 10 starts refused (9 fish, 1 other); reviewer's sheet `demo_group_00004_286s_review.jpg`. One stretch is little |
| does the scan's whale count tell lone from group? | No: it often puts two whale boxes on one near whale. | 0 of 11 breath stretches have at most one whale box in every second; in 00005 604-629 s the scan sees up to 3 whale objects where the tracker sees only the tagged whale's own body (open: own head and flipper, or an animal close below; asked of the user) |
| cost | About 9 GPU s per video s, under 2.3 GB. | all hq spans after WP11b (675 s + 147 s): about 1.8 + 0.4 GPU h. Not run: needs the user's go-ahead |
| the tracker on all hq spans (WP13, 2026-11-24) | Ran on all 135 spans (786 s tracked) in 2.25 GPU h: 149 whale numbers, 79 lasting 3 s or more, the longest 26 s; 8 links, all across breaks under 0.6 s. In the reviewer's look at 24 sampled numbers no number moves between animals; one number of 0.9 s sits on the sea surface. | CC-09-85 17 spans: 19 -> 18 numbers, 11 >= 3 s; 2025-02-03 118 spans: 138 -> 131, 68 >= 3 s; starts refused: surface 48, fish 11, other 2; own-back refusal never used. Most hq spans are short (median 4 s) with one whale, so this run does not test many-whale scenes or breaks across a breath. AI by-eye review; the user's look at `wp13_a_sample_01..04.jpg` is pending |

Next (2026-11-23 20:45): running: WP13 the tracker on all hq spans of both deployments (rules frozen, about 2.2 GPU
h in 15-minute pieces; `src/uwid/` is frozen for other agents until it is done); WP14 own back against a calf close
in front of the tagged mother, and bubbles numbered after a breath (study in `experiments/`, no live change); WP15
set-up and smoke test for the 3D mesh trial (the trial itself starts when WP13 is done); WP16 time-overlap check
between the tag and the photos of 2025-02-03 and 2025-01-02. Still open with the user: more recordings of CC-09-85
with the bottom in view (only 9 are on disk; 00013-00016 would also hold the tagged whale's photographed surfacings
of 2025-01-02); on 2025-02-03 the tag shows on photos the catalogue files under A03 and under A01 (which is the
tagged whale?); the comp-group clip's burned-in clock is about two hours off the notes' time for the tagged group.
WP16 is done (LOG 21:25). Waiting behind WP14: the sand redo on the confirmed sediment frames.

Next (2026-11-24 01:50): WP13 is done (LOG 01:42, reviewed 01:47). Started: WP15 phase 2, the 3D trial with the stand-in model on WP13's numbers (up to 4 frames per number lasting 3 s or more, at most 240 frames); WP17, the own-back change of WP14 behind a flag that is off by default, replayed on the calf stretch and the 17 candidate stretches. (WP15 phase 2 finished 03:55 and WP17 04:44, see their rows; no job of ours is running.)

Next (2026-11-24 21:27), on the user's answers of this evening: WP18 makes the own-back switch the tracker's
default and fixes far specks taken into a whale's outline; WP19 packs the results for the agent who will build the
video-analysis page in the web app (export bundle + `HANDOFF_WEB.md`). The 3D work is paused. After WP19 this repo
has no running work; the open questions below stay open. Not started (they wait for the user): making that flag the default, any change to the surface refusal or the link rules, the bottom-contact sensor label, a priority change for jobs that hold the GPU lock.

## What is on disk (checked 2026-11-20 19:00)

| item | what it is |
|---|---|
| `underwater_video/2025-12-20 CG.m4v` | 1920x1080, 29.97 frames/s, H.264 at about 10 Mbit/s. 95.3 s are shown; the container lists 3,020 frames (100.8 s) but an edit list hides the end and frames past about 2,857 do not decode. Video track only: no audio, no sensor track. Exported by Apple software on 2025-12-25. 21 keyframes, up to 8 s apart. |
| its burned-in clock | bottom right, for example `20Dec25 11:29:36.600`, in 0.1 s steps. The clip runs from 11:29:36.6 to about 11:31:12. |
| its content (40 frames looked at) | camera at the surface at 0-2 s, 13-16 s and after about 93 s, underwater otherwise; up to five or six other whales in view; bubble curtains at 64-70 s and 87-93 s; no fish seen (not yet checked at full size). |
| `underwater_video/2025-2-3 Mini-mother 11-96/Data/` | sensor log of camera tag CC-11-96, 2025-02-03 08:36:52 to 15:21:10 local (UTC-10): `.csv` (3.1 GB, 400 rows per second: depth, two light sensors, temperatures, accelerometer, gyroscope, magnetometer, GPS, hydrophone, battery, and camera fields `Camera`, `Camera time`, `CC status`, `CC vid. size`), `.txt` (tag settings and sensor list, which names a sensor "SWS", probably a salt-water switch), `.ubx` (raw GPS), `.bin` (305 MB, raw log), `.cfg` (not text). |
| `underwater_video/2025-2-3 Mini-mother 11-96/Video/` | the tag's recordings in 30-minute pieces, each in two forms with the same name: `.mov` (from the tag: HEVC 1920x1080, mono sound at 32 kHz, written in fragments) and `.m4v` (a conversion: H.264 1920x1080 at about 29.95 frames/s, stereo sound, a keyframe every 4-5 s). Names carry the local start time and a sequence number: `CC-11-96-20250203-101236-000-00004`. Pieces 00004 to 00006 (10:12 to 11:42) were on disk at 19:15, the copy was still running. |
| surface photos | 2025-02-03 is one of the photo days in the review app. 2025-12-20 is not on disk. |

The footage is video from a camera tag on a whale. The textured surface fixed at the bottom of every frame is the
tagged whale's own back, "the camera surfaces" means the tagged whale comes up to breathe, and the whales to detect
are the other whales around it.

Decided by the user, 2026-11-20:
- The tagged whale's back is its own class, `back`. It is to be identified, and it does not count as another animal
  in the scene.
- The tag is dropped onto the whale from a drone. The first recording of the 2025-02-03 deployment on disk (piece
  00004, from 10:12:36) contains the drone drop. Failed drops look the same up to the miss.
- Deck tests need no class of their own: "not underwater" is enough.
- The seafloor is its own class when underwater (`seafloor`). The development clip may show none: open water.
- `2025-12-20 CG.m4v` is "a rare clip of a large comp group with many whales per frame", not typical footage. It
  has no fish: the small dark shapes at 49.4-50.5 s that the AI labels called fish are not fish (user, 2026-11-20).
  Typical footage has far fewer whales in view, so false alarms on whale-free windows matter more than this clip
  suggests.
- More folders are being copied (and one more is planned) with "much better fish examples as well as drone, deck,
  above water, and seafloor examples". Fish and seafloor are judged on those, not on the 2025-12-20 clip. When they
  are on disk: inventory, labels on a sample, and the development / test split is redrawn by deployment.
- What counts as a whale (2026-11-20, late): "We really only care about whale when its large and is high quality
  footage of whale in frame. When its small enough that it would be fish sized anyway we dont care as that footage is
  less useful." A small thing is called fish when in doubt ("we can err on calling them fish and use examples to
  tighten it up"). So the costly error is a LARGE whale missed; a far, fish-sized whale missed is acceptable.
- A quality tier, `hq_whale`: "likely shallow depth, with good water clarity and lighting and a whale in large portion
  of frame. Those are some of the most valuable clips and will be the only clips we care about for the later goals of
  3D or animal clustering." The user's eye defines it; candidates go to the user on sheets before a threshold is
  trusted.
- Sensors "should be used as a high quality asset when available".
- Two stages (user, 2026-11-20): "using the cheap model to help us slice up the footage to find the areas with
  activity and then only using sam3 to get high quality masks and high quality individual tracking through the clip".
  WP8 (`experiments/07_tracks_demo/`) makes the first example videos with outline overlays.
- Tracking across blocked views (user, 2026-11-21): "detect the bubble curtain or above water moments and cut those
  out in the trackers eyes to try and see if the tracker better survives a small discontinuity rather than a blocked
  field of view". WP8b (`experiments/07_tracks_demo/gap/`) tests it: a gap-tolerance curve on clean clips, then the
  bubble curtain and breaths three ways (straight through, blocked frames cut, memory frozen).
- The fish calls on sheets `reference_fish_candidates_01` and `_08` (2025-02-03) are all fish (user).
- Second deployment, `TestSubset 2025-1-2 CATS -85` (tag CC-09-85, copy started 23:16): early recordings above water
  on the boat, some middle ones with seafloor, some late ones with a few other whales.
- Quality grades (user, 2026-11-22, after looking at `reference_quality_examples_01..10.jpg`): "the high are best
  the ok are fine and the low are not useful." The labeller's rule for hq / ok / low (in
  `experiments/02_reference_labels/README.md`) is therefore the definition to score against.
- What is kept and what the tiers are for (user, 2026-11-22, correcting my reading that low footage could be
  dropped): "low footage is still good we want to keep all footage that has detected stuff in it / any activity. Low
  just shouldnt be used in future when we try to do 3d and animal ID. Also sand is fine to keep the footage, it just
  shouldnt be used for additional stuff. Really any activity region is worth keeping and labeling with a text
  description. But just the high quality parts can maybe be used for more advanced stuff." So: keep every stretch
  with anything detected (a whale of any size, fish, and seafloor or sand once detectors for them exist); the tier
  (hq / ok / low / small / fish only) is a label, not a filter; every kept clip gets a text description; only the hq
  parts are flagged for the later 3D and identification work. Dropped: tag in air, and water with nothing detected.
- Silhouettes (user, 2026-11-22): large whales seen from below as dark shapes against the light keep the grade the
  labeller gave them and get their own mark: "lets keep them as marked silhouettes, if its useful for things we can
  use them, if not no worries." Column `silhouette` in the labels, a flag per window and clip in the scan.
- Tracks (user, 2026-11-22): "new tracks are less of a problem then a track swapping to a new whale." Purity first:
  a track number must never move to another animal; the same whale under several numbers is acceptable and is
  joined later by appearance. WP8c (`experiments/07_tracks_demo/pure/`) changes the tracker's policy to end a track
  rather than risk a swap.
- Lone-whale links (user, 2026-11-23, after the `demo_pure` videos): "I do feel like we could figure out a way to
  maintain track on some of the easy ones such as the calf with fish ... maybe some sort of size/depth could allow it
  to stay cotninuous, and also the after breath seems feasible to stay linked as well. I understand new tracks in the
  big multi whale scenes as its ambiguous if they changed places during the gap, but in shots where there really was
  only 1 other whale in view it seems possible for us to keep that linkage reliably". So: purity guards stay for
  multi-whale scenes; where only one other whale is in view around a break (breath, blocked view, a fish passing),
  the fragments are linked into one displayed number. The calf counts as a whale and is tracked as one. Asked of the
  tracking agent as part of WP11a (link layer above the guards, each link recorded with its evidence).
- Clip boundaries (user, 2026-11-23): "I would expect real full clips to begin when a whale appears in scene and end
  when there is no longer activity detected. We can even have a guard to prevent over segmenting the clips. That
  guard can make sure that there is no activity for at least >30sec and then cut the clip with just 1-2sec on either
  side of buffer to save the space". So a clip ends only after 30 s without activity, buffer 2 s each side (was:
  2 s pad, merge under 5 s). From the cached windows: 2025-02-03 161 clips / 86.6 min -> about 93 / 99.9 min;
  CC-09-85 61 / 16.6 min -> about 35 / 23.1 min. Asked of the scan agent in WP11b. The tracker demo videos the
  user watched were fixed-length test stretches, not scan clips.
- Seafloor (user, 2026-11-23, on the first seafloor sheets): "Seafloor_01 is a whale back, so is 02, 03 is just hazy
  water ... 05 is also just a whale back. Seafloor can be found in a couple of the middle long videos from the Test
  subset day. Largely after/before moments where whale screen blacks out from scratching along seafloor and kicking
  up ab unch of sediment." The labeller's seafloor spans are withdrawn. Rule since then: the first sheet of a new
  class goes to the user before anything is fitted on it.
- Seafloor confirmed (user, 2026-11-23 midday, on `seafloor_candidates_best_01.jpg`): "yes all of those seafloor
  candidates now show both the back of the whale in forground, and the seafloor in background. If it helps its not
  an issue if they got a combined class where it was just back/seafloor. Best if separate but not as issue if they
  are combined to just background." So: separate `seafloor` and `back` if the model can tell them apart, otherwise
  one background class; neither counts as another animal. The sediment (sand) class has not been confirmed on a
  sheet yet.
- Answers of 2026-11-23 evening (user):
  - Tracker run: "Yes go ahead and run the tracker." (all hq spans of both deployments, WP13.)
  - Sediment: "Yes those are the sediment frames of dragging along seafloor." (`sand_candidates_best_01.jpg`, the
    22 sediment frames of 00017.) The sand label may now be redone on them.
  - 2025-02-03, 00005 at 604-629 s: "is a calf swimming clone in front of the mom who is tagged with the camera"
    (my reading: "close in front"). So the tagged whale of this deployment is a mother with her calf, the scan was
    right to call a whale there, and the tracker was wrong to refuse it as the tagged whale's own back. The calf is
    a whale and gets a number.
  - CC-09-85, 00024 after the breath (numbers 2 and 3 of the whole-encounter run): "the pale animal is just bubbles
    there is no animals there after the breath, and the back ends low in frame it is especially tall in that clip."
    So both numbers after the breath are false: bubbles get a whale number. The proposed rule "across a breath,
    link to the single candidate that passes the checks" is dropped: it would have linked whale 1 to bubbles
    (size 1.39, appearance 0.249, both passing).
  - The three held-out breath demos "looks fine".
  - New goals: "once the high quality tracker runs lets attempt a trial run of using those to create 3D meshs from
    them with sam3" (my reading: SAM 3D Objects, the single-image 3D model released with SAM 3; WP15), and a thought
    experiment on the comp group: "Given that footage and the assumption that he camera is on a whale, could we use
    any of the information from the tracks in the video to potentially help with any of the animal clustering of
    the photos of that day. It may or may not help in any way" (WP16: reasoning plus a time-overlap check on the two
    tag days that have photos; nothing is built).
- Answers of 2026-11-24 evening (user), after the own-back replay (WP17) and the 3D trial (WP15):
  - The switch: "The video with the switch on looks quite good. Yes you may make the switch default." So
    `own_back_template` becomes the default of the tracker (WP18).
  - Number 2 on the calf sheet (the tall dark wedge at the left edge, 00005 612-621 s): "The yellow 2 dark wedge is
    a calf but it is quite far in distance so its ok if thats hard to tell." So it is an animal, not the mother's
    own flipper, and giving it a number is right.
  - A fault to fix: "One very small issue is it looks like the last few seconds show something very small in
    distance getting picked up as part of the 1 track?" (the calf video's last frames: number 1's outline also takes
    in a far speck; WP18).
  - 3D: "We can put a puase in the 3d work for now, go ahead and add where we are at in the notes we may pick it
    up later." WP15 is parked; its state is under "Parked".
  - Hand-over: "Once that last thing is ready please pack things up to be passed off to another agent who will be
    working on the web app and will be creating a new page for the video analysis" (WP19: a data bundle the web
    app can read without this repo's environment, and `HANDOFF_WEB.md`).
- More blocked views (user, 2026-11-22): the second deployment has long stretches where stirred-up sand blocks the
  view; whether another whale is in the background there is not known. The labels mark them
  (`blocked_spans.csv`, kind `sand`).
- The unit of decision is a short series of frames, about a 1 s window, not a single frame: "we should not focus on
  exact per frame we should leverage multiple frames as short series to determine action inside those say 1sec
  windows". Detectors may use several frames of a window (and its neighbours), and results are scored per window.

## How each question is measured

Labels per sampled frame (definitions and hard cases: `experiments/02_reference_labels/README.md`):

- `state`: `air` (lens out of the water), `crossing` (waterline or a sheet of water across the lens), `underwater`,
  `unusable` (black, covered, corrupt).
- `view`: `clear`, `bubbles` (half the frame or more hidden by bubbles), `dark`, `glare`.
- `other_whales`: number clearly visible, plus number faint (a shape in the haze one cannot be sure of).
- `back`: whether the tagged whale's own body is in the frame (none / back / another part, named).
- `seafloor`: none, visible, or unsure (a dark band in the haze that may be the bottom).
- `fish`: none, few, school. `other`: anything else alive or man-made (dolphin, shark, turtle, diver, boat).
- Events: each `exit` (underwater to air) and `entry` (air to underwater), to 0.1 s.
- On a subset of frames, pixel masks: each other whale (graded clear / hazy / faint), the tagged whale's `back`,
  `seafloor`, fish, and coarse regions for bubbles and for the sea surface seen from below.

| question | measure | truth |
|---|---|---|
| 3 underwater? | per-frame accuracy and per-state recall, with `crossing` counted on its own | by-eye frame labels; depth sensor where a sensor log exists |
| 4 surfacing? | events found within 0.5 s: share found, false events, timing error | by-eye events; depth crossing the surface |
| 1 whale? | frames with a clearly visible whale that are found, at a stated false-alarm rate on whale-free frames; separately for near / far / in bubbles / cut by the frame edge; false alarms on the carrier's back, bubbles and surface glare; mask overlap on the pixel subset | by-eye frame labels and reference masks |
| 2 fish? | fish frames found and false alarms; how often a whale is called fish and a fish called whale | by-eye labels on our footage where fish occur; a public fish data set otherwise |
| cost | frames per second on this GPU, GPU hours per hour of video | |

Scoring is per 1 s window. A window's truth comes from the frame labels inside it: its state is `underwater` or
`air` when all its labelled frames agree, otherwise `crossing` (it contains an exit or an entry); a whale, fish or
seafloor is present when any labelled frame in it shows one clearly. Per-frame numbers are reported next to the
window numbers only to show what the window step buys.

For the clip tool the measure is the share of whale time kept (losing whale footage is the costly error) against
the share of worthless footage dropped.

Split (revised 2026-11-20 after WP1): a claim about new footage needs a test on a DEPLOYMENT the method never saw.

- Whales, fish, back, seafloor (truth = by-eye labels): fitted and tuned on `2025-12-20 CG.m4v` only; the 2025-02-03
  recordings are the test and nothing is fitted on them.
- Water state (truth = the tag's depth sensor, which labels every second of the 2025-02-03 recordings for free):
  fitted on pieces 00004-00006, checked on pieces 00007-00008 of the same deployment, and tested across deployments on
  `2025-12-20 CG.m4v` (by-eye labels) and on the public deployment (Zenodo 4902325) if it is fetched.

- Trained parts (from 2026-11-22: the whale-against-fish box classifier and the seafloor / sand classes, WP10) need
  training footage, so the three labelled sources (2025-12-20 clip, 2025-02-03, 2025-01-02) are used
  leave-one-source-out: a number is only reported from a fold that did not see its own test source, and it says
  which source it was trained on. Settings of the text-prompted detector stay as fixed on the 2025-12-20 clip.

The 2025-12-20 clip is 95 s with whales almost throughout, so it cannot test the "drop" side. The sensor log shows
what can: pieces 00001-00003 (deck, two failed drone drops, a float test) and 00012-00014 (the tag floating after
release), which were not on disk on 2026-11-20.

## Work packages

| | what | needs |
|---|---|---|
| WP0 | Scaffold: environment, `uwid` package (video reader, burned-in clock reader, contact sheets, inventory of a footage folder, clip cutter, command line), tests, README | |
| WP1 | Metadata and sensors: read the tag CSV, find what depth, light, motion and the camera fields give for "underwater", "surfacing" and "tag not on a whale"; how sensor time maps to video frames | the CSV copy to finish; videos of that deployment for the sync check |
| WP2 | Research: which current models can find whales and fish underwater from a text prompt or with little training, which public data can test fish and whale-against-fish, prior work on camera-tag video | web |
| WP3 | Reference labels for the development clip: frame labels at 2 per second, exact surfacing events, pixel masks on about 30 frames | |
| WP4 | Water state from the image: colour statistics, a zero-shot image-text model, a small classifier on DINOv3 features, smoothing over time; surfacing events; checked against depth | WP0, WP3, WP1 |
| WP5a | Whales, fish, back, seafloor with text-prompted models: OWLv2 and LLMDet boxes with SAM 2.1 masks, Mask2Former for sand / sea, SAM 3 when the user's access arrives; persistence over the frames of a window | WP0, WP3, shortlist from WP2 |
| WP5b | The same classes by semantic segmentation with little training: a small classifier on DINOv3 patch features trained on the development clip's masks, tested on the other deployment | WP0, WP3 |
| WP6 | Fish, and whale against fish, on public data | WP5's model runners, data set from WP2 |
| WP7 | End to end: `uwid clips <dir>` from states and whale presence to kept clips and a dropped list; run on everything on disk; time per hour of video | WP4, WP5 |
| WP8, WP8b | Tracking demos on kept clips (OWLv2 boxes start SAM 3 tracker tracks, outline overlays); gap experiment (blocked frames cut out of the tracker's view). Done 2026-11-22 | WP7 |
| WP8c | Tracker policy that ends a track rather than risk a swap (purity over continuity). Running 2026-11-22 | WP8b |
| WP3b | Labels for 2025-02-03 (quality grades, silhouettes) and for the second tag CC-09-85 (seafloor, sand, blocked spans). Done 2026-11-22 | |
| WP9 | `uwid scan` on the second tag; keep-anything rule, tiers, descriptions, decision-coloured sheets. Done 2026-11-22 | WP7, WP3b |
| WP10 | Whale / fish / other box classifier (300-800 px band, blow at breaths) and seafloor / sand / dark per window. Classifier done 2026-11-22; seafloor and sand withdrawn 2026-11-23 | WP3b, WP9 |
| WP11c | Held-out test of the lone-whale link across breaths (up to 8 encounters chosen by a written rule from the scan and sensors, both tags) and up to 2 group stretches; all rules frozen. Done 2026-11-23 13:35: 3 breath encounters and 2 group stretches existed, about 20 GPU min; no wrong link, no moved number | WP11a |
| WP12 | Seafloor redo: dense search around blackouts and bottom contact, candidate sheets for the user first, new spans and masks, re-fit. Search done 2026-11-23 (clear seafloor in 00017 only, 94 s); candidate sheet confirmed by the user 2026-11-23 midday; step A done 13:03: per pixel it works inside 00017 (held-out episodes: 96.7 % of seafloor pixels, IoU 0.89; back + seafloor as one class IoU 0.96), as a 1 s window label it does not (faint episode 0/35; 21-35 % of 00023's back-with-haze windows called seafloor; one near whale's underside on 2025-02-03). Head saved as `head_v2_*`, NOT switched into the scan. Waits for bottom footage from a second recording and the user's word on the sediment sheet. Step B done 2026-11-23 23:21 (reviewed 23:22): sediment confirmed by the user that evening; a re-fitted picture `sand` head finds 58-59 of 77 held-out sediment windows but also calls 55-75 % of the confirmed seafloor windows, 102 of 386 windows of 00024 and 333 of 1,736 on 2025-02-03 (empty water): not usable, not switched in. Sensors mark the scraping episodes (rolled about 165 degrees, active, within 2.4 m of the recording's deepest point; a trial flag fires on 0 of 19,248 windows elsewhere): exploratory, one event; a sensor `bottom contact` label is the proposal | more recordings of CC-09-85 with bottom (00013-00016, 00018-00022 are not on disk); user confirmation of `sand_candidates_best_01.jpg` |
| WP11b | Classifier veto, dark / sand labels and the user's 30 s clip guard in `uwid scan`; both deployments re-scored. Done 2026-11-23 | WP10 |
| WP11a | Tracking stage: online purity policy, `uwid tracks` on hq spans, lone-whale link layer, held-out test, cost estimate. Done 2026-11-23 03:34; follow-ups on the 00024 encounter done 12:31 (blocked rule, stitched link, links.csv); whole-deployment runs need the user's go-ahead | WP8c, WP10 |
| WP13 | `uwid tracks` on all hq spans of both deployments with the rules frozen as of WP11c; review pack (random track sample, every link made, numbers that start soon after a breath, starts refused as own back). User's go-ahead 2026-11-23 evening. Done 2026-11-24 01:42 (reviewed 01:47): 135 spans, 786 s tracked, 149 numbers (79 >= 3 s), 8 links, 2.25 GPU h; results in `experiments/07_tracks_demo/tables.md` D9; one new flag in `trackstage` (`frame_store`, off by default; compared on one 4K span only) | WP11c |
| WP14 | Two errors named by the user 2026-11-23: a calf close in front of the tagged mother refused as own back (00005 604-629 s); bubbles numbered as animals after a breath (00024). Measure how often, test a per-recording own-back template, candidate sheets for the user before anything is fitted. Study only, no live change while WP13 runs | WP5b, WP11c. Done 2026-11-23 22:22 (reviewed 22:24): the calf was swallowed by the tracker's own SAM back object, not by the patch map; own back = SAM back object cut to a per-300-s template of where the back usually is frees it (0.998 / 0.999 inside own back -> 0.377 / 0.185) and flips nothing else in 21 stored runs; 17 more candidate stretches on a sheet for the user; bubbles: nothing separates the one confirmed case from real whales (paleness also flags white fins), no rule. Change not wired in: after WP13, behind a flag, replayed on WP13's refused starts | WP5b, WP11c |
| WP15 | 3D mesh trial (opened by the user 2026-11-23): single-image 3D from hq tracked frames (image + outline), SAM 3D Objects if access and GPU memory allow, otherwise an open model. Set-up and smoke test on existing tracks now; the trial on WP13's tracks after | WP13. Set-up and smoke test done 2026-11-23 22:35 (reviewed 22:38): SAM 3D Objects not usable here yet (gated, no access on this account; stated need 32 GB against a 20 GB card); stand-in TripoSG (MIT) runs in 5.2 GB at about 32 s per frame; 10 frames: 4 read as a whale, 3 as a humpback (one animal seen from below), the rest are invented shapes, and mesh distance does not tell animals apart. Weights in `NOAA_whales/underwater_models/`. Phase 2 done 2026-11-24 03:55 (reviewed 03:58): only 28 frames of 13 of 79 numbers show a whole whale (3,191 of 4,158 judged frames are cut by the frame edge); whole views: 18 of 28 meshes read as a whale; whales cut by the frame (second arm, 45 numbers): 15 whole whale, 23 a part, 7 invented; the mesh follows the outline from the camera's side (IoU median 0.84 / 0.88), has no surface, and does not tell numbers apart (AUC 0.63). Open with the user: fit the rigged humpback to a track's outlines instead; repeat on SAM 3D Objects if access comes | WP13 |
| WP17 | Own back in the tool = the tracker's back object cut to the per-300-s template of WP14, behind a flag that is off by default (scan box filter and tracker clean-up). Replay with the flag on: the calf stretch 00005 604-629 s, the 17 candidate stretches of WP14, and a count of every scan decision that changes on both deployments; sheets for the user. The flag becomes the default only after the user has seen them. Done 2026-11-24 04:44 (reviewed 04:47): flag `own_back_template` in `trackstage.run_span`, off by default, 86 tests; on WP13's 135 spans it changes nothing (0 of 1,618 start boxes, 0 roles); on the 17 stretches the calf gets a number for 19.8 s and 8 more numbers appear in 6 stretches (7 after links), among them two whole whales of 23 s and 15 s that were refused before and one dark wedge that may be the mother's own flipper; frame store against decoding identical on one 1080p stretch. Waiting for the user: the calf sheet and the wedge, then the default (answered 2026-11-24 evening: yes, see WP18) | WP13, WP14 |
| WP18 | On the user's answers of 2026-11-24: `own_back_template` on by default in the tracker (the old behaviour stays reachable); an outline keeps its main piece and drops far small pieces (the speck in the calf video's last seconds), with counts of what that changes on every stored run; the calf video again. Done 2026-11-24 21:37 (reviewed 21:40): default on (`--no-own-back-template` for the old behaviour); the 135 stored runs carry the template, numbers identical (157 / 149); specks dropped on read from 837 of 27,331 object-frames of the two deployments (median 0.4 % of an outline), no number changes; `ownback_replay_calf_v2.mp4`; 90 tests | WP17 |
| WP19 | Hand-over to the web-app agent: `uwid export` writes a bundle under `underwater_cache/export/` that needs no pandas / parquet / this repo's environment (recordings, per-second labels, kept stretches with descriptions, tracks with boxes and outlines, a browser-playable clip per tracked span); `HANDOFF_WEB.md` says what is there, how far to trust it and the rules. Started 2026-11-24 21:30 | WP18 |
| WP16 | Thought experiment (user, 2026-11-23): what could tag video and tracks add to the photo sorting of the same day. Reasoning by the main thread, plus one check with data: do photo times line up with the tagged whale's surfacings on 2025-02-03 and 2025-01-02. Nothing is fitted or built |. Done 2026-11-23 21:25 (reviewed 21:28): camera about 3 min ahead of the tag, joined to under 1 s from the photographed tag-on; 02-03: 57 of 185 photos within 5 s of a tag surfacing (chance 12), all of the tagged group; 01-02: 22 photos, all of the whale the notes mark as tagged, but its video pieces 00013-00016 are not on disk; no photos of 2025-12-20 on disk | |

WP0 to WP3 start together. WP4 and WP5 start when WP0 and WP3 are done. Test labels for new videos are made the same
way as WP3 once the videos are on disk.

## Parked until phase 1 is answered

(2026-11-23: the user opened two of these as trials: a 3D mesh trial, WP15, and a thought experiment on the link
to the photos, WP16. Grouping tracks into individuals stays parked.)

- Tracks and individuals: follow each whale through a clip, then group tracklets by markings. A clip gives free
  training signal: frames of one track are the same whale, two whales in one frame are different whales.
- 3D: fit the rigged humpback from `Whale-Image-Data-Factory` to masks and landmarks and unwrap the skin onto the
  model, rather than reconstructing a moving animal from one moving camera with no fixed background.
  **Paused by the user 2026-11-24** ("we may pick it up later"). Where it stands:
  - Done: single-frame image-to-3D trial with the stand-in model TripoSG (MIT) on the tracker's numbers (WP15; LOG
    2026-11-23 22:35 and 2026-11-24 03:55; `experiments/10_mesh3d/tables.md` S0-S5). Only 28 frames of 13 of 79
    numbers show a whole whale; 18 of 28 meshes read as a whale; whales cut by the frame: 15 of 45 a whole whale, 23
    a part; the mesh follows the outline from the camera's side, has no surface (no marks), differs between frames
    of one whale 1-4 s apart, and does not tell numbers apart (AUC 0.63).
  - Not done: the model the user named, SAM 3D Objects (`facebook/sam-3d-objects`: gated, needs the user's access
    request; stated need 32 GB against the 20 GB card; the same 73 frames can be rerun: `picks.csv`,
    `picks_partial.csv` under `underwater_cache/mesh3d/`).
  - Proposed, not started, not yet answered by the user: fit the rigged humpback to a track's outlines over many
    frames and paint the video onto it (keeps the real surface; uses the whole track).
  - To pick it up: environment `underwater_cache/envs/triposg` (5.7 GB), weights
    `NOAA_whales/underwater_models/TripoSG` (7.5 GB), code `experiments/10_mesh3d/` (`pick_frames.py` ->
    `prep_inputs.py` -> `run_triposg.py` -> `trial_report.py`), meshes and renders `underwater_cache/mesh3d/`,
    sheets `review_sheets/underwater/mesh3d_*`. About 32 s and 5-6 GB per mesh.
- Link to the photo catalogue: the burned-in clock and the tag's GPS put a video frame in an encounter; marks shared
  by both views (fluke trailing edge and underside, dorsal fin and hump) carry the identity.

## Open questions for the user

1. (Answered 2026-11-20, see above: the back is its own class and is not another animal.)
2. SAM 3 (the current best text-prompted segmenter) is behind a licence click on Hugging Face and this machine has
   no Hugging Face login. Until the user logs in we test the open alternatives.
3. Raw, unselected video (tests on deck, failed drops, a tag floating after release, long empty stretches) is what
   the "drop" side needs. So does fish: none seen so far.
4. The dev clip has a clock burned into the picture. Do the tag's own `.mov` files have it too, or only the
   conversions? (WP0 checks.)
