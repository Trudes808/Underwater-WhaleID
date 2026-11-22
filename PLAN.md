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

WP0 to WP3 start together. WP4 and WP5 start when WP0 and WP3 are done. Test labels for new videos are made the same
way as WP3 once the videos are on disk.

## Parked until phase 1 is answered

- Tracks and individuals: follow each whale through a clip, then group tracklets by markings. A clip gives free
  training signal: frames of one track are the same whale, two whales in one frame are different whales.
- 3D: fit the rigged humpback from `Whale-Image-Data-Factory` to masks and landmarks and unwrap the skin onto the
  model, rather than reconstructing a moving animal from one moving camera with no fixed background.
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
