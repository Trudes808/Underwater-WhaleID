# Hand-over: camera-tag video results for a video-analysis page in the web app

Written 2026-11-24 by the main session of `Underwater-WhaleID`, for the agent who will build the new page in
`/home/sat3737/NOAA_whales/Whale-Review-Web`. The user asked for this hand-over: "pack things up to be passed off
to another agent who will be working on the web app and will be creating a new page for the video analysis".

You do not need this repo's Python environment, and you should not install anything into it. Everything a page
needs is in one folder of plain files.

## What you get

| what | where |
|---|---|
| the results, as JSON / CSV / H.264 MP4 / JPEG | `/home/sat3737/NOAA_whales/underwater_cache/export/web_v1/` (entry point `index.json`) |
| the contract for every file and field | `docs/EXPORT_FORMAT.md` in this repo (the same text is `README.md` in the bundle) |
| how the results were made and how far each was checked | `PLAN.md` (plan, the user's decisions with quotes) and `experiments/LOG.md` (one entry per piece of work, each with a reviewer's note) |
| pictures and videos the user has already looked at | `/home/sat3737/NOAA_whales/review_sheets/underwater/` |

Read `docs/EXPORT_FORMAT.md` before writing code. It defines time, frame and coordinate conventions, and the one
rule a player must follow: frame j of a bundle clip is frame j of its track file.

## What the footage is

Video from camera tags (CATS tags) that ride on a whale's back, looking along the back. The lower part of most
frames is the tagged whale's own back. Each tag also logs depth and motion. A deployment is one tag on one whale on
one day, recorded in pieces of about 30 minutes.

| bundle id | what it is | size | notes |
|---|---|---|---|
| `2025-02-03_CC-11-96` | tag on a mother with a calf (the user: "a calf swimming close in front of the mom who is tagged with the camera") | 9 recordings, 4.45 h, 1080p | test deployment; 2025-02-03 is also a photo day in the web app |
| `2025-01-02_CC-09-85` | 9 of this tag's recordings, on a whale in a group; boat footage at the start, the whale scraping along the seafloor in the middle | 2.35 h, 4K | test deployment; 2025-01-02 is also a labelled photo day; recordings 00013-00016 and 00018-00022 are not on disk |
| `2025-12-20_CG` | a 95 s exported highlight of a large competitive group, many whales per frame | 1 file, 1080p, no sensor log | the development clip: the only footage any setting was tuned on; photos of that day are not on disk |

## How the results are produced

1. **Sensors first.** Depth and motion say whether the tag is on the whale, in the air, on a drone, floating, under
   water or at the surface. Picture models run only on under-water, on-whale time.
2. **Scan** (`uwid scan`): a detector looks at two frames per second and gives every second of every recording
   labels (whale, small whale-like shape, fish, picture-quality tier). Seconds with anything detected are joined
   into **kept stretches**, each with a best tier and a text description. Everything else is listed as dropped,
   with the reason.
3. **Tracker** (`uwid tracks`): only on the high-quality spans inside kept stretches, SAM 3 outlines each whale on
   every frame and gives it a **number**.

## Words used in the data

- **Window**: one second of a recording (one row of `windows/<recording>.csv`).
- **Kept stretch**: seconds with activity, joined when gaps are under 30 s, with 2 s of padding. The user's rule:
  a clip begins when a whale appears and ends after more than 30 s without activity.
- **Tier**: `hq`, `ok`, `low` (picture quality of a whale), `small` (fish-sized whale-like shape), `fish`. Tiers
  are labels, not a filter. The user: "we want to keep all footage that has detected stuff in it / any activity.
  Low just shouldnt be used in future when we try to do 3d and animal ID."
- **hq span / `advanced_ok`**: the part of a kept stretch good enough for later identification work. The tracker
  ran on these.
- **Number**: a whale number is per tracked span, not per animal. The tracker follows a purity rule set by the
  user ("new tracks are less of a problem then a track swapping to a new whale"): when in doubt it ends a number
  and starts a new one. The same animal can carry several numbers in one span, and numbers mean nothing across
  spans. Call them "track 1, track 2" on a page, never an identity.
- **Back**: the tagged whale's own back. The user: it is "identified as back", and "it shouldnt count towards other
  animal in scene". It is exported as its own polygon.
- **Blocked frames**: frames hidden from the tracker (lens out of the water, bubbles). No outlines there.
- **Groups of tracked spans**: `hq` are the spans the tracker runs on by rule. `extra` are 17 stretches near the
  tagged whale's back that were tracked on request; they include the calf stretch the user reviewed (2025-02-03,
  recording 00005, 607-627 s).

## How far to trust each thing

All picture-side checks are an AI agent's by-eye labels unless a line says the user confirmed it. Counts are in
`PLAN.md` ("Findings so far") and the LOG.

| thing | trust | what to keep in mind on a page |
|---|---|---|
| under water / surfaced / in air, with a sensor log | high | from depth, good to about 0.3 s; the dev clip has no log and uses a picture model |
| tag on the whale / drone flight / floating / in air | high on these two tags | rules rest on few events (3 drone drops) |
| a whale is in view this second | good | on 2025-02-03, 192 of 221 labelled whale seconds found, 7 of 570 false |
| whale against fish | weak for small things | the user accepts that small whale-like shapes may be called fish; only large, clear whales matter |
| tier `hq` | useful | the user judged sample sheets: "the high are best the ok are fine and the low are not useful"; `ok` against `low` and the silhouette flag are weak rules |
| the text description of a stretch | read as a summary | "five whale boxes at once" counts detector boxes, not animals; the scan often puts two boxes on one near whale |
| seafloor, sand, dark | low | fitted on one recording; `seafloor_share` is high in open water on 2025-01-02. Do not present these as findings |
| a tracked outline is on a whale | good | in a random sample of 24 numbers every outline was on a whale or part of one |
| a number stays on one animal | good in what was checked | no number moved between animals in the sample; all 8 joins checked were right |
| numbers in crowded scenes | short | with several whales in view the purity rule ends numbers early; expect many short numbers on the dev clip |
| shapes right after a breath | doubtful | the user found one "pale animal" after a breath to be "just bubbles"; one exported number sits on the sea surface (2025-02-03, 00005, 379.7-380.7 s, number 2) |
| the calf stretch | user-reviewed | the user on the video with the current rules: "looks quite good"; number 2 there "is a calf but it is quite far in distance" |

What the tracks do not cover: stretches that are not high quality (most of the kept time) have labels but no
outlines.

## Video facts that will bite

- **The source recordings do not play in most browsers.** They are HEVC (2025-02-03 tagged `hev1`, 2025-01-02
  4K `hvc1`). The bundle's clips are H.264 at 1280 x 720 and play everywhere, but they exist only for tracked
  spans (152 in the first run; see `index.json` for the current count).
- **Whole-recording playback needs a proxy that nobody has made.** 6.8 hours of source video is a real cost in CPU
  and disk; ask the user before making it. There is no system ffmpeg on this machine; this repo's `.venv` has it
  through `imageio-ffmpeg` (you may run that interpreter, not install into it).
- **Do not use the `.m4v` copies of 2025-02-03 as a time base.** They are H.264 and play, but they have dropped
  frames, drift of up to 1.5 s per file and washed-out levels. Every time in the bundle refers to the `.mov`.
- **Already-cut clips of kept stretches** exist for 2025-02-03 only (`underwater_cache/clips/`, 155 of 158 are
  HEVC copies) and do not play in a browser either.
- **Review videos with outlines drawn in** (40 files, H.264) are in `review_sheets/underwater/`, for example
  `demo_wp13_01..06_*.mp4` (the six longest tracks) and `ownback_replay_calf_v2.mp4` (the calf stretch).
- **Clocks.** `clock` is the clock burned into the picture. On 2025-02-03 it is the sensor time plus 2.03 s. On the
  dev clip it reads about two hours later than the field notes' time for that group; that question is open with
  the user.
- **Frame rate.** The 2025-02-03 recordings are not exactly 30000/1001 fps. Each track file lists the true source
  time of every frame; use it when you show times.

## Rules that bind a page

These come from this repo's `CLAUDE.md` and from the user. They apply to anything you build on this data.

- `/home/sat3737/NOAA_whales/underwater_video/` is read-only. Never write, move, rename or delete there.
- Treat `underwater_cache/` and this repo as read-only too. If you need something changed or added in the export,
  rerun the commands below or ask the session that owns this repo; do not edit bundle files by hand.
- Nothing made from the footage goes into a git repository: no frames, clips, posters, sensor tables, positions.
  Serve media from the cache path; do not copy it into the web app's tracked files. Do not upload footage or
  frames to any outside service.
- People: the first minutes of several recordings are on the boat or on a drone (tag state `air` or `flight`) and
  can show the crew. The scan drops those parts and no bundle clip contains them. Do not show them, and never name
  or identify a person. Photo file names from the photo days carry a person's name; none are in the bundle.
- If you ever need the GPU: one shared 20 GB card, every job as
  `flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 900 nice -n 10 <command>`, at most 9 GB. Nothing in
  the bundle needs it.

## Not in the bundle, and where it is

| thing | where | note |
|---|---|---|
| depth at full rate, motion, light | the tag's log under `underwater_video/<deployment>/Data/` | `windows.csv` has depth once per second, which is enough for a timeline |
| which photos were taken while the tagged whale surfaced | counts only, in `experiments/11_photo_link/tables.md` | the tag clock and camera clock differ by about 3 minutes (camera ahead; offsets -173.3 s and -170.2 s). The cache for this holds photo keys with a person's name: do not display or copy them. Linking video to the photo pages is a decision for the user, and which whale carries the tag on 2025-02-03 is still open |
| 3D meshes | `review_sheets/underwater/mesh3d_*`, state under "Parked" in `PLAN.md` | paused by the user on 2026-11-24; not for the page unless the user asks |
| raw masks and per-frame tables | `underwater_cache/tracks/` (npz, parquet) | the bundle's polygons agree with the masks to IoU 0.986 or better on 20 random checks |

## Ideas for the page

These are suggestions. The design is yours and the user's.

1. A recording timeline from `windows.csv`: the depth trace with bands for tag state, water state, tier and kept
   stretches. It shows at a glance what was dropped and why, which is the user's first goal for this project.
2. A list of kept stretches with tier and description (`stretches.json`), opening the tracked spans inside each.
3. A span player: the bundle clip with the outlines drawn on a canvas from the track file, with switches for the
   tagged whale's back, small tracks, refused starts, and markers where a number ends or two are joined. The user
   asked early on for "short examples ... with outline overlays" and reviews by watching such videos.
4. Plain labels: "track" for a number, the group (`hq` or `extra`) visible, and a note where trust is low.

## Open questions with the user (not answered as of 2026-11-24)

1. Which whale carries the tag on 2025-02-03 (the tag shows on photos the catalogue files under two animals).
2. Whether the dev clip's burned-in clock is about two hours ahead of local time.
3. Whether recordings 00013-00016 and 00018-00022 of CC-09-85 can be copied over.
4. Whether to drop the picture-based sand label and use a sensor "bottom contact" label instead.
5. Whether the drone-approach footage at the start of recordings should be kept.
6. Whether to request access to SAM 3D Objects, and whether to fit the rigged humpback model to tracks (3D work is
   paused).

## Regenerate and check

```
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
nice -n 10 .venv/bin/python -m uwid.cli export                       # rewrites JSON / CSV, keeps existing clips
/usr/bin/python3 -I experiments/12_export/check_bundle.py structure  # standard library only
.venv/bin/python experiments/12_export/check_bundle.py media         # clips against tracks, polygon accuracy
```

A new deployment enters the bundle after `uwid scan <footage folder>` and `uwid tracks "<deployment>"` (the second
needs the GPU lock). The commands and their options are listed in `README.md`; how the tracker run on all
high-quality spans went is in `experiments/07_tracks_demo/tables.md`, section D9.

## State of this repo at hand-over

No job is running. The tracker's defaults are final as of 2026-11-24 (own-back template on, far specks dropped from
outlines). 3D work is paused. Detection questions that remain open are listed in `PLAN.md`. The user commits this
repo; agents do not.
