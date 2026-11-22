# 07_tracks_demo (WP8): stage 2 = masks and individual tracks through stage-1 stretches, with outline videos

The chain the user asked for: stage 1 (OWLv2 "whale", 2 frames per 1 s window, WP5a's fixed operating point) marks
the windows with a whale; stage 2 runs a video segmenter on a short stretch of those windows, every frame, and keeps
each whale's identity. Code: `src/uwid/tracks.py` (stage-1 reader, `run_tracks`, `load_tracks` cleaning,
`track_summary`, `reference_overlap`, `render_tracks`), scripts here. Nothing was fitted or tuned on any video; the
few rules below (start size, small size, back share, duplicate share) were set once before the runs and not changed
after looking at results, except the "fish-sized track" rule (section 4), added after seeing clip 05.

Truth for outlines: WP3's reference masks (AI by-eye labels, partly SAM 2.1 outlines themselves). Identity: my own
by-eye check of dense sheets (every 0.4-0.5 s, zoomed where unclear); there is no identity truth. All "swap / lost"
counts below are AI by-eye judgements on one viewing.

## 1. Stretches (chosen from stage 1's runs, `stretches.py` -> `results/stretches.csv`)

A stage-1 run = consecutive whale windows (both frames of a 1 s window with a whale box >= 0.25), gaps of one
whale-free window bridged. I listed the runs of >= 8 s per recording with the number and size of whale boxes, looked
at 12 frames of 10 candidates, and picked one per kind. Input: the tag's `.mov` for 2025-02-03 (same frame index as
the `.m4v` that stage 1 and the reference masks used; checked on 2 frames: the same picture), the `.m4v` for the dev
clip (the only form). Each stretch starts on a stage-1 frame (frame index a multiple of 15).

| clip | recording | times (s) | stage-1 run it sits in | kind |
|---|---|---|---|---|
| 01 | 2025-12-20 CG.m4v (dev) | 24.0-36.0 | 18-66 | (a) competitive group, 5-6 whales crossing |
| 02 | 2025-12-20 CG.m4v (dev) | 61.0-76.0 | 18-66 and 69-87 (the bubble curtain splits them) | (d) whales hidden by a bubble curtain and back |
| 03 | CC-11-96-...-00007.mov | 1212.5-1230.0 | 1214-1230 | (b) one large whale passing in good light |
| 04 | CC-11-96-...-00004.mov | 650.0-668.0 | 650-724 | (b) very near whale head-down, touching the back |
| 05 | CC-11-96-...-00006.mov | 456.0-476.0 | 409-485 | (c)+(d) calf close with fish, goes down behind the back and returns |
| 06 | CC-11-96-...-00005.mov | 736.0-754.0 | 735-760 | (e) far hazy whale coming closer |
| 07 | CC-11-96-...-00006.mov | 822.0-840.0 | 821-868 | (e)+(d) hazy calf near the back, dives out of view, fish |

## 2. How tracks start (step 2 of the brief), clips 01 and 05

Three ways, same stretches, same back prompt, same cleaning and drawing:

- (i) `text_sam3`: SAM 3 video (`Sam3VideoModel`), text "whale" alone; its own detector runs on every frame and adds
  / drops objects. The back comes from (ii)'s run (SAM 3 video takes no box prompt).
- (ii) `owl_sam3`: SAM 3 tracker (`Sam3TrackerVideoModel`). Object 0 = the back from a fixed box over the bottom
  25 % on the first frame. On every stage-1 frame (every 15th) each OWLv2 whale box >= 0.25 whose longest side is
  >= 0.10 of the frame width starts a new track unless a current track already explains it (a track mask lies >= 50 %
  inside the box or fills >= 25 % of it, or >= 60 % of the box is back).
- (iii) `owl_sam2`: the same with SAM 2.1 video (`Sam2VideoModel`, hiera-large).

| clip | method | whale ids shown | reference whales matched | median IoU (all ref. whales, unmatched = 0) | back IoU | GPU s per video s | peak GPU GB |
|---|---|---|---|---|---|---|---|
| 01 | (ii) owl_sam3 | 6 | 16 / 17 | 0.80 | 0.93-0.98 | 13.1 | 2.0 |
| 01 | (i) text_sam3 | 7 | 16 / 17 | 0.71 | (from ii) | 15.4 | 3.2 |
| 01 | (iii) owl_sam2 | 4 (+1 drawn small) | 13 / 17 | 0.75 | 0.97-0.99 | 8.1 | 1.4 |
| 05 | (ii) owl_sam3 | 4 (2 are fish) | 1 / 1 | 0.95 | 0.70 | 12.4 | 2.5 |
| 05 | (i) text_sam3 | 3 | 1 / 1 | 0.98 | (from ii) | 9.3 | 2.7 |
| 05 | (iii) owl_sam2 | 5 | 1 / 1 | 0.43 | 0.71 | 8.1 | 1.9 |

Reference frames: clip 01 f720, f810, f900, f1065 (17 whale instances: 6 clear, 9 hazy, 2 faint); clip 05 f13991
(the calf, hazy). The faint "pale shape along the fluke" at f810 is matched by nobody (IoU <= 0.02, the label itself
is uncertain); "left, spotted" at f900 is unmatched in all three because the near whale's outline swallows it.

By eye (dense sheets):

- (ii) owl_sam3, clip 01: every whale in view gets one id; 0 identity swaps seen. Whale 2 (near fluke whale) carries a
  separate thin piece below it on 25.6-34.4 s (probably its other white flipper; cannot be told apart) and at 29.8-
  30.4 s its outline swallows the spotted whale at the left (reference f900: IoU 0 for that whale). Whale 3 joins two
  whales (the bubbly whale at the right edge and a second whale below it) at 25.6-26.0 s, then settles on one. Whale 4
  (a fluke cut by the top-left corner) is lost after 0.4 s. Clip 05: the calf is whale 1 for all 20 s, including when
  it goes down behind the back's hump (471-476 s) and comes back up; 8 more tracks start on fish (OWLv2 calls close
  jacks "whale"), 6 of them stay fish-sized and are drawn grey "small", 2 grow past the size rule and are numbered.
- (i) text_sam3: the same whales and similar outlines on clip 01, but SAM 3's text detector also produces thin masks
  hugging the left / right frame edge from ~26 s on (a strip drawn along the border, attached to a whale's id; on
  clip 05 to the calf's id from 458 s), and it makes the near whale's long flipper a whale of its own (id 7,
  24.6-30.4 s). On clip 05 it ignores most fish (good) but finds the calf 0.5 s later than stage 1 did. The
  transformers build here lacks the `kernels` package, so SAM 3 video's hole filling / sprinkle removal are skipped;
  the edge strips may be that.
- (iii) owl_sam2: clip 01 similar but the far whale's fluke centre-frame gets an outline that stays under the
  fish-size rule (drawn grey, no number), the far whale at the lower right gets its track 0.6 s later (27.1 vs
  26.5 s), the top-left fluke gets none, and the same two-whale join of whale 3 happens at 25.6-26.0 s. Clip 05 is where it fails: at 467.9 s only the fluke is outlined (the
  body is lost; reference IoU 0.43 against 0.95), and when the calf comes back along the bottom at 471.4 s it is split
  into the old id (head) and a NEW id 5 (body); from 474.9 s the calf is only id 5: one identity change. SAM 3 kept
  id 1 whole through the same moment.

Verdict: (ii) OWLv2 boxes + SAM 3 tracker. It starts tracks exactly where stage 1 saw whales (the point of the chain),
has no edge strips, catches the far whale earlier than SAM 2.1, and kept the calf's identity where SAM 2.1 lost it.
Its weakness is that it inherits stage 1's fish-as-whale boxes (clip 05); (i) is better on that one point. SAM 2.1 is
the cheapest (8 vs 12-13 GPU s per video s) but its outlines and identity are clearly worse on the test clip. Both
clip 01 and clip 05 are rendered twice (SAM 3 and SAM 2.1, same start).

## 3. Per clip, chosen method (ii)

Videos and sheets: `/home/sat3737/NOAA_whales/review_sheets/underwater/demo_tracks_NN_<name>.mp4` / `.jpg`
(1280x720, full frame rate 29.97, H.264 yuv420p, no sound track, 2.6-24 MB; sheet = 8 frames across the clip). On
every frame: whale outlines in a fixed colour per id with the id near the top of the outline; fish-sized tracks thin
grey `sN small`; the back thin light blue `back`; top line = file name, time in the recording, burned-in clock,
method; second line = ids in view, and ids last seen large that are gone ("lost: N (since t)").

| clip | whale ids shown | swaps / lost (by eye) | wrong outlines (by eye) | reference overlap | GPU s per video s | peak GB |
|---|---|---|---|---|---|---|
| 01 competitive group | 6 (5 whales tracked the whole clip + 1 corner fluke) | 0 swaps; id 4 (fluke cut by the top-left corner) lost after 0.4 s | id 2 carries a thin extra piece below it 25.6-34.4 s (probably its own other flipper); id 2 swallows the spotted whale at the left 29.8-30.4 s; id 3 joins two whales 25.6-26.0 s; id 2 spreads along the surface as it leaves at the top (30.8-31.6 s) and leaves a small remnant there 33-35 s | 16/17 whales matched, median IoU 0.80 (unmatched = 0); back 0.93-0.98 | 13.1 | 2.0 |
| 02 bubble curtain | 12 for about 6-7 whales | before the curtain 3 whales, ids 1-3, right. In the curtain (64-66 s) ids 1, 2 and two new ones sit on bubble streaks; 66-68.5 s nothing drawn. After it ids 2 and 3 re-attach (same animals? cannot be told), the other returning / new whales get new ids 7, 8, 10, 11, 12 | outlines on bubbles 64.1-66 s; id 3 joins the huge pectoral fin and the near whale below it 72.5-73.1 s (the reference says the fin is probably that whale's own); a thin outline along the right frame edge 72.6-73 s | 19/22 matched, median 0.76; back 0.56-0.86 (bubbles) | 15.6 | 3.1 |
| 03 single whale passing | 1 | 0 swaps, 0 lost; the track starts at 1213.6 s, about 1 s after the head shows at the top edge (first stage-1 frame with the whale) | flipper briefly a separate piece of the same id 1223.5-1224.5 s; drawn `small` from 1229.3 s as it swims off (right) | no reference frame | 7.4 | 1.5 |
| 04 very near whale on the back | 1 | 0 swaps, 0 lost | none clear; the outline runs along the back's edge where the whale touches it (656-668 s), two specks at the bottom edge (663.6, 667.1 s) | no reference frame | 8.0 | 1.5 |
| 05 calf, fish, behind the back | 4 (calf + its flipper split off at the start + 2 fish) | calf = id 1 for all 20 s, kept while it goes down behind the back's hump and comes back (471-476 s): **old identity kept**. 10 tracks started, 8 on fish | 2 fish numbered as whales (id 3 465.4-474.3 s, id 4 468.4-474.9 s); 6 fish drawn grey `small`; the calf's long flipper is its own track 457.9-459.5 s (id 2), then hidden as a copy | calf 0.95 (1 frame); back 0.70 | 12.4 | 2.5 |
| 06 far hazy whale approaching | 1 | 0 swaps, 0 lost | from 0.1-0.2 of the frame width (736-744 s) the outline is a rough blob; it tightens as the whale comes closer (to ~0.5) | no reference frame | 7.9 | 1.5 |
| 07 hazy calf near the back | 0 (the calf is drawn grey `s1 small`) | **1 swap**: the only track starts on a fish at 822.2 s and moves onto the calf by 824.8 s; because it then "covers" every later stage-1 box of the calf (0.10-0.23 of the width), no whale track ever starts; from 837 s s1 sits on the back's hump | the back's outline swallows the calf 824-826 s; s1 covers only the calf's head and upper body; the whole track stays under the fish-size rule | no reference frame | 7.7 | 1.5 |

Twice rendered (same start, same rules, SAM 2.1 video in place of the SAM 3 tracker):
`demo_tracks_01_comp_group_crossing_SAM2.1_same_start.mp4` and `demo_tracks_05_calf_fish_behind_back_SAM2.1_same_start.mp4`
(section 2 says what differs; in 05 watch 467-476 s).

Outline overlap with the reference masks by grade (clips 01, 02, 05; unmatched reference whale = 0):

| method | grade | n | median IoU | IoU >= 0.5 |
|---|---|---|---|---|
| owl_sam3 (01, 02, 05) | clear | 13 | 0.90 | 11 |
| owl_sam3 (01, 02, 05) | hazy | 21 | 0.77 | 18 |
| owl_sam3 (01, 02, 05) | faint | 6 | 0.29 | 2 |
| owl_sam2 (01, 05) | clear | 6 | 0.92 | 5 |
| owl_sam2 (01, 05) | hazy | 10 | 0.51 | 5 |
| text_sam3 (01, 05) | clear | 6 | 0.90 | 5 |
| text_sam3 (01, 05) | hazy | 10 | 0.70 | 9 |

For comparison WP5a's single-frame numbers on the dev mask frames: SAM 3 text 0.89, OWLv2 + SAM 2.1 0.83 (matched
instances only; here unmatched count as 0, and frames are reached by tracking, not prompted).

What fails, and how often (7 clips, about 118 s): identity swaps seen 1 (clip 07, fish to calf); a returning whale
got its old id in clip 05 (SAM 3) and in 2 of the post-curtain whales of clip 02, new ids for the rest; outlines on
bubbles in clip 02 only; two-whale joins 3 times (01 twice, 02 once); fish numbered as whales 2 (clip 05); the back
swallowing a whale once (clip 07); thin frame-edge outline once (clip 02). Clips 03, 04, 06 have nothing clearly wrong.

## 4. The back and small things

- Back = object 0 of every run, a thin light-blue outline labelled `back`. IoU with the reference back: dev clip 01
  0.93-0.98 (4 frames), clip 02 0.56-0.86 in bubbles (7 frames), clip 05 0.70 (1 frame).
- Whale tracks lying >= 60 % on the back (median over their frames) would be dropped; none were (the largest back
  share of any whale track was 0.002). Stage-1 boxes lying >= 60 % on the back start no track.
- Fish-sized: a track whose outline's longest side stays below 0.15 of the frame width (90th percentile) is drawn
  thin grey, labelled `small`, with no whale number (added after seeing clip 05, where 6 of 8 fish tracks fall under
  it); a whale track drawn on a frame where it is < 0.10 of the width is drawn thin grey with its number and `small`.
- Duplicates: on a frame where >= 70 % of one whale outline lies inside an older one, the younger is not drawn; a
  track hidden like that on (nearly) all its frames gets no number.

## 5. Cost

Measured on this machine's shared RTX 4000 Ada (other jobs ran on the GPU at the same time), every frame, from
decoding to stored masks, model loading (~20 s per stretch) not counted:

| method | video s | GPU s per video s | range per clip | peak GPU memory |
|---|---|---|---|---|
| owl_sam3 (chosen) | 117.5 (7 clips) | 10.1 | 7.4 (one whale) to 15.6 (12 tracks, bubbles) | 1.5-3.1 GB |
| owl_sam2 | 31.6 (2 clips) | 8.1 | 8.1-8.1 | 1.4-1.9 GB |
| text_sam3 | 31.6 (2 clips) | 11.6 | 9.3-15.4 | 2.7-3.2 GB |

Memory stays flat over a stretch because tracker history older than 24 frames is dropped (`_prune`). Masks are
stored bit-packed at 1280x720: about 0.12 MB per video second (`underwater_cache/tracks/`, 20 MB for all runs here).

A 30-minute recording of the 2025-02-03 deployment (stage-1 kept share = whale windows / all 1 s windows, pieces
00004-00010, `results/kept_share.csv`):

| what is processed | share of the recording | video s per 30 min | GPU hours per 30-min piece at 8-15 GPU s per s (mean 10) |
|---|---|---|---|
| every stage-1 whale window | 13.4-55.4 % (mean 24.8 %) | 446 (241-997) | 1.2 (0.5-4.2) |
| + 1 s padding each side | mean 28.5 % | 513 | 1.4 |
| only windows whose largest whale box >= 0.2 of the width | mean 15.7 % | 283 | 0.8 |
| only windows whose largest whale box >= 0.3 of the width | mean 8.9 % | 160 | 0.45 |

So stage 2 on every stage-1 window costs about 2.5 GPU hours per hour of video at full frame rate, roughly 8 times
stage 1 (0.3 GPU h per video h). Restricting it to windows where stage 1 already sees a large whale (the user's
interest) brings it to about 0.9 GPU h per video h at >= 0.3 of the width. Running stage 2 on every second frame
would roughly halve it again (not tried).

## Rerun

```
D=experiments/07_tracks_demo; PY=.venv/bin/python
$PY $D/stretches.py                          # stage-1 runs and kept shares (CPU)
nohup $D/jobs/run_all.sh > /home/sat3737/NOAA_whales/underwater_cache/tracks/run_all.log 2>&1 &   # GPU, locked, resumable
$PY $D/render.py 01 02 03 04 05 06 07 --method owl_sam3 --crf 19        # videos + sheets + results/*.json (CPU)
$PY $D/render.py 01 05 --method owl_sam2 --name-suffix _SAM2.1_same_start --crf 19
$PY $D/render.py 01 05 --method text_sam3 --no-video
$PY $D/judge.py 05 --method owl_sam3 --out <dir>                        # dense sheets for judging by eye
$PY $D/report.py                                                         # number tables
```

# WP8b: cutting blocked views out of the tracker's eyes (`gap/`)

The user's idea: detect the bubble curtains and above-water moments and remove them from what the tracker sees, so it
meets a short discontinuity instead of a blocked view.

## B1. Blocked-view rule (fixed 2026-11-21 before any WP8b tracker result was looked at; `gap/blocked.py`)

A frame is blocked when (1) the lens is not under water, or (2) bubbles hide most of the view above the back:
- (1) with a sensor log (2025-02-03): the frame lies in [exit - 0.2 s, entry + 0.2 s] of a depth surfacing
  (`tagdata/surfacings.csv`, `pts_exit_pred` / `pts_entry_pred`); without a log (2025-12-20 clip): >= 25 % of the
  frame's patches classed `air` by the WP5b patch classifier;
- (2) >= 50 % of the patches not classed `back` are classed `bubbles`.
Patch classes: WP5b linear classifier on DINOv3 ViT-B/16 patches at 1280 px. 2025-12-20: cached predictions of the
four fold classifiers, every 5th frame, each frame scored by the classifier whose held-out time block holds it (but
all fitted on that clip: no independent test there, only a tool for choosing frames). 2025-02-03: the classifier
fitted on all 31 dev mask frames, every 3rd frame of the `.m4v`. Frames between scored frames take the nearest call;
blocked runs < 0.2 s are dropped, then open runs < 0.3 s between blocked runs are filled. No new track starts on a
blocked frame in variants b and c.
