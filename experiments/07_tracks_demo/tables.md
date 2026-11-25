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

## B2. Gap tolerance on clean stretches (step 1; identity truth = the uncut run, no eye involved)

`gap/synthetic.py`. Bases: the uncut OWLv2 + SAM 3 tracker runs of dev 24-48 s (competitive group, new uncut run),
demo clips 03, 04, 05, 06. For g = 0.5, 1, 2, 4, 8 s and gap centres at 1/3 and 2/3 of each stretch (clamped so
2 s before and 2.5 s after fit): a fresh run starts 2 s before the gap, the g seconds are not shown to the tracker
(its frame counter stays consecutive: it meets a jump), and it ends 2.5 s after. Per whale that the uncut run shows
both in the last 0.5 s before and 0.5-2.5 s after the gap: its number before = best-overlapping cut track (IoU >= 0.3);
after, the most common best-overlapping cut track decides KEPT / NEW number / LOST / SWAPPED (took the number of
another whale). 50 cut runs, 105 whale cases, 12 left out (not tracked in the 2 s before the gap).

| gap (s) | n | kept | new number | lost | swapped |
|---|---|---|---|---|---|
| 0.5 | 18 | 18 (1.00) | 0 | 0 | 0 |
| 1 | 19 | 17 (0.89) | 0 | 2 | 0 |
| 2 | 18 | 15 (0.83) | 2 | 1 | 0 |
| 4 | 20 | 15 (0.75) | 0 | 2 | 3 |
| 8 | 18 | 13 (0.72) | 1 | 2 | 2 |

| split | 0.5 s | 1 s | 2 s | 4 s | 8 s |
|---|---|---|---|---|---|
| large whales (>= 0.3 of the width), kept | 10/10 | 8/8 | 7/8 | 6/7 | 6/7 |
| small whales (< 0.3), kept | 8/8 | 9/11 | 8/10 | 9/13 | 7/11 |
| 1 whale in view, kept | 6/6 | 6/6 | 6/6 | 7/8 (1 swap) | 6/6 |
| 2-3 whales in view, kept | 2/2 | 1/2 | 1/2 | - | 1/2 |
| 4+ whales in view (dev group), kept | 10/10 | 10/11 | 8/10 | 8/12 | 6/10 |

Reading: a lone whale (the calf, the single passing whale) survives a jump of up to 8 s almost always (31/32);
in the competitive group, survival falls from 10/10 at 0.5 s to 6/10 at 8 s, and from 4 s on, wrong-number cases
(swaps, 3 of 22) appear next to losses. Large whales hold better than small ones (37/40 against 41/53 over all
gaps). Caveats: the cut runs have only 2 s of history before the gap (a longer history may help); the group cases
all come from one stretch of the development clip; n per cell is small. Picture:
`review_sheets/underwater/demo_gap_curve.png`; tables `gap/results/gap_curve.csv`, `gap_outcomes.csv`.

## B3. Real blocked views (steps 3-4)

Cases (`gap/cases.py`): the dev bubble curtain (demo clip 02, 61-76 s), the dev breath (5-22 s; whales 5-11 s and
from 18 s), and the only 4 of 66 2025-02-03 surfacings in video with stage-1 whale windows in >= 2 of the 4 s before
the exit and >= 2 of the 4 s after the entry (00005 372 s, 00006 582 s and 1520 s, 00007 1466 s; the other whale is
the calf, so identity across the breath is known). The dev bubble stretch at 87-93 s was not used: the camera leaves
the water at ~93 s and the clip ends in air, so there is no "after".

Blocked spans from the rule (B1): curtain 63.8-64.9 + 65.5-68.4 s (4.2 s blocked within a 4.7 s episode); dev breath
11.8-17.9 + 19.3-20.4 s (7.3 s within 8.7 s); breaths 3.2-4.2 s each. Variants: a straight through, b blocked frames
cut out (jump), c blocked frames shown but nothing from them stored in the tracker's memory (possible in the
transformers SAM 3 tracker by deleting their memory entries after each step).

Numbers (automatic: ids seen in the 1 s before and 0.5-2.5 s after the blocked episode; "same number" is NOT "same
animal") and my by-eye judgement (one viewing of dense a|b|c sheets; AI by-eye):

| case | gap (s) | variant | ids in clip | outlines drawn on blocked frames | same number before/after | by eye |
|---|---|---|---|---|---|---|
| 01 curtain | 4.7 | a | 12 | 49/125 | 2 of 3 (2, 3) | 2 numbers re-attach, cannot judge if same animals; 4 new ids for whales after; outlines on bubble streaks 64.1-65.6 s |
| | | b | 10 | 0/125 | 3 of 4 (2, 3, 4) | the same 2 re-attachments (cannot judge) + a fish-sized track; 4 new ids after; no outlines on bubbles |
| | | c | 11 | 55/125 | 2 of 3 | like a for identity; outlines on bubbles where the blocked frames are shown |
| 02 dev breath | 8.7 | a | 10 | 144/220 | 4 of 4 (1, 3, 4, 6) | NOT identity: #1 began on the exhalation bubble cloud (before the block) and jumps to a whale after it (all three variants); #4 and #6 come back as stale outlines on wrong pieces (a thin strip, part of a whale); #3 (pale whale in the corner) cannot judge; outlines on the surface froth and in air through the whole block |
| | | b | 9 | 0/220 | 2 of 4 (1, 3) | #1 same wrong jump, #3 cannot judge; the whales after the breath get 3 new numbers with clean outlines; over-blocking: 19.3-20.4 s hides a whale clearly visible through bubbles |
| | | c | 6 | 155/220 | 3 of 4 | between a and b; draws outlines on froth and air |
| 03 calf, 00005 | 4.2 | a / b / c | 2 / 2 / 2 | 21 / 0 / 55 of 126 | calf 1 = 1 in all | calf kept in all three |
| 04 calf, 00006 | 3.2 | a / b / c | 3 / 4 / 4 | 33 / 0 / 59 of 95 | calf kept in all | calf kept in all three; a outlines the horizon / sea seen from air (2 tracks); after re-entry every variant starts one track on the sea surface seen from below |
| 05 calf, 00006 | 3.4 | a / b / c | 1 / 1 / 1 | 60 / 0 / 80 of 102 | calf kept in all | calf kept in all three; a draws a small outline on a dark speck in air |
| 06 calf, 00007 | 4.2 | a / b / c | 3 / 1 / 1 | 17 / 0 / 27 of 127 | a: 0 of 1; b, c: 1 of 1 | a: the calf comes back as NEW #3 and a bogus track #2 starts on the sea surface; b and c keep the calf as #1 with no extra track |

Summary over the 6 cases:
- Outlines on blocked views (air, froth, bubble streaks): a draws them on 324 of 795 blocked frames, c on 431, b on
  none (by construction: it is not shown those frames and starts nothing there).
- The calf across a breath (3.2-4.2 s, one other whale in view): kept 3/4 straight through, 4/4 with b (and c).
  This matches the curve: a lone whale survives a 4 s jump 7/8 times.
- Multi-whale dev cases (4.7 and 8.7 s): no identity gain that can be shown. The animals move and new whales arrive
  during the block; numbers that re-attach cannot be checked by eye; the curve predicts only ~60-67 % kept at 4-8 s
  with 4+ whales even for a clean jump. Cutting does give cleaner tracks after the block in the dev breath (new
  numbers on the right whales instead of stale outlines on wrong pieces).
- Cost: b is cheaper because it skips frames (5.7-10.3 against 7.2-15.6 GPU s per video s).

Verdict: cutting helps a little and hurts nothing measurable here: it removes every outline drawn on bubbles, froth
or air, avoids the stale and bogus tracks they leave behind, and kept the calf's number in the one breath of four
where straight-through lost it; it cannot rescue identities across long blocks in a moving group (the curve says a
4-8 s jump keeps 60-75 % there). The rule's weak point: bubble streaks over a clearly visible whale are called
blocked (dev 19.3-20.4 s), so a whale can be hidden for 1 s for nothing. Variant c keeps identities like b but still
shows outlines on blocked views; b is the one to use.

Videos (left a, right b; blocked frames shown dark on the right, marked "view blocked: not shown to the tracker",
no outlines; 1920x540, H.264 yuv420p, no sound, 8-frame sheet beside each):
`demo_gap_01_dev_bubble_curtain`, `demo_gap_02_dev_breath`, `demo_gap_03_breath_00005_372s`,
`demo_gap_04_breath_00006_582s`, `demo_gap_05_breath_00006_1520s`, `demo_gap_06_breath_00007_1466s` (.mp4 / .jpg).

Rerun: `nohup gap/jobs/run_gap.sh` (GPU queue `gap/run_gpu.py`, resumable), then `gap/synthetic.py eval`,
`gap/real.py eval`, `gap/real.py render`, `gap/real.py judge <case> --out <dir>`.

# WP8c: purity first (`pure/`)

The user, on the gap results: "new tracks are less of a problem than a track swapping to a new whale." A track
number must never move from one animal to another; fragments (one whale under several numbers) are acceptable.
Policy in `src/uwid/tracks.py`: `PurityConfig`, `purity_cuts`, `apply_cuts`, `purify` (guards END a number; the
rest of the track gets a new one) and `fish_like`; the renderer writes "N ended: <reason>" where a guard fired.
The guards run on a finished run (cut and renumber), which is what the online policy does except that online a new
number would start from the next stage-1 box. Reused: every WP8 / WP8b run (no tracker re-run); new GPU work: three
backward runs and DINOv3 patch features every 3rd frame of every reused run (`pure/run_gpu.py`).

## C1. Where numbers move today, and what in the tracker's own outputs shows it

| kind | where (old policy) | signal available at that moment | caught by |
|---|---|---|---|
| swap after a jump | gap runs at 4 and 8 s in the group (3 of 22) | length of the hidden span, number of whale-sized tracks alive before it | blocked-view limit (`max_block_s`, lone exception) |
| fish -> calf | clip 07, 822.3-824.9 s | first second moves 2.9-3.9 body lengths/s at 0.17 of the width (fish-like); appearance drifts 0.05 -> 0.24 within 2.5 s; speed falls below 1 body length/s only at 826 s | strict appearance drift (0.10) + fish screen |
| join of two whales | clip 01 25.6-26.0 s (#3) and 29.8-30.4 s (#2 swallows the spotted whale); clip 02 72.5-73.1 s (#3, fin + near whale) | clip 02: area jump and swallowing of another live track; clip 01: none: the swallowed whale had no track of its own and the outline grew smoothly | clip 02 only (swallow / area jump); clip 01 not caught |
| bubble cloud -> whale | dev breath, #1 starts on the exhalation cloud (11.7 s) and is on a whale at 18.3 s | 6 s blocked view; the start region is bubbles | blocked-view limit |
| stale outlines after a breath | dev breath #4, #6 land on wrong pieces | blocked view, re-found after absence, appearance drift 0.16-0.22 | several |
| fake track on the sea surface after re-entry | 00006 582 s, 00007 1466 s | start region is surface (patch classes) | start screen (bubbles / surface / air share) |
| fish numbered as whales | clip 05 (2), fish swapping among fish | fish-like first second | fish screen (whale number withheld) |

Fish against whale in a track's first second (`pure/results/fishstats.csv`; displayed tracks of clips 01, 03, 05,
06, 07 and two breaths): fish tracks move 2.75-8.35 body lengths per second at 0.10-0.25 of the frame width; whale
tracks 0.07-1.01, except a far whale entering at the top edge (5.3, but 0.59 of the width) and two fake surface
tracks. So "smaller than 0.3 of the width AND faster than ~1.5-2.5 body lengths/s in the first second" separates
them here. This threshold was set looking at clip 05 / 07 (test recording 00006): exploratory, needs another
deployment with fish to confirm. The 287 px fish (0.15 of the width) that OWLv2 scores 0.54 "whale" is of this kind.

## C2. The trade (`pure/evaluate.py`, `pure/results/trade.csv`)

Truth for "moved": the 50 gap cut runs against their uncut run (a displayed track that overlaps, for >= ~5 frames
each, two uncut tracks that the uncut run shows apart at the same time = two animals); by eye elsewhere (9 moves /
joins I saw in WP8 / WP8b, `EVENTS` in `evaluate.py`; AI by-eye). Strengths are named sets of all guards.

| policy | numbers moved (gap runs, all / whale numbers) | tracks per uncut whale | whale frames still outlined (gap runs) | by-eye moves fixed | whale numbers on the 7 demo + 12 real runs |
|---|---|---|---|---|---|
| old | 18 / 12 | 1.11 | 0.742 | 1/9 | 108 |
| loose | 10 / 7 | 1.40 | 0.735 | 6/9 | 134 |
| medium | 6 / 4 | 1.64 | 0.722 | 6/9 | 160 |
| **strict (shipped default)** | **0 / 0** | **1.95** | **0.685** | **7/9** | **209** |

One guard at a time (medium strength): appearance drift 18 -> 9 moved (7 whale numbers), shape / area jump 12 (8),
blocked-view limit 15 (9), absence 15 (12), swallowing 18 (12), fish screen 19 (8 whale numbers: it withholds whale
numbers from fish), grows-from-fish-size 17 (11), start screen 18 (12; it removes fake starts, not moves). So the
drift and jump guards do most of the work, the blocked-view limit handles the jumps, and they add up: no single
guard reaches zero.

Per run (`pure/results/per_run_numbers.csv`): single clean whales (clips 03, 04, 06) keep ONE number with no ending
under strict; the group clip 01 keeps 6 numbers (2 endings); the bubble curtain goes from 12 to 23 numbers; the calf
breaths from 1-2 to 3-5 numbers (appearance drift near the surface, and the 4.2 s breath now exceeds the 4 s lone
limit, so the calf gets a new number after it). Fish tracks fragment heavily (clip 05: 6 -> 65 `sN small` numbers),
which costs nothing for whale numbers.

Still moving: the two joins in clip 01 (25.6-26.0 s, 29.8-30.4 s): one outline covers two whales for 0.4-0.6 s
and nothing in the tracker's outputs changes abruptly. No guard tested catches them.

Forward / backward (clips 01, 07 and the dev breath run backwards, `pure/fb.py`): flags 3 forward tracks in the dev
breath (including the bubble cloud -> whale #1), none in clip 01 or clip 07 (both directions make the same mistake
there). It doubles the GPU cost and caught nothing the cheaper guards miss; not shipped.

## C3. Sand (second deployment)

The labelling agent's `blocked_spans.csv` lists 10 sand spans, all in 00017 (2025-01-02, whale feeding near the
bottom at ~49 m), and none has a whale before and after, so no tracker comparison was possible. The blocked-view
rule on 102 sand frames and 10 open-water frames of 00017 (`pure/sand.py`, `results/sand_00017.csv`): it calls 78 %
of sand frames blocked, but also 60 % of open-water frames, both through the `air` class (the dev-fitted classifier
calls this deployment's murky water "air"); bubbles share stays at 0.11. With a sensor log (this deployment has one)
the air route is not used, so the rule does NOT see sand: it would need its own class or labelled frames.

## C4. Videos (strict setting; "N ended: reason" where a number was ended)

`review_sheets/underwater/demo_pure_01_comp_group.mp4` (clip 01: the two joins still there),
`demo_pure_02_calf_with_fish.mp4` (calf as 5 fragments, fish as `sN small`), `demo_pure_03_bubble_curtain.mp4`
(no outlines on blocked frames, 17 endings, 23 numbers), `demo_pure_04_breath_00007_1466s.mp4`,
`demo_pure_05_hazy_calf_fish_swap.mp4` (s1 on the fish ends as it reaches the calf; the calf goes on under a new number, still drawn small); `.jpg` sheets beside.

Rerun: `nohup pure/jobs/run_pure.sh` (GPU: backward runs, features), `pure/evaluate.py trade|merge|moved|fishstats`,
`pure/fb.py`, `pure/sand.py` (GPU, locked), `pure/render.py --level strict`.

# WP11a: the tracking stage online (`uwid tracks`, `src/uwid/trackstage.py`, `online/`)

## D1. What runs

`uwid tracks "<deployment>" [--preset sensitive] [--only <rec>:<t0>-<t1>] [--max-seconds 30] [--render]` reads the
scan's `intervals.json` at run time (hq spans of kept stretches with `advanced_ok`), and per span (resumable, one
done-marker per span, a new span starts only in the first minutes of a locked command, every locked command
`flock ... timeout 900 nice -n 10`):
1. patch pass (DINOv3 + WP5b classes every 3rd frame) -> blocked frames by the WP8b rule (lens out from the depth
   sensor via the window table's `sensor_t0`, else the `air` share; bubbles >= 50 % of the non-back patches);
2. blocked frames are cut out of the tracker's view;
3. starts = the scan's decision objects labelled `whale` (back filter, whole-frame rule, >= 300 px) on its detector
   frames, if no live track explains the box, if < 30 % of the box is bubbles / surface / air patches, and if the
   WP10 box classifier calls the box's row `whale` (start screens in that order);
4. the strict purity guards act online: the tracker object is removed from the SAM 3 session at the frame a guard
   fires (memory gone), the ending is recorded; a later start box may begin a new number. Fish-like first seconds
   (< 0.3 of the width, > 1.5 body lengths/s) get no whale number.
Output per span `underwater_cache/tracks/<deployment>/<recording>/<t0>-<t1>_online/` (masks.npz, tracks.parquet,
meta.json with endings / refused starts / blocked ranges / start boxes, emb.npz, patch.npz, summary.json), one
`index.csv` per deployment. Long stretches (whole encounters) run as overlapping 20-26 s pieces and are joined by
`trackstage.stitch` (an object of the next piece that overlaps an object of the previous piece by mean IoU >= 0.5 in
the shared second keeps its id). 4K: all boxes, sizes and the back box are in 1920 x 1080 tracker coordinates; the
4K frames are only scaled; no pixel constant broke.

## D2. Lone links (the user's easy cases), `tracks.LinkConfig` / `link_lone`

A separate layer: fragments keep their ids, a chain shows one number, every candidate is kept with its evidence
(`tr.links`: gap, breath, other animals, detector counts, size ratio, appearance distance, why). Link A -> B when
- the scene is lone around the break: no other whale-sized (>= 300 px) fragment at the same time as A or B or during
  the break, whose box is apart from A's / B's box (a touching outline is a part of the same animal); and no
  detector object unexplained by any track on >= 2 detector frames (1 s) in the window 2 s before A ends to 2 s after
  B starts (window: 4 detector frames each side; a second whale in a group scene shows for seconds);
- the break is <= 8 s, or <= 12 s when the depth sensor puts a breath in it (longest lens-out of 164 breaths on the
  two tags 9.7 s, plus froth after re-entry);
- checks (can veto, never trigger): size ratio <= 1.5 + 0.25 per second of break; appearance distance of masked
  DINOv3 features <= 0.30.
Plus the calf rule: a track under the size limit that is not fish-like and stays >= 1 s gets a whale number.
Set on the old stretches only and frozen 2026-11-23 01:30 (a duplicate-merge step was tried and switched off: on
the gap runs 1 of its 3 joins was two different whales):

| old stretches (after-the-fact strict + layer) | result |
|---|---|
| lone-whale gap runs (B3, B4, B6), truth = uncut run | 12 links, 12 right; tracks per whale 1.70 -> 1.30; 10 lone breaks left (6 too long, 4 not lone) |
| calf + fish gap runs (B5) | 5 links, 5 right; 2.00 -> 1.55 per whale |
| group gap runs (B1) | 0 links; 1.77 per whale either way |
| dev clip runs (clips 01, 02, curtain, breath; 6 runs) | 0 links |
| calf breaths 00006 (2 runs x 2) and 00007 (2 runs) | 00007: linked across the 4.4 s breath (both variants); 00006 1520 s: 4-5 links, 1 number for the calf; 00006 582 s: 2 links, 6 -> 4 numbers (surface fakes) |
| calf with fish (clip 05), after the fact | 9 -> 8 numbers: the calf's flipper track and fish tracks next to it make most breaks "not lone"; 1 break vetoed by size (2.38) |
| hazy calf (clip 07) | 0 links; the calf now gets a whale number at 832 s (calf rule), its earlier fragment stays `small` |

## D3. Held-out stretches (rule written before any tracker output: `online/select.py`)

8 stretches (4 from 2025-02-03 outside 00006 / 00007, 4 from CC-09-85) + 2 breath spans on CC-09-85 on purpose.
By eye (AI by-eye, one viewing of 1 s sheets): every held-out stretch turned out to hold ONE other whale; no
multi-whale held-out scene, so "no link in multi-whale scenes" is tested only on the old stretches.

| stretch | whales (eye) | numbers moved (eye) | whale numbers purity -> with links | outlined frames | endings | links (eye) | refused starts |
|---|---|---|---|---|---|---|---|
| 0203 00004 497-524 very near whale | 1 | 0 | 2 -> 2 (flipper tracked on its own, co-visible) | 787/802 | drift 1 | 0 | box classifier: fish 2 (a jack, right), clutter 1 |
| 0203 00008 56-76 whale diving | 1 | 0 | 2 -> 1 | 586/592 | drift 1 | 1 right | 0 |
| 0203 00005 332-348 near whale, sun | 1 | 0 | 2 -> 1 | 470/472 | drift 1 | 1 right | 0 |
| 0203 00011 1062-1078 whale overhead | 1 | 0 | 1 -> 1 | 472/472 | 0 | 0 | 0 |
| CC-09-85 00023 284-304 near whale, 42 m | 1 | 0 | 1 -> 1 | 578/578 | 0 | 0 | 0 |
| CC-09-85 00023 252-271 near whale, 42 m | 1 | 0 | 1 -> 1 | 557/563 | area jump 1 | 0 | 0 |
| CC-09-85 00024 83-101 whale alongside at the surface | 1 | 0 | 1 -> 1 | 533/533 | 0 | 0 | clutter 1 |
| CC-09-85 00011 499-506 (at the surface, blow) | 0 | - | 0 -> 0 | 0/143 | 0 | - | clutter 5 (box classifier on the same boxes: other 5/5) |
| breath: CC-09-85 00024 108-118, whale alongside | 1 | 0 | 2 -> 1 | 226/293 | shape jump 1 | 1 right (1.8 s) | 0 |
| breath: CC-09-85 00017 176-186, blow + mirrored back | 0 | - | 0 -> 0 | 0/218 | 0 | - | clutter 8 (box classifier: other 8/8) |

Totals: 0 numbers moved; 3 links, 3 right, 0 wrong, 0 lone breaks left; tracks per whale 1.50 -> 1.13 over the 8
stretches with a whale (12 -> 9 numbers for 8 whales). The box classifier withheld no number from a real whale (its 2
refusals were a jack crossing at 512-515 s). No real whale stayed without a whale number (0 `small` tracks on the
held-out stretches). Breaths: no number starts on the blow, the surface or the mirrored back (13 of 13 refused);
the outline drawn there is only the back's. Over-blocking: on 00024 108-118 the depth sensor's lens-out (with its
0.2 s pads) hides 57 frames where the whale alongside is clearly visible; the number is linked across it.

No longer held-out (2026-11-23 review): the blocked rule was changed after looking at 00024, so 00024 108-118 and
the whole 00024 encounter (75-164 s) are no longer held-out stretches for the blocked rule (or for anything that
depends on which frames are blocked, such as the links across the breath). The numbers above are the run before the
change; the re-run is in D7.

Patch classes on CC-09-85: the start screen (clutter share) and the blocked rule's bubble part rely on them; the
drift guard uses DINOv3 features but not the classes. On these spans the clutter screen refused all 13 blow / surface
starts and the box classifier (run afterwards on the same boxes) agreed 13/13; blocked frames there came from depth
(58 / 57 / 70 frames) with few bubble calls (27 / 0 / 3).

## D4. Online against after the fact (same start boxes, WP5a)

| stretch | numbers online / after | endings online / after | online numbers that map to ONE after-the-fact number | outlined frames online / after |
|---|---|---|---|---|
| dev clip 01 | 5 / 6 | 2 / 2 | 5/5 | 359 / 359 |
| clip 05 calf + fish | 6 / 9 | 17 / 8 (10 shape jumps, mostly on fish) | 3/5 | 568 / 587 |

They agree on the group clip; on the calf clip the online run keeps the calf longer (the after-the-fact cut its
masks into 5 pieces, online the calf is number 3 from 459.5 s to the end once linked) but ends more fish tracks.
Online, blocked frames are cut out (after the fact they were only not drawn), so the two are not frame-identical.

## D5. Cost (online, every frame, shared RTX 4000 Ada, model loading not counted)

| | video s | GPU s per video s | peak GPU |
|---|---|---|---|
| 2025-02-03 held-out (1080p HEVC) | 78 | 9.7 (7.5-12.9) | 1.9 GB |
| CC-09-85 held-out + breaths (4K, scaled) | 78 | 8.8 (5.0-10.7; breaths 5-7: fewer frames shown) | 1.9 GB |
| old stretches (WP5a starts) | 32 | 8.4-11.7 | 2.0-2.3 GB |

Estimate (not run; needs the user's go-ahead): 2025-02-03 hq spans 758 s -> about 2.1 GPU h (all 1,402 s of its hq
stretches: 3.9 h); CC-09-85 hq spans 189 s -> 0.5 h (all 352 s of hq stretches: 0.9 h). The scan's new clip rule
(WP11b, 30 s quiet before a clip ends) makes stretches longer; hq spans keep their cut, so the hq-span figures hold.

## D6. Videos (`review_sheets/underwater/`, H.264 yuv420p, no sound, `.jpg` sheet beside each)

- `demo_online_01_cats85_00024_whole_encounter_near_whale_breath` (whole encounter 75-164 s of 00024, four stitched
  pieces, 86 s, 4K; 26.2 MB; re-rendered 2026-11-23 04:11 after the D7 fixes): the whale alongside is number 1 from
  77.7 s to 119.0 s (one lone link across a piece boundary at 98 s, gap 0.27 s; no frame blocked before 119.0 s);
  blocked 119.0-120.0 s only (lens-out where the picture says crossing); after the breath numbers 2 and 3 (2 ends by
  appearance drift at 126.3 s); 4 -> 3 numbers. Not linked across the breath (D7 C). Clock not read on 4K frames ("--").
  Before the fix (03:36 render): blocked 109.4-110.0, 115.3-115.9, 117.3-120.0 s, number 2 at 117 s, 5 -> 4 numbers.
- `demo_online_02_0203_00008_whole_encounter` (whole encounter 25-103 s of 00008, four stitched pieces, 75 s): one
  whale from its first appearance to the last frame, one number (one lone link at 58 s, gap 0.2 s).
- `demo_online_03_cats85_00023_near_whale_42m` (held-out, one number), `demo_online_04_cats85_00017_breath_blow_no_number`
  (breath: no number starts, only the back is outlined), `demo_online_05_0203_00004_very_near_whale` (held-out; the
  whale's flipper is a second number for 3 s).
- `demo_link_01_calf_with_fish` (the online run of clip 05 + links: the calf is number 1 for 3 s, then number 3 to the
  end through 2 lone links; fish `sN small`), `demo_link_02_breath_00007_1466s` ("1 linked: lone, gap 4.4 s" across
  the breath; after the fact), `demo_link_03_hazy_calf_fish` (no link; the calf gets a whale number at 832 s by the
  calf rule, its first fragment stays `small`; after the fact).

## D7. Review follow-ups (2026-11-23; `online/run_followup.py`, `jobs/run_followup.sh`, one locked GPU job 03:57-04:06)

**A. Blocked rule, as it is now.** A frame is blocked when (i) bubbles cover >= 50 % of the non-back patches, or
(ii) the lens is out of the water: with a depth log, the sensor's lens-out (surfacing exit + 0.3 s to entry - 0.4 s,
0.2 s pads) AND the scan's picture water model for that 1 s window (`windows/<recording>.csv`, column
`water_picture`) is not `underwater` (i.e. `air` or `crossing`); a window the picture model has no row for keeps
the sensor decision; without a depth log, air share >= 25 % of patches as before. Runs < 0.2 s are dropped, gaps
< 0.3 s filled (unchanged). No new level was fitted: the rule uses the scan's own per-window label. Before running
it was checked on the old breath stretches only (2025-02-03 00007 1465-1468 s: crossing, air, air, crossing;
00006 581-584 s: crossing, air, air, crossing; both stay blocked); 00024 was not used to set it (its windows 107-118 s
are all `underwater`, 119-120 crossing, 121-123 air, 124 crossing).

| run (online, scan starts) | blocked frames before -> after | outlined frames before -> after | whale numbers before -> after | endings before -> after |
|---|---|---|---|---|
| 00024 97-120 (piece) | 117 -> 30 (109.4-110.0, 115.3-115.9, 117.3-120.0 -> 119.0-120.0 s) | 526 -> 623 of 653 | 2 -> 1 | shape jump 115.9 s -> none |
| 00024 108-118 (held-out breath span) | 57 -> 0 | 226 -> 293 of 293 | 2 -> 1 (link no longer needed) | shape jump 115.9 s -> none |
| 00024 119-142 (piece; starts 125.2 s, after the breath) | 0 -> 0 | 229 -> 229 of 503 | 2 -> 2 | drift 126.3 s (same) |
| 2025-02-03 00007 1459-1475 (breath, new run on this route) | 126 (87 lens-out, all 87 confirmed by the picture, + bubbles; 1465.7-1469.9 s) | 238 of 472 | 2 -> 1 with the link (gap 5.84 s) | area jump 1465.4 s |
| whole 00024 encounter 75-164 (stitched) | 117 -> 30 | - | purity 5 -> 4 numbers, with links 4 -> 3 | - |

00007: there is no earlier run on the scan route, so its "before" is the same patch pass under the old rule: 126
frames as well (the picture agreed on every lens-out frame). The real lens-out stays blocked. The gap was 4.4 s in
WP8c / `demo_link_02` (after-the-fact run from WP5a boxes, gap/ rule from surfacing pts): 1465.14 -> 1469.5 s. It is
5.84 s now (1465.40 -> 1471.24 s) because this run starts numbers only from the scan's whale decision objects, and
the first one after the breath is at 1471.24 s; the block itself ends at 1469.9 s (bubbles after the re-entry).

**B. Link missing in the stitched encounter (fault, fixed).** Not a stitching artefact as such: the lone test counts
detector boxes that no tracker outline explains in the 2 s before and after a break. On blocked frames the tracker
draws no outline, so every detector box there was "unexplained". In the span run 108-118 the window after the break
was cut at 118 s; in the stitched run it reached into the blocked 117.3-120.0 s and found 2 such frames -> "not
lone". Fix (`tracks.link_lone`): blocked / hidden frames are left out of the before and after counts, as they already
were for "during". Test: `tests/test_tracks.py::test_lone_link_ignores_detector_on_blocked_frames` (detector box on
10 blocked frames after a break: linked; same boxes unblocked: not linked). On the old pieces the stitched encounter
then linked 1 -> 1 across 115.3 s (gap 0.97 s), as the span run did. After fix A that break no longer exists.

**C. Across the breath in 00024 (report only, no rule change).** Every candidate the link layer judged, made or not,
is now written to `links.csv` (columns a, b, t_end, gap_s, breath, limit_s, others, det_before / during / after,
lone, size_ratio, app_dist, checks_ok, made, why, num_a, num_b): per span folder, per encounter
(`<recording>/<t0>-<t1>_encounter/links.csv`) and per deployment beside `index.csv` (`<deployment>/links.csv`). Before
this only the first candidate was kept when none was made. Candidates from number 1 (ends 118.97 s) in the stitched
encounter:

| candidate | gap | breath limit | others in view | detector before / during / after | size ratio (limit 1.5 + 0.25 / s = 3.07) | appearance distance (limit 0.30) | checks | made | why |
|---|---|---|---|---|---|---|---|---|---|
| 1 -> 2 | 6.27 s | 12 s | 3 | 0 / 0 / 0 | 2.40 | 0.353 | fail (appearance) | no | not lone |
| 1 -> 3 | 6.27 s | 12 s | 2 | 0 / 0 / 0 | 1.39 | 0.249 | pass | no | not lone |

Both are refused only because 2 and 3 are in view at the same time after the breath. A rule change that WOULD link
1 -> 3 (for the user to decide; not made): across a breath, when several fragments start after it, link to the one
candidate whose size and appearance checks pass if every other candidate fails them. By eye I cannot tell whether 2
or 3 (or both) is the whale alongside.

**Other spans.** Fix B and the "all candidates" change were applied to every stored span (links.csv back-filled on
CPU, no re-run): the made links are unchanged everywhere: 00008 encounter 2 -> 1 numbers, one link at 58.02 s (gap
0.23 s), as before; 00005 332-348 one link; 00006 456-476 (old) 2 links; no new link and none lost. Only the four
re-run spans' summaries changed. Old runs kept for the comparison: `<span>_online_pre_fixA` (3 folders, 4.2 MB; not
read by `write_index`).


## D8. WP11c: held-out lone-whale links across breaths + a look at groups (rules frozen; `online/select_breath.py`, `report_breath.py`)

**Selection** (scan window tables + depth sensor only, written before any tracker run; `results/selection_breath.csv`,
skips with reasons in `results/selection_breath_skipped.csv`). 112 lens-out runs of 1-12 s by the frozen blocked rule
(depth lens-out AND picture `air` / `crossing`) on the two test deployments. 24 have an active whale window within the
12 s after. 11 also have an hq / ok whale window within the 5 s before. 8 of those 11 overlap a left-out stretch
(00005 x3, 00024 x2, 00006 581 s, 00007 1465 s, 00011 1080 s), leaving **3 breath encounters, all on 2025-02-03, none
on CC-09-85** (CC-09-85: 32 of its breaths have no hq / ok whale window in the 5 s before). The coordinator's first
rule also asked for at most one scan whale object (>= 300 px) in every window of +-10 s: 0 of the 11 passed, including
the known lone case 00007 1465 s (2 windows with 2 objects), because the scan often puts two whale boxes on one near
whale (and in 00005 607 s both "whales" are the tagged whale's own head and flipper, below). That condition was
dropped by the coordinator before any tracker run; the count is kept as `multi_windows` (1, 1, 2 for the three).
Groups: runs of >= 3 s with >= 2 scan whale objects in a window: 19 runs; the 2 longest from different recordings,
not left out: 00004 291 s (29 s run) and 00005 612 s (8 s run). GPU: one job per set, 1,025 s of tracking in total
(8 pieces; lock held 13:12-13:20 and 13:20-13:32), <= 2.0 GB.

**By eye first** (AI by-eye, plain 1 s frames, no outlines, written to `results/breath_eye.csv` before the tracker
output was looked at): all 3 breath encounters show ONE whale before and ONE after (00004: after only at 623 s at the
left edge; 00005: faint pale shape 1566-1569 s and a whale at the bottom right 1576-1579 s; 00006: only 912-913 s at
the left edge during the bubbles). So all 3 are lone encounters; none goes under groups.

**Breath encounters (lone).** Tool's decision; numbers by eye on 1 s sheets with outlines + zoomed frames (AI by-eye).

| encounter (run) | breath (blocked) | numbers on the whale | links made (gap, size ratio, appearance) | links refused (why, kind) | number on blow / surface | moved between animals | whale on ONE number |
|---|---|---|---|---|---|---|---|
| 00004 608-637.1 | 618.0-622.5 | 1 (609.7-618.0, then 622.8-623.7) | 1 -> 1, 4.77 s, 1.10, 0.243, lone (detector 1 / 0 / 0) | - | 0 | 0 | yes |
| 00005 1551.8-1579.5 (2 lens-outs) | 1561.3-1564.8, 1571.4-1574.9 | 1 (1552.3-1558.9, 1566.3-1569.2), 2 (1575.8-1579.5) | 1 -> 1 across the 1st breath, 7.41 s, 1.82, 0.248 (the fragment after it is a pale mottled patch at the bottom right: by eye probably the whale's body, not certain) | 1 -> 2 across the 2nd lens-out: 6.61 s, lone, size 2.02 (limit 3.15), appearance 0.364 > 0.30: "checks contradict", kind (c) | 0 | 0 | no (1 -> 2) |
| 00006 899-927 | 909.0-912.0 | 1 (899.3-908.8, ended by appearance drift 0.2 s before the block), 2 (912.8-913.0 on the whale leaving at the top-left corner) | - | 1 -> 2: 3.90 s, lone, size 1.41, appearance 0.358 > 0.30: "checks contradict", kind (c) | 1: number 2 stays on a sliver of the surface at the top edge 913.3-916.2 s after the whale has left | 0 | no (1 -> 2) |

Totals (3 lone encounters, 4 breaths / lens-outs crossed): 1 of 3 encounters ends with the whale on ONE number; 2
links made (both lone, both by eye consistent with one whale; the 00005 one uncertain), 2 refused, both kind (c), both
on the appearance check only (0.358, 0.364 against 0.30; size ratios passed); kinds (a), (b), (d): 0. 0 numbers moved
between animals; 1 number left on the surface for 2.9 s (00006). The whale was visible without a number for ~1.5 s at
the start of 00004 (the run starts at the first scan start box, 609.7 s) and for ~0.4 s after the block in 00006.

**Groups.**

| stretch | animals (eye, plain frames) | numbers per animal (eye, with outlines) | numbers moved | links made | refused starts |
|---|---|---|---|---|---|
| 00004 286-306 | 2 whales (A right / bottom, B top-left from 291 s) + a slender animal above A at 300-302 s (far whale or large fish: cannot tell); many fish | A: 1 (286.3-301.8) then 4 (302.2-306.0); B: 2 (291.3-304.7) then 5 (305.2-306.0; its mask also holds a fish in the last 0.7 s); slender animal: s1 then 3 (301.2-301.5, the outline also takes A's flipper) | 0 | 0 (5 candidates, all "not lone") | 10 (box classifier: fish 9, other 1) |
| 00005 607-627 | plain frames: I wrote "1 whale far below + a dark fin at the left edge (probably the tagged whale's flipper)"; with the tool's back outline it is the tagged whale's own head / hump and flipper: 0 free whales | 0 (both tracked objects are >= 99.8 % inside the back: role `on_back`) | 0 | 0 | 0 |

The scan's two "whale" objects in 00005 607 s are both the tagged whale's own body; in 00004 291 s they are two real
whales. My plain-frame count for 00005 607 s was wrong (corrected after seeing the back outline).

**Videos** (rule: in selection order, the first two breath encounters with a made link and the first two without;
only one has none): `review_sheets/underwater/demo_breath_01_linked_00004_608s`, `demo_breath_02_linked_00005_1552s`,
`demo_breath_03_notlinked_00006_899s` (.mp4 + .jpg).

Caveats: 3 encounters is a small test, all from one deployment and one tag; the "after" side is short in two of them
(1 s of whale in 00004, 0.3 s in 00006); the by-eye calls are one viewing of 1 s sheets; nothing was changed or tuned
on these stretches.

## D9. WP13: the tracker on all hq spans of both test deployments (rules frozen as of WP11c)

**Spans.** Exactly what `uwid tracks <deployment>` runs: every hq span of a kept stretch with `advanced_ok` in the
scan's `sensitive` intervals.json (read at run time, after the WP11b scan changes), capped at 30 s around its centre
(none is longer); each run starts at the span's first scan start-box frame. CC-09-85: 17 spans, 147 s (3
recordings); 2025-02-03: 118 spans, 675 s (9 recordings). Most spans are short (2025-02-03: median 4 s, 1-27 s; CC-09-85: median 7 s, 1-20 s).
Reused: 00024 108-118 (run 2026-11-23 04:01 with the current blocked rule). Seven spans run before the 12:31 rule
change (WP11a held-out names 00023 252-271 / 284-304, 00024 83-101, 00004 497-524, 00005 332-348, 00008 56-76,
00011 1062-1078) were moved aside as `*_pre_wp13` and run again.

**How it ran** (CPU overload 21:00-01:00, load 47-60 on 37 cores, other project at nice 0; mine at nice 10):
- v1 (`uwid tracks`, 21:50-22:05, one piece): 3 short CC-09-85 spans finished (00017 290-291, 294-296, 00023
  39-46, decoded), then cut off at 900 s inside 00023 120-138.
- v2 (`online/wp13_queue.py` pieces under the lock, `jobs/run_wp13.sh`, 22:26-01:28): same spans and same
  `trackstage.run_span`; queue 1080p shortest first, then 4K; per-span estimate from the last measured rate; a cut
  span never re-run by the same path. Frame store on (`trackstage` `frame_store` flag, off by default, new test
  `test_frame_store_gives_the_decoded_frames`): `online/wp13_predecode.py` decoded upcoming spans without the lock
  into `tracks/tmp_frames/` (<= 3 GB, deleted after each span). Store check before use (`online/wp13_store_check.py`):
  00017 290-291 (4K, 23 frames) run again from the store: masks 0 of 23 frames differ, table and scores identical,
  patch labels, blocked frames, endings, refusals and numbers identical; tracking 19.5 -> 6.7 s. (No 1080p span
  was compared.) Paths: 2025-02-03 95 spans from the store, 23 decoded (store not ready or > 3 GB); CC-09-85 3 from
  the store, 13 decoded (incl. v1's 3), 1 reused. 00023 120-138 (cut at 22:05) ran once more, decoded, at 01:29
  when the load had dropped to 6-24 (188 s).
- Lock held: 8,088 s (2.25 h) by the driver, plus the store-check piece (not timed, < 10 min). Tracks cache 223 MB;
  `/home` 179-197 GB free throughout.

| deployment | spans | video s | numbers (before -> after links) | numbers >= 3 s | track-seconds | blocked frames | links made | links not made (why) | starts refused (reason) | GPU tracking min | GPU min per span-min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CC-09-85 | 17 | 142.1 | 19 -> 18 | 11 | 142.8 | 0 | 1 | 0 | surface / bubbles / air 1 | 22.4 | 9.45 |
| 2025-02-03 | 118 | 643.7 | 138 -> 131 | 68 | 635.4 | 32 | 7 | not lone 11, checks contradict 3 | surface / bubbles / air 47, box classifier fish 11, other 2 | 83.2 | 7.75 |

(video s = frames tracked, from each span's first start-box frame; GPU tracking minutes = the tracker's own
`seconds_total`, without model loading, patch pass and waiting.) Numbers per span: 2025-02-03 0: 2 spans, 1: 106, 2:
7, 3: 1, 4: 2; CC-09-85 1: 16, 2: 1. All 8 links made are short breaks (0.10-0.53 s, none across a breath; hq spans
hold almost no lens-out). Bubble suspects (a number starting <= 10 s after a lens-out run): 9, all on 2025-02-03.
Calf suspects (own-back role, >= 3 s): 0 - no object in any of the 135 spans got the own-back role at all; the
00005 604-629 s calf stretch is not an hq span, so it was not tracked.

**Review pack** (`review_sheets/underwater/`, tool's decision only; rules in `online/wp13_report.py`):
`wp13_a_sample_01..04.jpg` (24 numbers >= 3 s: CC-09-85 has only 11, all taken, 2025-02-03 13; seed 20261123),
`wp13_b_links_01..02.jpg` (all 8 made links), `wp13_c_bubble_suspects_01.jpg` (9), no `d` sheet (0 calf
suspects). Videos (the six longest numbers, one per span, whole span, 9.2 MB together):
`demo_wp13_01_2025-02-03_00004_498s_n1`, `_02_2025-02-03_00004_649s_n1`, `_03_2025-02-03_00008_56s_n1`,
`_04_CC-09-85_00023_285s_n1`, `_05_CC-09-85_00023_252s_n1`, `_06_CC-09-85_00023_120s_n1` (.mp4 + .jpg; lives
17.7-26.2 s). Lists in the cache: `tracks/wp13_{spans,numbers,links,sample_a,bubble_suspects,calf_suspects}.csv`.
The deployment-level `tracks/<deployment>/links.csv` now also holds the encounter rows (00024 75-164: 3; 2025-02-03
encounters 00004 608, 00005 1552, 00006 899, 00008 25-103: 5).

**Reading a number's outlines from the cache (for the 3D work).** One folder per span:
`underwater_cache/tracks/<deployment>/<recording stem>/<t0>-<t1>_online/` (t0, t1 = the span in video seconds;
`<deployment>/index.csv` lists them). `meta.json`: `video` (source path), `frames` = [f0, f1) absolute frame
indices (shown frames of `uwid.video.frame_table(video)`; `t0` = time of f0 in the video's own presentation seconds,
the clock of the scan windows), `fps`, `store_size` = [1280, 720], `blocked` = [[a, b), ...] absolute frame ranges
not shown to the tracker (lens out where the picture agrees, bubbles; no outline there), `back_box`. `masks.npz`:
`obj_<k>` = uint8 array (n_frames, 720, 160), each row `np.packbits` of a 1280-px row (unpack with
`np.unpackbits(a, axis=1)[:, :1280]`); frame j = absolute frame f0 + j. `obj_0` is the tagged whale's own back
(prompted with the bottom 25 % box). `tracks.parquet`: one row per object and frame (i = j, frame, t = video seconds,
obj, score, area, box x0 y0 x1 y1 in 1920 x 1080 coordinates for any video size). Raw object ids are not the shown
numbers: `uwid.trackstage.load_online(folder)` then `trackstage.link(tr, deployment, "scan")` gives `tr.number`
(obj -> number after cleaning and lone links), `tr.mask(obj, j)` (bool 720 x 1280) and `tr.visible(obj, j)`;
`links.csv` beside it has every link candidate with its evidence. Masks are at 1280 x 720 for every source (4K
sources are scaled), so scale x by width / 1280 and y by height / 720 to go back to source pixels.
