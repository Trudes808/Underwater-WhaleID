# 10_mesh3d (WP15): single-image 3D meshes of whales from tracked tag-video frames

Nothing here is ground truth: there is no measured 3D shape of any of these whales. Every number below is either a
self-consistency number (mesh against mesh, mesh against the input outline) or a comparison with published humpback
proportions.

## S0. Frame rule for the smoke test (written 2026-11-23 21:00, before any mesh was made)

Pool: frames of whale numbers in the existing online tracker runs (`uwid tracks` output, read with
`uwid.trackstage.load_online`) of 2025-02-03 pieces 00004-00007, CC-09-85 piece 00024 and the dev clip
`2025-12-20 CG` (run `0024.00-0036.00_online`). Folders whose name ends in `_pre_*` are old copies and are not used.
Every 5th frame is looked at (6 per second).

A frame of number N passes when all of these hold (outline at the stored 1280 x 720 size):

1. role `whale` with a display number, not drawn `small` on that frame, not a hidden duplicate;
2. not a blocked frame (meta `blocked` ranges);
3. large: outline area >= 3 % of the frame (27,648 px);
4. whole animal inside the frame: the outline's box is at least 6 px from every frame edge;
5. clear of the tagged whale's back: no outline pixel within 12 px of the back outline (object 0) on that frame,
   and the outline's lowest pixel above the top of the run's fixed back box;
6. one piece: the largest connected part holds >= 90 % of the outline;
7. sharp: variance of the Laplacian of the grey frame inside the outline (eroded 3 px) at least the median of that
   number's passing frames.

Pick (AI by-eye, on a contact sheet of the best-scoring passing frame per number and second): one frame each for a
side view, a view from below, a view from behind, the calf (if a calf passes the rule), and the comp-group dev clip;
then, for two of these numbers, two more passing frames of the same number 1-3 s away (consistency test). The
viewpoint words are AI by-eye labels.

Input to the 3D model: the decoded frame (1920 x 1080) with the tracker's outline (scaled up, nearest neighbour) as
the alpha channel; nothing else.

How many frames the rule leaves (`select_smoke.py`, 16 online runs, every 5th frame): 84 frames of whale numbers pass
the table part (area, edges, back box, not blocked), 83 the outline part (1 near the back), 43 the sharpness half.
They come from 5 runs and 5 numbers (00004 609-612 s n1: 7; 00005 1552-1553 s n1: 3; 00006 899-905 s n1: 19;
00024 111.7-112.7 s n1: 6; dev clip n1 / n2 / n3: 1 / 2 / 5). No frame of the calf passes: in the calf run
(00006 456-476 s) the large frames reach below the fixed back box or touch the back outline, so the calf frame
below is a stated exception (it fails only clause 5b, the fixed back box; the back outline is not touched).

## S1. Can SAM 3D Objects run here? (checked 2026-11-23 20:40-21:00, official sources only)

| item | finding | source |
|---|---|---|
| access | `facebook/sam-3d-objects` is gated (manual approval). The account logged in on this machine has no access: README and LICENSE download, every checkpoint file returns 403 GatedRepoError. | `hf_hub_download` with the repo `.venv` |
| how to get it | the user requests access on https://huggingface.co/facebook/sam-3d-objects (form: full legal name, date of birth, country, affiliation, job title; the card asks for the full organisation name without acronyms; it cannot be edited after submission) | model card |
| licence | SAM License (19 Nov 2025): non-exclusive, worldwide, royalty-free use / modify / redistribute; publications must acknowledge SAM Materials; no ITAR / military / weapons use; redistribution only with the licence | `LICENSE` in the HF repo |
| input / output | RGBA image with the object's mask in the alpha channel (or image + mask); output a Gaussian splat (`.ply`), optionally a mesh (`.glb`, texture baking), with pose and layout | README, issue #54 / #75 |
| install | conda/mamba env (CUDA 12.1 toolkit, gcc 12), torch 2.5.1 cu121, pytorch3d, kaolin, spconv, xformers, MoGe, Hydra patch | `doc/setup.md`, `environments/default.yml`, `requirements*.txt` |
| stated GPU need | "A NVIDIA GPU with at least 32 Gb of VRAM"; maintainer: "The smallest GPU we've been able to run the model reliably is a V100" | `doc/setup.md`; issue #6 (maintainer gleize) |
| low-memory options (all from users, not tested by Meta) | `downsample_ss_dist: 2-4` in pipeline.yaml (contributor, "results not guaranteed"); skip the mesh decoder (`decode_formats=["gaussian"]`); one user reports a 16 GB card working, another OOM on 16 GB and 24 GB; 12 GB took about 1 hour; splitting sub-models over 2 GPUs works | issues #6, #30, #68, #129 |
| weight sizes | 13.2 GB total; `ss_generator.ckpt` 6.69 GB, `slat_generator.ckpt` 4.91 GB, decoders 0.15-0.36 GB each | HF file list (sizes are public) |
| here | Not run: no access. With a 9 GB cap it would need code changes (half precision, each stage moved on and off the GPU, skipped mesh decoder) that nobody reports working under 16 GB. Not measured. | |

Stand-in chosen: TripoSG (VAST-AI, MIT licence for code and weights, 1.5B rectified-flow model, image -> mesh; README
asks for >= 8 GB). Others looked at: TRELLIS (MIT, README needs 16 GB), TRELLIS.2 (MIT, 24 GB), Hunyuan3D-2 /
2mini (Tencent community licence with territory limits, 6 GB for shape), TripoSR (MIT, 6 GB, older and coarser).

## S2. Set-up

| item | value |
|---|---|
| env | `underwater_cache/envs/triposg` (virtualenv, Python 3.12, torch 2.6.0 cu124, diffusers 0.32.2, transformers 4.49.0), 5.7 GB; script `jobs/install_triposg_env.sh` |
| code | `underwater_cache/third_party/TripoSG` at commit fc5c409 (2025-04-18), 34 MB, unmodified |
| weights | `underwater_cache/weights/TripoSG` from `VAST-AI/TripoSG`, 7.5 GB (fp32 files, run in fp16) |
| not used | RMBG-1.4 background remover (the tracker outline is the alpha); the flash decoder (needs the `diso` CUDA extension, no nvcc here): the hierarchical decoder + skimage marching cubes at 512^3 is used instead |
| run | `jobs/run_smoke_meshes.sh`: pieces under `flock ... timeout 900 nice -n 10`, at most ~4 min each, lock given back 30 s between pieces; torch capped at 9 GB per process |

## S3. Smoke test (10 frames, TripoSG, seed 42, 50 steps, guidance 7; 2026-11-23 22:05-22:27)

No ground truth. "By eye" = AI by-eye (one viewing of each sheet by the agent). IoU = outline of the mesh seen from
the best of the 24 axis-aligned views (it was the model's +z view for all 10) against the tracker outline, both
cropped and padded alike: it says the mesh agrees with its own input, not that the shape is right.

| id | source (number) | viewpoint (AI by-eye) | s / mesh | torch peak GB | nvidia-smi MiB | IoU | by eye: reads as humpback? what was invented |
|---|---|---|---|---|---|---|---|
| 01_side | CC-09-85 00024 112.4 s (1) | flank, near, fluke and flipper outside the outline | 31.5 | 5.21 | 5602 | 0.92 | No: a smooth potato-shaped blob; the outline itself holds no fins |
| 02_below | 2025-02-03 00005 1552.2 s (1) | from below, flippers out, glare | 32.7 | 5.22 | 5622 | 0.88 | Whale yes (fluke, two flippers, taper); flippers shorter than a humpback's; dorsal side invented with a small fin |
| 03_behind_a | 2025-02-03 00006 899.9 s (1) | from behind / below, fluke toward camera | 36.8 | 5.19 | 5582 | 0.93 | No: the foreshortened outline is lifted into a long thin "fluke on a stalk"; no flippers, no head |
| 04_behind_b | same number, +1.3 s | same | 31.5 | 5.21 | 5602 | 0.61 | No: a fish tail with scale-like ridges |
| 05_behind_c | same number, +2.8 s | same | 32.2 | 5.21 | 5602 | 0.85 | No: T-shaped fluke + stalk with a stub |
| 06_compgroup | dev 28.2 s (2) | comp group, flank / belly, a raised blade inside the outline | 39.4 | 5.19 | 5582 | 0.52 | Whale-shaped but dolphin-like: short flippers, the raised blade became the fluke |
| 07_dev_below_a | dev 33.4 s (3) | from below, flippers spread | 30.9 | 5.19 | 5582 | 0.84 | Yes: long flippers, body taper, fluke (one flat blade); back and dorsal hump invented |
| 08_dev_below_b | same number, +1.3 s | same | 30.0 | 5.18 | 5582 | 0.85 | Yes: as 07, flippers shorter, two-lobed fluke |
| 09_dev_below_c | same number, +2.5 s | same | 35.2 | 5.20 | 5602 | 0.87 | Yes: the most humpback-like (long flippers, two-lobed fluke, knobbly head) |
| 10_calf | 2025-02-03 00006 469.8 s (6), rule exception | near silhouette from below | 30.3 | 5.19 | 5582 | 0.92 | No: a giant two-lobed "fluke" on a small round blob |

Summary: 4 of 10 meshes read as a whale (02, 07, 08, 09), 3 as a humpback (07-09: all the same animal, flippers
spread, seen from straight below, the easiest view); 06 is whale-shaped but not a humpback. The 3 behind views, the
flank view and the calf silhouette gave shapes that are not whales. Time 30-39 s per mesh (median 32), model load
1.3-3.5 s warm (134 s cold, plus about 70 s of imports on the first piece). Peak memory 5.18-5.22 GB (torch), 5.6 GB
(nvidia-smi, this process), inside the 9 GB cap.

Consistency (`mesh_tools.aligned_distance`: 6,000 surface points per mesh, centred and scaled to unit RMS radius,
best of 24 PCA-axis starts + rigid ICP, no mirror, symmetric chamfer in RMS-radius units; the same mesh resampled
and rotated gives 0.02):

| pair kind | pairs | mean | min - max |
|---|---|---|---|
| same number, 1.2-2.8 s apart | 6 | 0.104 | 0.087 - 0.117 |
| other number, same deployment (animal unknown) | 10 | 0.124 | 0.084 - 0.195 |
| different animal (other deployment) | 29 | 0.169 | 0.084 - 0.285 |
| different animal, both meshes whale-like (02, 06, 07, 08, 09) | 4 | 0.102 | 0.084 - 0.133 |

7 of 29 different-animal pairs are as close as the farthest same-number pair (AUC 0.84 over 6 x 29). The separation
comes from the failed meshes (01 blob, 10 calf); whale-like meshes of different animals in different deployments
(02 against 07: 0.084) are as close as frames of one animal. The distance measures "whale-shaped or not", not the
individual.

Proportions (`mesh_tools.proportions`, PCA frame; flipper estimate = (largest lateral span of a length slab -
largest depth) / 2, which also counts the fluke when the fluke is the widest part):

| id | length / depth | lateral span / length | flipper estimate / length |
|---|---|---|---|
| 02_below | 3.94 | 0.72 | 0.23 |
| 07_dev_below_a | 3.23 | 0.93 | 0.31 |
| 08_dev_below_b | 5.34 | 0.61 | 0.21 |
| 09_dev_below_c | 2.95 | 0.68 | 0.17 |
| published | - | - | flippers "up to one-third of the body length" (Jefferson, Leatherwood & Webber 1993, FAO Species Identification Guide, Marine Mammals of the World, p. 60) |

The three meshes of one animal give 0.17-0.31 for the flipper and 2.95-5.34 for length / depth, so one frame does not
fix these proportions. I found no verified published length / depth (fineness) ratio for humpbacks in the time
taken (Woodward, Winn & Fish 2006, J. Morphol. 267: 1284-1294 has the body measurements, full text not read).

Outputs: meshes `underwater_cache/mesh3d/smoke/meshes/*.ply` (+ JSON per mesh), inputs `.../smoke/inputs/`, picks
`.../smoke/picks.csv`; sheets `review_sheets/underwater/mesh3d_smoke_01_side.jpg` ... `_10_calf.jpg`; numbers
`results/smoke_metrics.json`.

## S4. Phase 2 (prepared, not run)

`pick_frames.py "<deployment>"` applies the S0 rule to every finished online run, keeps frames inside the scan's hq
spans, and per number takes up to 4 frames spread over outline shape (orientation, elongation, size, fill; >= 1 s
apart; sharpest first). Written `picks.csv` feeds `prep_inputs.py` -> `run_triposg.py` (in locked pieces) ->
`smoke_report.py` (sheets + metrics; its all-pairs alignment costs about 20 s CPU per pair, so for many meshes only
pairs within a number should be run). Tried on CC-09-85 while WP13 was still running: 2 picks from the runs done so
far. Cost per 100 frames: about 53 GPU min of generation (32 s each) in about 13-15 locked pieces (+ 2-4 s load per
piece when the weights are in the page cache, about 3.5 min for the first cold piece), plus CPU: picking took 14 min for the 16 smoke runs
(decoding is the cost), sheets about 10 s per mesh.

## S5. Phase 2: the trial on WP13's tracks (both test deployments; started 2026-11-24 01:47)

### S5.1 Picks (`pick_frames.py`, fixed before any mesh; `underwater_cache/mesh3d/trial/{judged,picks}.csv`, `pick_counts.json`)

Input: the 135 span folders of `tracks/wp13_spans.csv` only; shown numbers after `load_online` + `link` as in
`wp13_numbers.csv`; numbers lasting >= 3 s: 68 (2025-02-03) + 11 (CC-09-85) = 79. Frame rule S0 unchanged, every 5th
outlined frame: 4,158 frames judged. The two numbers the reviewer named (00005 365-367 number 1, life 1.70 s; 00005
375-389 number 2, life 0.93 s) are below 3 s, so the life rule drops them; they are also excluded by name.

First failing clause, in rule order (frames):

| clause | 2025-02-03 | CC-09-85 |
|---|---|---|
| drawn small | 5 | 0 |
| area < 3 % | 115 | 11 |
| at the frame edge (< 6 px) | 2,453 | 738 |
| below the fixed back-box top | 600 | 30 |
| within 12 px of the back outline | 0 | 7 |
| outside an hq span | 0 | 0 |
| sharpness below the number's median | 90 | 5 |
| passed | 98 | 6 |
| total | 3,361 | 797 |

(hidden duplicate, blocked, split: 0. Blocked frames carry no outline, so they never enter.)

Numbers by passing frames (4 = 4 or more): 2025-02-03 (68): 0: 56, 1: 2, 2: 1, 3: 2, 4: 7. CC-09-85 (11): 0: 10,
4: 1. So 13 of 79 numbers have any passing frame.

Picks: limit 4 per number was never reached by the cap (240): 28 picks from 13 numbers. Picks per number (frames
>= 1 s apart): 2025-02-03 1 pick: 5 numbers, 2: 3, 3: 1, 4: 3; CC-09-85 2: 1. Overlap with the smoke test: 2025-02-03
00006 898-909 number 1 is the smoke test's 00006 899 animal (4 picks at 899.6-904.1 s, other frames than the smoke
test's) and CC-09-85 00024 number 1 is the smoke test's 01_side animal (111.7 and 112.7 s). The rule mostly removes
near whales: 3,191 of 4,158 frames fail only because the outline reaches the frame edge.

### S5.2 Arm B "whales cut by the frame" picks (`pick_partial.py`, fixed before any arm-B mesh; `trial_partial/`)

Coordinator's addition (2026-11-24): the frames arm A drops are mostly near whales cut by the frame edge, the footage
the user mostly has. Numbers: the 66 numbers >= 3 s with 0 arm-A frames (2025-02-03 56, CC-09-85 10). Their frames
are judged again by S0 without the edge clause (sharpness >= the number's median still applies); one frame per
number, largest outlined area, ties by sharpness; list ordered by life_s, longest first. Lock budget 45 min.

| first failing clause (edge clause off) | 2025-02-03 | CC-09-85 |
|---|---|---|
| area < 3 % | 89 | 11 |
| below the fixed back-box top | 1,814 | 273 |
| within 12 px of the back outline | 0 | 59 |
| split outline | 8 | 2 |
| sharpness below the number's median | 390 | 194 |
| passed | 414 | 199 |

Picks: 45 numbers (2025-02-03 36, CC-09-85 9). Dropped: 21 numbers with no frame passing the remaining clauses
(20 whose every frame fails at the back: below the fixed back-box top, for one also the back outline; 1 whose every
frame is under 3 % of the frame).

### S5.3 Arm A results (28 meshes, 13 numbers; TripoSG settings as the smoke test, seed 42, 50 steps, guidance 7)

No ground truth. 4a calls are AI by-eye (the agent, one viewing of input view / side / top renders), written to
`trial/eye_calls.csv` before anything was computed from them: "whale" = reads as a whole whale (body with fluke
and / or flippers, head-to-tail taper), "part" = reads as a whale part alone (fluke, flipper, head, body section),
"invented" = neither. 3 of 28 calls are noted as uncertain (whale vs part twice, part vs whale once).

| | whale | part | invented | n |
|---|---|---|---|---|
| all | 18 (64 %) | 2 (7 %) | 8 (29 %) | 28 |
| 2025-02-03 | 18 | 2 | 6 | 26 |
| CC-09-85 | 0 | 0 | 2 | 2 |
| outline compact (elongation < 2.5) | 15 | 1 | 4 | 20 |
| outline elongated (>= 2.5) | 3 | 1 | 4 | 8 |
| outline 3-6 % of the frame | 12 | 2 | 2 | 16 |
| outline >= 6 % | 6 | 0 | 6 | 12 |
| long axis near horizontal (< 30 or >= 150 deg) | 8 | 0 | 6 | 14 |
| diagonal | 3 | 1 | 2 | 6 |
| near vertical (60-120 deg) | 7 | 1 | 0 | 8 |

Most "whale" meshes are a generic whale (fluke + spindle body + head, small or no flippers), seen from below with the
fluke toward the camera; 3 of 18 have humpback-like long flippers. Of the 8 "invented": 4 are whales seen flat from
below with flippers spread or very near (wing / kite / bird shapes), 2 the flank of the near CC-09-85 whale (disc,
blob), 2 fluke-toward-camera frames (fluke on a blob, disc on a stalk).

4b agreement inside a number (`trial/pairs.csv`; 6,000 surface points kept per mesh, 3,000 used, centred, unit RMS
radius, best of 24 PCA starts + rigid ICP, no mirror, symmetric chamfer in RMS-radius units):

| deployment | pair kind | pairs | median | min - max |
|---|---|---|---|---|
| 2025-02-03 | same number | 24 | 0.119 | 0.048 - 0.260 |
| 2025-02-03 | different number (random 50 of 301, seed 7) | 50 | 0.168 | 0.063 - 0.250 |
| CC-09-85 | same number | 1 | 0.119 | - |
| 2025-02-03, both meshes "whale" | same number | 15 | 0.101 | 0.048 - 0.218 |
| 2025-02-03, both meshes "whale" | different number | 20 | 0.121 | 0.063 - 0.213 |

AUC same < different 0.63 (24 x 50); 13 of 50 different-number pairs are closer than the same-number median. CC-09-85
has one number in arm A, so no different-number pair. The measure does not separate numbers: frames of one whale are
hardly more alike than frames of different whales, as in the smoke test.

4c outline agreement (mesh seen from the best of the 24 axis views, which was the model's +z view for all 28, against
the tracker outline, both cropped and padded alike): median 0.84 (0.37-0.94), 6 of 28 below 0.70. By call: whale
median 0.85 (18), part 0.84 (2), invented 0.83 (8). High IoU does not mean whale-like: the best IoU (0.94) is the
CC-09-85 flank blob.

4d time and memory: 29.1-42.3 s per mesh (median 31.7); torch peak 5.17-5.28 GB; nvidia-smi 5.56-5.72 GB. Lock held
949 s (two pieces: 747 s and 202 s), model load 1.2-1.7 s.

Sheets: `mesh3d_trial_a_best_iou_01.jpg` (12 best by 4c), `mesh3d_trial_b_random_01..02.jpg` (24 of 28, seed
20261124), `mesh3d_trial_c_number_01..06.jpg` (the 6 numbers with most meshes: 4, 4, 4, 3, 2, 2 meshes; the two with 2
were taken from five tied numbers in name order).

### S5.4 Arm B results (45 meshes, one per number; whales cut by the frame; never pooled with arm A)

4a (AI by-eye, `trial_partial/eye_calls.csv`, written first; 13 of 45 calls noted as uncertain, mostly part vs
invented):

| | whale | part | invented | n |
|---|---|---|---|---|
| all | 15 (33 %) | 23 (51 %) | 7 (16 %) | 45 |
| 2025-02-03 | 13 | 17 | 6 | 36 |
| CC-09-85 | 2 | 6 | 1 | 9 |
| outline compact (< 2.5) | 10 | 19 | 6 | 35 |
| outline elongated (>= 2.5) | 5 | 4 | 1 | 10 |
| outline 3-6 % | 2 | 1 | 2 | 5 |
| outline >= 6 % | 13 | 22 | 5 | 40 |
| near horizontal | 4 | 11 | 5 | 20 |
| diagonal | 8 | 2 | 0 | 10 |
| near vertical | 3 | 10 | 2 | 15 |

"Part" here is mostly the honest answer: the model keeps the cut as a flat face and gives the head, a body section or
a flipper (23). In 15 it completes a whole whale beyond the frame edge (invented, but whale-shaped; 3 with humpback-like
long flippers); 7 are other shapes (cup, plates, a triangle for a thin sliver of outline, a turtle).

4c: median IoU 0.88 (0.31-0.97), 5 of 45 below 0.70; whale 0.86 (15), part 0.90 (23), invented 0.92 (7).
4d: 28.6-60.7 s per mesh (median 34.1); torch peak 5.17-6.14 GB; nvidia-smi 5.56-6.31 GB. Lock held 1,715 s of the
2,700 s budget (pieces 783, 777, 155 s); all 45 made, none left out.
Sheets: `mesh3d_trial_partial_01..04.jpg` (all 45 in pick-list order: life_s, longest first).
