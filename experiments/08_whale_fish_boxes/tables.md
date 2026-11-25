# 08_whale_fish_boxes (WP10 part 1): whale / fish / other for the detector's boxes

Run 2026-11-22. All truth = AI by-eye labels, not ground truth. Every classifier number says what it was fitted on;
nothing is reported from a fold that saw its own test source. Sources: `dev` = the 2025-12-20 clip, `d1` =
2025-02-03 (tag CC-11-96, pieces 00004-00010), `d2` = 2025-01-02 (tag CC-09-85, 4K).

## Plain answer

- The scan's size rule (whale score >= 0.25 and >= 300 px = whale) keeps every large near fish as a whale: 24 of 24
  jacks of 300-800 px on 2025-01-02 (extra boxes) and 4 of 4 in the reference labels. Whale-beats-fish (rule B)
  removes only 4 of 24 and 2 of 4.
- A linear classifier on frozen DINOv3 crop embeddings (tight crop + 2x context) plus box geometry and the five prompt
  scores, used as a VETO after the scan's rule (a box stays a whale only when the rule says whale AND the classifier's
  best class is whale): large jacks kept as whale 24 -> 2 of 24 (trained on dev + 2025-02-03) or 24 -> 1 (trained on
  2025-02-03 alone), with 0 of 18 large whales lost on the same boxes. Reference labels, 2025-01-02 (trained on dev +
  2025-02-03): fish 4 -> 0 of 4, whales >= 300 px 18 -> 16 of 21 kept, other things kept as whale 12 -> 4 of 27.
  2025-02-03 (trained on dev + 2025-01-02): whales >= 300 px 163 -> 163 of 168, other 7 -> 1 of 42; extra whales 35
  -> 35 of 35. The dev clip (trained on both test deployments): whales 61 -> 58 of 76, other 11 -> 4 of 84.
- The reverse direction (trained on 2025-01-02 alone, 75 labelled boxes, 7 fish) loses whales: 163 -> 148 of 168 on
  the 2025-02-03 references, 35 -> 30 of 35 extra whales. It cannot test fish there (2025-02-03 has no fish of 300 px
  or more in any label); it shows that 75 boxes of one deployment are too few.
- The tagged whale's own blow, the surface and the back mirrored in it at breaths (orchestrator request): the scan
  keeps all 69 such boxes in hq / ok windows under 1 m as whales; the classifier (not fitted on any blow box) calls 0
  of 26 (2025-02-03) and 0 of 43 (2025-01-02) whale, and keeps 24 of 27 and 34 of 34 real whales under 1 m. Re-fitting
  with the blow boxes did not help (2025-01-02: 0 -> 4 blow boxes kept); the delivered model is the one without them.
- Step 4 (the box 0.5 s later: overlap with the next frame's boxes, size ratio, similarity of the same square) adds
  nothing on top of the image features (identical counts) and only a little to the no-image model; no tracker built.
- Below 300 px the classifier is not better than the size rule for the user's purpose: it calls 18 of 108 small fish
  of 2025-02-03 whale, where the size rule keeps none as a whale. Hence the recommendation to keep the size rule and
  use the classifier only as a veto on boxes the rule already calls whale.

## Data: labelled objects

Unit = one object: OWLv2 patch boxes of one labelled underwater frame (prompts whale, fish, school of fish, small fish,
air bubbles; the scan's settings) whose best whale / fish prompt score is >= 0.2, grouped at IoU >= 0.5; the best box is
the object. Boxes in the burned-in clock are dropped. The detector was re-run on the labelled frames themselves (the
JPEGs the labels were made on; 2025-02-03 from the `.m4v`, 2025-01-02 from the `.mov` at 1920 px).

Truth rules (`table.py`):
- mask frames: share of the box covered by whale / fish / back / unsure pixels: fish if fish >= 10 % (and whale
  < 10 %; both >= 10 % = ambiguous); whale if whale >= 15 % and fish < half of it; ambiguous if >= 50 % unsure; back
  if back >= 30 %; nothing otherwise. The dev clip's withdrawn "fish" are unsure.
- frames without masks: fish = matches a box of `fish_calls.json` (IoU >= 0.3, >= 70 % inside the call, or the call
  >= 70 % inside a box at most 4x its area). Whale only on frames with exactly one clear other whale: the anchor is
  the best "whale" box (>= 0.1) whose longest side is 0.55-1.6x the labelled `whale_size`; boxes with IoU >= 0.5 or
  >= 60 % inside it are whale, boxes < 10 % inside it are `other` when no faint whale is counted. Everything else on
  frames with 2+ whales, faint whales, uncounted fish or unsure specks is ambiguous and left out (reasons counted in
  the generated tables). Frames with no whale, no faint whale and no fish: every box is `other` (back or nothing).
- Known truth errors seen on the sheets: d1 00007 f38335, a whale's flipper next to the counted whale, is `other`
  (the anchor rule); d2 00023 f29834, a dark fish-like shape over a whale mask, is `whale` (mask rule).

| source | whale | fish | other (back / nothing) | ambiguous, left out |
|---|---|---|---|---|
| dev (191 labelled frames, 31 masks) | 113 | 0 | 222 | 905 (817 on frames with 2+ whales and no mask) |
| d1 (840 frames, 14 masks) | 224 | 108 (none >= 300 px) | 228 | 175 |
| d2 (420 frames, 29 masks) | 25 | 7 (4 >= 300 px) | 43 | 6 |

Extra test boxes (`extra.py`, `extra_labels.json`): because the references hold only 4 fish >= 300 px, 84 boxes the
scan keeps as whales (whale >= 0.25, 300-800 px, from the scan's own detector cache of the `.mov` recordings; half
drawn among boxes whose fish prompt >= 0.3; frames within 90 frames of a labelled frame skipped; seed 20261122) were
labelled by eye by the WP10 agent on crops without scores: d1 35 whale, 1 unsure (all 24 "fishy" d1 boxes are
whales: large fish do not occur in d1); d2 18 whale, 24 fish (jacks, mackerel-like, very near the lens), 4 nothing
(surface from below), 2 unsure. Used as test boxes only, except in the delivered model.

## Model

`uwid.boxclf`: features = DINOv3 ViT-B/16 class token + mean patch token of a square crop (side = longest box side,
224 px) and of a 2x context square, box geometry (10 numbers, 1920 px units), logits of the five prompt scores and
whale minus best fish. StandardScaler + multinomial logistic regression, class weights balanced; C chosen by an inner
leave-one-recording-out search on the training sources only (chosen 0.0003-0.001 for the image models).

Delivered: `underwater_cache/boxclf/boxclf_v1.npz`, fitted on all reference boxes of the three sources + the 81
labelled extra boxes (1,051 boxes: 415 whale, 139 fish, 497 other), C 0.0003. Not scored (it saw every source).
`boxclf_v1_with_breath.npz` adds the 133 labelled breath boxes (not recommended, see below).

## Results (leave one source out; generated tables in `results/report.md`, copied below)

### The band that matters, by fold (`tight+ctx`; scan rule A -> classifier alone)

| fold (trained on -> tested on) | fish 300-800 called whale | whales 300-800 not called whale | whales >= 800 not whale | other 300-800 called whale |
|---|---|---|---|---|
| dev + d2 -> d1 | 0 -> 0 of 0 | 4 -> 0 of 141 | 1 -> 0 of 27 | 3 -> 3 of 19 |
| dev + d1 -> d2 | 4 -> 0 of 4 | 3 -> 1 of 10 | 0 -> 1 of 11 | 5 -> 1 of 11 |
| d1 -> d2 | 4 -> 0 of 4 | 3 -> 1 of 10 | 0 -> 0 of 11 | 5 -> 1 of 11 |
| d2 -> d1 | 0 -> 0 of 0 | 4 -> 13 of 141 | 1 -> 2 of 27 | 3 -> 1 of 19 |
| d1 + d2 -> dev | 0 -> 0 of 0 | 12 -> 7 of 66 | 3 -> 1 of 10 | 6 -> 8 of 37 |

(The baseline's "whale not called whale" in this table are whales the detector scores under 0.25; the classifier
sees every box >= 0.2 and can name them, but as a veto it never adds a whale.)

### Extra boxes (all kept as whales by the scan today)

| fold | fish: rule A / rule B / classifier call whale | whales lost: A / B / classifier | nothing called whale: A / B / classifier |
|---|---|---|---|
| dev + d1 -> d2 | 24 / 20 / 2 of 24 | 0 / 0 / 0 of 18 | 4 / 4 / 0 of 4 |
| d1 -> d2 | 24 / 20 / 1 of 24 | 0 / 0 / 0 of 18 | 4 / 4 / 0 of 4 |
| dev + d2 -> d1 | (no fish) | 0 / 0 / 0 of 35 | |
| d2 -> d1 | (no fish) | 0 / 0 / 5 of 35 | |

Feature sets on the extra fish (dev + d1 -> d2): geometry + scores only 11 of 24 still whale; tight crop 3; tight +
context 2.

### What it still gets wrong (sheets `boxclf_wrong_01..05`)

In the large band: whales seen edge-on behind bubbles or in haze called `other` (dev f585, f1545, f2640); a whale
beside the back whose box also covers the back (d2 00023 f32555, 1,904 px); a jack against the light called whale
(d2 00023 f23302, 324 px, p 0.59; f23167, 407 px, p 0.47); thin pale arcs (tag antenna reflection, dev f810) and the
back's whole-frame box (d2 00023 f1264) called whale. Below 300 px: most errors are small fish called `other` (67 of
94 fish < 150 px on d1, harmless for keeping) and 17 called whale.

### Breath boxes (orchestrator request 2026-11-22; `breath.py`, `breath_labels.json`, `results/breath.json`)

The scan's best box of every hq / ok window with depth_m < 1 on both deployments (189 windows; windows of one
recording closer than 2 s to a chosen one skipped -> 138 boxes), labelled by eye by the WP10 agent on full frames with
the box in white and no scores: 2025-02-03 27 whale, 26 other (blow, bubble burst, surface, mirrored back), 3 unsure;
2025-01-02 34 whale, 43 other, 3 fish, 2 unsure. 2025-02-03 00012's whale beside the floating tag (8 boxes) may be
the tagged whale itself; counted as another whale.

| fold | blow / surface boxes kept as whale: scan / classifier before re-fit / after re-fit | real whales < 1 m lost: before / after |
|---|---|---|
| dev + d2 -> d1 | 26 / 0 / 0 of 26 | 3 / 3 of 27 |
| dev + d1 -> d2 | 43 / 0 / 4 of 43 | 0 / 0 of 34 |
| d1 -> d2 | 43 / 0 / 0 of 43 | 0 / 0 of 34 |
| d2 -> d1 | 26 / 1 / 1 of 26 | 2 / 4 of 27 |

The re-fit adds the other deployment's breath boxes to training (same leave-one-source-out rule). The 300-800 px fish
numbers do not move (extra fish called whale 2 / 1 of 24 before and after; reference fish 0 of 4), reference whales
>= 300 px lost 0 -> 1 of 168 (d1) and 2 -> 1 of 21 (d2). The 3 lost shallow whales on d1 (p whale 0.24-0.46): a dark
whale part at the left edge under the mirrored back (00004 f18516, breath id 1) and the whale cut by the right
edge in a bubble cloud beside the floating tag (00012 f46976, f47051). Sheets: `review_sheets/underwater/boxclf_breath_01.jpg` (12 blow / surface boxes) and `_02` (12 real
shallow whales), decision of the delivered model's folds drawn.

## Use (for the scan and the tracker)

```python
from uwid import boxclf
res = boxclf.classify(frame_rows, rgb_frame)     # rows of ONE frame in the scan cache format, frame in their pixels
keep = (rows["query"] == "whale") & rule_a_whale & (res["boxclf_class"] == "whale")   # veto
```

`classify` returns, per input row, `boxclf_class` (whale / fish / other, "" under the 0.2 score floor),
`boxclf_p`, `p_whale`, `p_fish`, `p_other`, `boxclf_obj` (rows of one object share the answer). It loads DINOv3
once and keeps it. Cost (from extract.py timings): about 20 ms of GPU per object (two 224 px crops); run it only on
frames where rule A already has a whale box.

## Files

`data.py` (sources), `partners.py` (frames +0.5 s, CPU), `extract.py owl feat` (GPU), `table.py` (truth + baseline),
`train.py` (folds, delivered model), `report.py` (`results/report.md`, `report.json`), `sheets.py`
(`boxclf_fixed_01..05`, `boxclf_wrong_01..05`), `extra.py` + `extra_labels.json`, `breath.py` + `breath_labels.json`.
Cache `underwater_cache/boxclf/` (detector rows, crop features, predictions, weights, extra / breath frames).

# Generated tables (`results/report.md`)

## Labelled objects (detector boxes >= 0.2, grouped), by source, truth and size band (px, 1920 px frame)

| source truth | <150 | 150-300 | 300-500 | 500-800 | >=800 | all |
|---|---|---|---|---|---|---|
| d1 ambiguous | 133 | 9 | 11 | 9 | 13 | 175 |
| d1 back | 1 | 0 | 0 | 0 | 0 | 1 |
| d1 fish | 94 | 14 | 0 | 0 | 0 | 108 |
| d1 nothing | 33 | 2 | 1 | 0 | 0 | 36 |
| d1 other | 139 | 11 | 9 | 9 | 23 | 191 |
| d1 whale | 9 | 47 | 87 | 54 | 27 | 224 |
| d2 ambiguous | 1 | 4 | 0 | 0 | 1 | 6 |
| d2 back | 1 | 0 | 0 | 0 | 5 | 6 |
| d2 fish | 1 | 2 | 2 | 2 | 0 | 7 |
| d2 nothing | 7 | 0 | 2 | 0 | 1 | 10 |
| d2 other | 7 | 1 | 4 | 5 | 10 | 27 |
| d2 whale | 2 | 2 | 4 | 6 | 11 | 25 |
| dev ambiguous | 172 | 192 | 274 | 180 | 87 | 905 |
| dev back | 0 | 0 | 0 | 0 | 1 | 1 |
| dev nothing | 41 | 27 | 12 | 6 | 15 | 101 |
| dev other | 43 | 27 | 15 | 4 | 31 | 120 |
| dev whale | 9 | 28 | 40 | 26 | 10 | 113 |

Ambiguous objects (left out of training and scoring), by reason:

| source | why | n |
|---|---|---|
| d1 | 2+ whales, no mask | 16 |
| d1 | faint whale counted | 4 |
| d1 | faint whale only | 28 |
| d1 | mask | 1 |
| d1 | near the anchor | 11 |
| d1 | no anchor box | 3 |
| d1 | school box with whale | 4 |
| d1 | small box on a fish frame | 86 |
| d1 | unsure specks | 22 |
| d2 | faint whale only | 2 |
| d2 | mask | 1 |
| d2 | near the anchor | 1 |
| d2 | no anchor box | 1 |
| d2 | unsure specks | 1 |
| dev | 2+ whales, no mask | 817 |
| dev | faint whale counted | 3 |
| dev | faint whale only | 72 |
| dev | mask | 6 |
| dev | near the anchor | 7 |

How the truth was decided (non-ambiguous objects):

| source | truth | why | n |
|---|---|---|---|
| d1 | back | mask | 1 |
| d1 | fish | fish call | 65 |
| d1 | fish | mask | 43 |
| d1 | nothing | mask | 36 |
| d1 | other | no whale, no fish | 69 |
| d1 | other | off the anchor | 122 |
| d1 | whale | anchor | 213 |
| d1 | whale | mask | 11 |
| d2 | back | mask | 6 |
| d2 | fish | fish call | 1 |
| d2 | fish | mask | 6 |
| d2 | nothing | mask | 10 |
| d2 | other | no whale, no fish | 25 |
| d2 | other | off the anchor | 2 |
| d2 | whale | anchor | 6 |
| d2 | whale | mask | 19 |
| dev | back | mask | 1 |
| dev | nothing | mask | 101 |
| dev | other | no whale, no fish | 112 |
| dev | other | off the anchor | 8 |
| dev | whale | anchor | 7 |
| dev | whale | mask | 106 |

## Baseline (no training): the scan's rules

Rule A = scan (whale score >= 0.25 and >= 300 px = whale; shorter = small; fish prompt >= 0.5 = fish; else none). Rule B = A + whale score must beat fish score. Columns: what the rule calls the object.

### Rule A on 2025-12-20 clip

| truth band | n | whale | small | fish | none |
|---|---|---|---|---|---|
| whale <150 | 9 | 0 | 4 | 0 | 5 |
| whale 150-300 | 28 | 0 | 23 | 0 | 5 |
| whale 300-500 | 40 | 33 | 0 | 0 | 7 |
| whale 500-800 | 26 | 21 | 0 | 0 | 5 |
| whale >=800 | 10 | 7 | 0 | 0 | 3 |
| other <150 | 84 | 0 | 3 | 1 | 80 |
| other 150-300 | 54 | 0 | 12 | 2 | 40 |
| other 300-500 | 27 | 4 | 0 | 0 | 23 |
| other 500-800 | 10 | 2 | 0 | 0 | 8 |
| other >=800 | 47 | 5 | 0 | 3 | 39 |

### Rule A on 2025-02-03 (CC-11-96)

| truth band | n | whale | small | fish | none |
|---|---|---|---|---|---|
| whale <150 | 9 | 0 | 2 | 0 | 7 |
| whale 150-300 | 47 | 0 | 42 | 0 | 5 |
| whale 300-500 | 87 | 83 | 0 | 0 | 4 |
| whale 500-800 | 54 | 54 | 0 | 0 | 0 |
| whale >=800 | 27 | 26 | 0 | 0 | 1 |
| fish <150 | 94 | 0 | 34 | 41 | 19 |
| fish 150-300 | 14 | 0 | 11 | 1 | 2 |
| other <150 | 173 | 0 | 11 | 24 | 138 |
| other 150-300 | 13 | 0 | 4 | 0 | 9 |
| other 300-500 | 10 | 1 | 0 | 0 | 9 |
| other 500-800 | 9 | 2 | 0 | 0 | 7 |
| other >=800 | 23 | 4 | 0 | 1 | 18 |

### Rule A on 2025-01-02 (CC-09-85)

| truth band | n | whale | small | fish | none |
|---|---|---|---|---|---|
| whale <150 | 2 | 0 | 1 | 0 | 1 |
| whale 150-300 | 2 | 0 | 1 | 0 | 1 |
| whale 300-500 | 4 | 4 | 0 | 0 | 0 |
| whale 500-800 | 6 | 3 | 0 | 0 | 3 |
| whale >=800 | 11 | 11 | 0 | 0 | 0 |
| fish <150 | 1 | 0 | 1 | 0 | 0 |
| fish 150-300 | 2 | 0 | 2 | 0 | 0 |
| fish 300-500 | 2 | 2 | 0 | 0 | 0 |
| fish 500-800 | 2 | 2 | 0 | 0 | 0 |
| other <150 | 15 | 0 | 2 | 1 | 12 |
| other 150-300 | 1 | 0 | 1 | 0 | 0 |
| other 300-500 | 6 | 2 | 0 | 0 | 4 |
| other 500-800 | 5 | 3 | 0 | 0 | 2 |
| other >=800 | 16 | 7 | 0 | 0 | 9 |

### Key errors of the baseline (300-800 px band; whales >= 800 px)

| rule|test source | fish_300_800 | fish_called_whale | whale_300_800 | whale_not_whale | whale_ge800 | whale_ge800_not_whale | other_300_800 | other_called_whale |
|---|---|---|---|---|---|---|---|---|
| A|dev | 0 | 0 | 66 | 12 | 10 | 3 | 37 | 6 |
| B|dev | 0 | 0 | 66 | 12 | 10 | 3 | 37 | 3 |
| A|d1 | 0 | 0 | 141 | 4 | 27 | 1 | 19 | 3 |
| B|d1 | 0 | 0 | 141 | 5 | 27 | 1 | 19 | 3 |
| A|d2 | 4 | 4 | 10 | 3 | 11 | 0 | 11 | 5 |
| B|d2 | 4 | 2 | 10 | 3 | 11 | 0 | 11 | 5 |

## Classifier, leave one source out (feature set `tight+ctx` unless named)

### `geom+scores`

| fold | n | fish 300-800: base A -> clf | whale 300-800 lost: base A -> clf | whale >=800 lost: base A -> clf | other 300-800 called whale: base A -> clf | C |
|---|---|---|---|---|---|---|
| test dev | 335 | 0 -> 0 of 0 | 12 -> 13 of 66 | 3 -> 2 of 10 | 6 -> 5 of 37 | 0.1 |
| test d1 | 560 | 0 -> 0 of 0 | 4 -> 3 of 141 | 1 -> 1 of 27 | 3 -> 6 of 19 | 0.1 |
| test d2 | 75 | 4 -> 2 of 4 | 3 -> 0 of 10 | 0 -> 0 of 11 | 5 -> 2 of 11 | 0.1 |
| d1 -> d2 | 75 | 4 -> 1 of 4 | 3 -> 0 of 10 | 0 -> 0 of 11 | 5 -> 3 of 11 | 0.1 |
| d2 -> d1 | 560 | 0 -> 0 of 0 | 4 -> 39 of 141 | 1 -> 2 of 27 | 3 -> 3 of 19 | 0.1 |

### `geom+scores+motion`

| fold | n | fish 300-800: base A -> clf | whale 300-800 lost: base A -> clf | whale >=800 lost: base A -> clf | other 300-800 called whale: base A -> clf | C |
|---|---|---|---|---|---|---|
| test dev | 335 | 0 -> 0 of 0 | 12 -> 14 of 66 | 3 -> 3 of 10 | 6 -> 3 of 37 | 0.1 |
| test d1 | 560 | 0 -> 0 of 0 | 4 -> 0 of 141 | 1 -> 0 of 27 | 3 -> 4 of 19 | 0.1 |
| test d2 | 75 | 4 -> 0 of 4 | 3 -> 0 of 10 | 0 -> 0 of 11 | 5 -> 1 of 11 | 0.1 |
| d1 -> d2 | 75 | 4 -> 0 of 4 | 3 -> 0 of 10 | 0 -> 0 of 11 | 5 -> 1 of 11 | 0.1 |
| d2 -> d1 | 560 | 0 -> 0 of 0 | 4 -> 21 of 141 | 1 -> 0 of 27 | 3 -> 1 of 19 | 0.03 |

### `tight`

| fold | n | fish 300-800: base A -> clf | whale 300-800 lost: base A -> clf | whale >=800 lost: base A -> clf | other 300-800 called whale: base A -> clf | C |
|---|---|---|---|---|---|---|
| test dev | 335 | 0 -> 0 of 0 | 12 -> 11 of 66 | 3 -> 1 of 10 | 6 -> 7 of 37 | 0.0003 |
| test d1 | 560 | 0 -> 0 of 0 | 4 -> 2 of 141 | 1 -> 1 of 27 | 3 -> 1 of 19 | 0.03 |
| test d2 | 75 | 4 -> 0 of 4 | 3 -> 1 of 10 | 0 -> 0 of 11 | 5 -> 1 of 11 | 0.003 |
| d1 -> d2 | 75 | 4 -> 0 of 4 | 3 -> 1 of 10 | 0 -> 0 of 11 | 5 -> 1 of 11 | 0.001 |
| d2 -> d1 | 560 | 0 -> 0 of 0 | 4 -> 9 of 141 | 1 -> 2 of 27 | 3 -> 0 of 19 | 0.003 |

### `tight+ctx`

| fold | n | fish 300-800: base A -> clf | whale 300-800 lost: base A -> clf | whale >=800 lost: base A -> clf | other 300-800 called whale: base A -> clf | C |
|---|---|---|---|---|---|---|
| test dev | 335 | 0 -> 0 of 0 | 12 -> 7 of 66 | 3 -> 1 of 10 | 6 -> 8 of 37 | 0.0003 |
| test d1 | 560 | 0 -> 0 of 0 | 4 -> 0 of 141 | 1 -> 0 of 27 | 3 -> 3 of 19 | 0.0003 |
| test d2 | 75 | 4 -> 0 of 4 | 3 -> 1 of 10 | 0 -> 1 of 11 | 5 -> 1 of 11 | 0.001 |
| d1 -> d2 | 75 | 4 -> 0 of 4 | 3 -> 1 of 10 | 0 -> 0 of 11 | 5 -> 1 of 11 | 0.0003 |
| d2 -> d1 | 560 | 0 -> 0 of 0 | 4 -> 13 of 141 | 1 -> 2 of 27 | 3 -> 1 of 19 | 0.003 |

### `tight+ctx+motion`

| fold | n | fish 300-800: base A -> clf | whale 300-800 lost: base A -> clf | whale >=800 lost: base A -> clf | other 300-800 called whale: base A -> clf | C |
|---|---|---|---|---|---|---|
| test dev | 335 | 0 -> 0 of 0 | 12 -> 7 of 66 | 3 -> 1 of 10 | 6 -> 8 of 37 | 0.0003 |
| test d1 | 560 | 0 -> 0 of 0 | 4 -> 0 of 141 | 1 -> 0 of 27 | 3 -> 3 of 19 | 0.0003 |
| test d2 | 75 | 4 -> 0 of 4 | 3 -> 1 of 10 | 0 -> 1 of 11 | 5 -> 1 of 11 | 0.001 |
| d1 -> d2 | 75 | 4 -> 0 of 4 | 3 -> 1 of 10 | 0 -> 0 of 11 | 5 -> 1 of 11 | 0.0003 |
| d2 -> d1 | 560 | 0 -> 0 of 0 | 4 -> 13 of 141 | 1 -> 2 of 27 | 3 -> 1 of 19 | 0.003 |

### Confusion by band, `tight+ctx`, fold test dev (columns: classifier's call)

| truth band | n | whale | fish | other |
|---|---|---|---|---|
| whale <150 | 9 | 5 | 0 | 4 |
| whale 150-300 | 28 | 23 | 0 | 5 |
| whale 300-500 | 40 | 35 | 0 | 5 |
| whale 500-800 | 26 | 24 | 0 | 2 |
| whale >=800 | 10 | 9 | 0 | 1 |
| other <150 | 84 | 1 | 0 | 83 |
| other 150-300 | 54 | 5 | 0 | 49 |
| other 300-500 | 27 | 4 | 0 | 23 |
| other 500-800 | 10 | 4 | 0 | 6 |
| other >=800 | 47 | 6 | 0 | 41 |

### Confusion by band, `tight+ctx`, fold test d1 (columns: classifier's call)

| truth band | n | whale | fish | other |
|---|---|---|---|---|
| whale <150 | 9 | 3 | 0 | 6 |
| whale 150-300 | 47 | 45 | 0 | 2 |
| whale 300-500 | 87 | 87 | 0 | 0 |
| whale 500-800 | 54 | 54 | 0 | 0 |
| whale >=800 | 27 | 27 | 0 | 0 |
| fish <150 | 94 | 17 | 10 | 67 |
| fish 150-300 | 14 | 1 | 7 | 6 |
| other <150 | 173 | 4 | 1 | 168 |
| other 150-300 | 13 | 3 | 0 | 10 |
| other 300-500 | 10 | 1 | 0 | 9 |
| other 500-800 | 9 | 2 | 0 | 7 |
| other >=800 | 23 | 1 | 0 | 22 |

### Confusion by band, `tight+ctx`, fold test d2 (columns: classifier's call)

| truth band | n | whale | fish | other |
|---|---|---|---|---|
| whale <150 | 2 | 1 | 0 | 1 |
| whale 150-300 | 2 | 2 | 0 | 0 |
| whale 300-500 | 4 | 3 | 1 | 0 |
| whale 500-800 | 6 | 6 | 0 | 0 |
| whale >=800 | 11 | 10 | 0 | 1 |
| fish <150 | 1 | 0 | 1 | 0 |
| fish 150-300 | 2 | 0 | 2 | 0 |
| fish 300-500 | 2 | 0 | 2 | 0 |
| fish 500-800 | 2 | 0 | 2 | 0 |
| other <150 | 15 | 2 | 3 | 10 |
| other 150-300 | 1 | 0 | 1 | 0 |
| other 300-500 | 6 | 1 | 0 | 5 |
| other 500-800 | 5 | 0 | 0 | 5 |
| other >=800 | 16 | 4 | 0 | 12 |

### Confusion by band, `tight+ctx`, fold d1 -> d2 (columns: classifier's call)

| truth band | n | whale | fish | other |
|---|---|---|---|---|
| whale <150 | 2 | 0 | 0 | 2 |
| whale 150-300 | 2 | 2 | 0 | 0 |
| whale 300-500 | 4 | 3 | 1 | 0 |
| whale 500-800 | 6 | 6 | 0 | 0 |
| whale >=800 | 11 | 11 | 0 | 0 |
| fish <150 | 1 | 0 | 1 | 0 |
| fish 150-300 | 2 | 0 | 2 | 0 |
| fish 300-500 | 2 | 0 | 2 | 0 |
| fish 500-800 | 2 | 0 | 2 | 0 |
| other <150 | 15 | 1 | 3 | 11 |
| other 150-300 | 1 | 0 | 1 | 0 |
| other 300-500 | 6 | 1 | 0 | 5 |
| other 500-800 | 5 | 0 | 0 | 5 |
| other >=800 | 16 | 7 | 0 | 9 |

### Confusion by band, `tight+ctx`, fold d2 -> d1 (columns: classifier's call)

| truth band | n | whale | fish | other |
|---|---|---|---|---|
| whale <150 | 9 | 1 | 0 | 8 |
| whale 150-300 | 47 | 36 | 0 | 11 |
| whale 300-500 | 87 | 75 | 1 | 11 |
| whale 500-800 | 54 | 53 | 0 | 1 |
| whale >=800 | 27 | 25 | 0 | 2 |
| fish <150 | 94 | 0 | 6 | 88 |
| fish 150-300 | 14 | 0 | 5 | 9 |
| other <150 | 173 | 2 | 0 | 171 |
| other 150-300 | 13 | 2 | 0 | 11 |
| other 300-500 | 10 | 0 | 0 | 10 |
| other 500-800 | 9 | 1 | 0 | 8 |
| other >=800 | 23 | 1 | 0 | 22 |

## Extra test boxes, 300-800 px (scan detector cache, AI by-eye labels by the WP10 agent; test only)

Every box here is one the scan keeps as a whale (whale score >= 0.25, 300-800 px). Half were drawn among boxes whose best fish prompt scores >= 0.3. Ambiguous boxes (3) left out.

| features | fold | fish: rule A / B / clf called whale | whale: rule A / B / clf lost | other: rule A / B / clf called whale |
|---|---|---|---|
| geom+scores | test d1 | 0 / 0 / 0 of 0 | 0 / 0 / 3 of 35 | 0 / 0 / 0 of 0 |
| geom+scores | test d2 | 24 / 20 / 11 of 24 | 0 / 0 / 0 of 18 | 4 / 4 / 2 of 4 |
| geom+scores | d1 -> d2 | 24 / 20 / 6 of 24 | 0 / 0 / 0 of 18 | 4 / 4 / 2 of 4 |
| geom+scores | d2 -> d1 | 0 / 0 / 0 of 0 | 0 / 0 / 20 of 35 | 0 / 0 / 0 of 0 |
| tight | test d1 | 0 / 0 / 0 of 0 | 0 / 0 / 0 of 35 | 0 / 0 / 0 of 0 |
| tight | test d2 | 24 / 20 / 3 of 24 | 0 / 0 / 0 of 18 | 4 / 4 / 0 of 4 |
| tight | d1 -> d2 | 24 / 20 / 2 of 24 | 0 / 0 / 0 of 18 | 4 / 4 / 0 of 4 |
| tight | d2 -> d1 | 0 / 0 / 0 of 0 | 0 / 0 / 4 of 35 | 0 / 0 / 0 of 0 |
| tight+ctx | test d1 | 0 / 0 / 0 of 0 | 0 / 0 / 0 of 35 | 0 / 0 / 0 of 0 |
| tight+ctx | test d2 | 24 / 20 / 2 of 24 | 0 / 0 / 0 of 18 | 4 / 4 / 0 of 4 |
| tight+ctx | d1 -> d2 | 24 / 20 / 1 of 24 | 0 / 0 / 0 of 18 | 4 / 4 / 0 of 4 |
| tight+ctx | d2 -> d1 | 0 / 0 / 0 of 0 | 0 / 0 / 5 of 35 | 0 / 0 / 0 of 0 |

## Recommended use: scan rule A, then the classifier as a veto (`tight+ctx`)

A box stays a whale only when rule A calls it whale AND the classifier's best class is whale. Below 300 px nothing changes (the size rule stays: small = fish or far whale).

| boxes | fold | whale >= 300 px kept as whale: A -> A+veto | fish >= 300 px kept as whale: A -> A+veto | other >= 300 px kept as whale: A -> A+veto |
|---|---|---|---|
| reference | test dev | 61 -> 58 of 76 | 0 -> 0 of 0 | 11 -> 4 of 84 |
| reference | test d1 | 163 -> 163 of 168 | 0 -> 0 of 0 | 7 -> 1 of 42 |
| reference | test d2 | 18 -> 16 of 21 | 4 -> 0 of 4 | 12 -> 4 of 27 |
| reference | d1 -> d2 | 18 -> 17 of 21 | 4 -> 0 of 4 | 12 -> 5 of 27 |
| reference | d2 -> d1 | 163 -> 148 of 168 | 0 -> 0 of 0 | 7 -> 2 of 42 |
| extra | test d1 | 35 -> 35 of 35 | 0 -> 0 of 0 | 0 -> 0 of 0 |
| extra | test d2 | 18 -> 18 of 18 | 24 -> 2 of 24 | 4 -> 0 of 4 |
| extra | d1 -> d2 | 18 -> 18 of 18 | 24 -> 1 of 24 | 4 -> 0 of 4 |
| extra | d2 -> d1 | 35 -> 30 of 35 | 0 -> 0 of 0 | 0 -> 0 of 0 |

