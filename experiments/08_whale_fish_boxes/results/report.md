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

