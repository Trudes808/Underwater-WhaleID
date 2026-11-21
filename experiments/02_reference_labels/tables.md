# 02_reference_labels: counts

AI by-eye labels (WP3). Definitions: `README.md`.

## Dev clip `2025-12-20 CG.m4v`: frame labels (191 frames, every 15th of 2,857 decoded frames)

| column | counts |
|---|---|
| state | underwater 180, air 8, crossing 3, unusable 0 |
| view | clear 149, bubbles 42, dark 0, glare 0 |
| other_whales | 0: 46, 1: 7, 2: 14, 3: 60, 4: 50, 5: 14 |
| faint_whales | 0: 88, 1: 80, 2: 21, 3: 2 |
| back | back 182, none 9 |
| seafloor | none 191 |
| fish | none 191 (corrected 2026-11-20: f1485, f1500, f1515 were `few`; the user judged the shapes not fish) |

Frames with no clearly visible other whale: 46 (11 air / crossing, 35 underwater: entry froth, just under the
surface, bubble curtains 64-70 s and 87-93 s).

## Dev clip events

| event | frame | time (s) | crossing frames |
|---|---|---|---|
| entry | 24 | 0.801 | 21-23 |
| exit | 437 | 14.579 | 417-436 |
| entry | 492 | 16.414 | 491 |
| exit | 2806 | 93.614 | 2796-2805 |

## Dev clip masks (31 frames)

| item | count |
|---|---|
| whale instances | 85: clear 25, hazy 40, faint 20 |
| fish instances | 0 (the 13 small dark shapes on f1485 / f1500 are painted `unsure` since 2026-11-20) |
| frames with back / bubbles / surface / air / unsure regions | 27 / 16 / 19 / 2 / 4 |
| frames with no other whale | 6 (f60, f390, f450, f1995, f2700, f2820) |
| seafloor | 0 frames (none in the clip) |

## Test recordings 2025-02-03 (CC-11-96): frame labels, 120 sampled frames each (AI by-eye, scoring only)

| piece | state air / underwater | view bubbles | other_whales 0 / 1 / 2 | faint > 0 | back none | seafloor | fish frames (few / school), fish counted | events in windows |
|---|---|---|---|---|---|---|---|---|
| 00004 | 8 / 112 (7 on the drone) | 0 | 94 / 23 / 3 | 6 | 7 | none 120 | 13 / 0, 34 | drop + 1 entry |
| 00005 | 4 / 116 | 1 | 93 / 27 / 0 | 8 | 1 | none 120 | 4 / 0, 10 | 12 (6 breaths) |
| 00006 | 2 / 118 | 1 | 56 / 64 / 0 | 14 | 3 | none 120 | 24 / 1, 58 | 4 (2 breaths) |
| 00007 | 0 / 120 | 0 | 84 / 34 / 2 | 4 | 1 | none 120 | 0 / 0, 0 | 12 (6 breaths) |
| 00008 | 3 / 117 | 0 | 98 / 22 / 0 | 7 | 1 | none 120 | 1 / 0, 5 | 4 (2 breaths) |
| 00009 | 1 / 119 | 1 | 87 / 33 / 0 | 6 | 1 | none 120 | 2 / 0, 3 | 8 (4 breaths) + 1 breath outside |
| 00010 | 1 / 119 | 1 | 107 / 13 / 0 | 7 | 1 | none 120 | 0 / 0, 0 | 4 (2 breaths) |
| all | 19 / 821 | 4 | 619 / 216 / 5 | 52 | 15 | none 840 | 44 / 1, 110 | 1 drop, 26 breaths |

No `crossing`, `dark`, `glare` or `unusable` frame was sampled in the test pieces; view is `clear` otherwise.

Events (frame, time in s; exits then entries; crossing ranges in each events.json):
- 00004: drone drop, release f2723 (90.92), hit f2747 (91.72), entry f2796 (93.35), settled about f2830.
- 00005: exit/entry f46807/46875, 47098/47196, 47532/47635, 48186/48262, 48588/48686, 49204/49296 (1562.9-1646.0).
- 00006: 17429/17507 (581.5/584.1), 18797/18884 (627.1/630.0).
- 00007: 39916/39987, 40388/40478, 41527/41602, 42425/42518, 43931/44006, 44357/44442 (1332.2-1483.2).
- 00008: 39558/39638 (1319.7/1322.4), 40077/40158 (1337.0/1339.7).
- 00009: 4038/4125 (134.7/137.6), 4322/4409 (144.2/147.1), 19354/19434 (645.7/648.3), 20200/20291 (673.9/676.9); outside the windows about 32124/32221 (1071.7/1074.9).
- 00010: 9382/9452 (313.0/315.3), 9780/9867 (326.3/329.2).

## Size and quality of the largest other whale (WP3b part A; frames with a clearly visible other whale)

AI by-eye, 366 frames. Rule: `README.md`, "Size and quality". `near hq` = `ok` frames I would move to `hq` if
backlit silhouettes count.

| video | frames | hq / ok / low | ok with size >= 0.5 | whale_size counts | cut by the edge |
|---|---|---|---|---|---|
| dev 2025-12-20 | 145 | 15 / 94 / 36 | 9 | 0.2: 8, 0.3: 62, 0.4: 51, 0.5: 13, 0.6: 7, 0.7: 4 | 52 |
| 00004 | 26 | 4 / 9 / 13 | | 0.1: 4, 0.2: 5, 0.3: 11, 0.4: 1, 0.5: 1, 0.6: 4 | 16 |
| 00005 | 27 | 1 / 13 / 13 | | 0.1: 3, 0.2: 8, 0.3: 5, 0.4: 5, 0.5: 5, 0.6: 1 | 10 |
| 00006 | 64 | 1 / 30 / 33 | | 0.1: 9, 0.2: 22, 0.3: 16, 0.4: 8, 0.5: 5, 0.6: 3, 0.9: 1 | 42 |
| 00007 | 36 | 5 / 11 / 20 | | 0.1: 6, 0.2: 13, 0.3: 7, 0.4: 3, 0.5: 3, 0.6: 1, 0.7: 2, 0.9: 1 | 22 |
| 00008 | 22 | 0 / 7 / 15 | | 0.1: 6, 0.2: 9, 0.3: 5, 0.4: 1, 0.5: 1 | 13 |
| 00009 | 33 | 0 / 5 / 28 | | 0.1: 11, 0.2: 17, 0.3: 2, 0.4: 1, 0.5: 2 | 19 |
| 00010 | 13 | 0 / 7 / 6 | | 0.1: 1, 0.2: 4, 0.3: 3, 0.4: 5 | 5 |
| test 00004-00010 | 221 | 11 / 82 / 128 | 18 | 0.1: 40, 0.2: 78, 0.3: 49, 0.4: 24, 0.5: 17, 0.6: 9, 0.7: 2, 0.9: 2 | 127 |

Size by grade: dev hq 0.5-0.7 (15), ok 0.3-0.6, low 0.2-0.3; test hq 0.5-0.9 (11), ok 0.3-0.9, low 0.1-0.5 (two
`low` at 0.4-0.5 are slivers at the edge). Area share (rough): test 180 of 221 at <= 0.05, 10 at >= 0.20; dev 110 of
145 at <= 0.05. Close calls: dev 5 + 4 `near hq`, test 21 + 16 `near hq`.

On the test deployment 118 of 221 whale frames (53 %) have the whale at <= 0.2 of the frame width (fish-sized by the
user's measure) and 54 (24 %) at >= 0.4; 11 frames (5 % of the whale frames, 1.3 % of all 840 sampled frames) are
`hq`.

Size check against the reference masks (31 frames with both): mean |eye - mask| 0.046, 26/31 within 0.1, bias -0.015.
