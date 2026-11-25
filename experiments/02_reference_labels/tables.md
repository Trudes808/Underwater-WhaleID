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

## Second deployment 2025-01-02 (CC-09-85): frame labels (AI by-eye, WP3b part B, scoring only)

| piece | frames | state | view | other_whales 0 / 1 (faint > 0) | back none | seafloor none / unsure / visible | fish few | quality hq / ok / low |
|---|---|---|---|---|---|---|---|---|
| 00005 boat | 20 | air 20 | clear 19, glare 1 | 20 / 0 (0) | 20 | 20 / 0 / 0 | 0 | |
| 00006 boat | 20 | air 20 | clear 20 | 20 / 0 (0) | 20 | 20 / 0 / 0 | 0 | |
| 00009 boat | 20 | air 20 | clear 20 | 20 / 0 (0) | 20 | 20 / 0 / 0 | 0 | |
| 00010 | 60 | air 9 (7 drone), underwater 51 | clear 59, bubbles 1 | 58 / 2 (7) | 7 | 60 / 0 / 0 | 0 | 0 / 0 / 2 |
| 00011 | 60 | underwater 60 | clear 60 | 60 / 0 (4) | 0 | 60 / 0 / 0 | 0 | |
| 00012 | 60 | air 1, underwater 59 | clear 60 | 60 / 0 (0) | 0 | 60 / 0 / 0 | 0 | |
| 00017 | 60 | air 1, crossing 1, underwater 49, unusable 9 | clear 39, dark 11, sand 10 | 59 / 1 (2) | 16 | 14 / 24 / 22 | 0 | 0 / 1 / 0 |
| 00023 | 60 | underwater 60 | clear 56, bubbles 3, glare 1 | 47 / 13 (2) | 0 | 46 / 1 / 13 | 5 | 2 / 8 / 3 |
| 00024 | 60 | crossing 1, underwater 59 | clear 60 | 45 / 15 (2) | 16 | 60 / 0 / 0 | 2 | 0 / 9 / 6 |
| all | 420 | air 52, crossing 2, underwater 357, unusable 9 | sand 10, dark 11, bubbles 4, glare 2 | 389 / 31 (17) | 99 | 380 / 25 / 35 | 7 | 2 / 18 / 11 |

Whale size on the 31 whale frames: 0.1: 1, 0.2: 7, 0.3: 5, 0.4: 9, 0.5: 5, 0.6: 2, 0.8: 1, 1.0: 1. `back` none on
00024 = tag floating after release (12 frames) and 4 frames near the surface edge cases.

Seafloor spans (`seafloor_spans.csv`): visible 00017 842-1390 s (548 s, sand plane fading into haze), 00023 0-306 s
(306 s, bottom fading into haze); unsure 00017 296-842 s (546 s, bottom contact, sediment, black), 00023 306-316 s;
none in 00010, 00011, 00012, 00024 and the rest of 00017 / 00023.

Blocked spans (`blocked_spans.csv`, 44 rows): air 27 (26 breaths of 5-11 s and the 92 s drone flight of 00010);
sand 10 spans, 203 s, and dark 7 spans, 215 s, all in 00017 332-840 s. Another whale within 10 s: before and after
2 spans (00024 66-74 s and 120-125 s, probably the same animal both times), after only 1 (00023 1010-1015 s),
unsure 1 (00010 367-374 s); none around the 00017 sediment stretch.

Events, 00023 400-580 s (0.2 s steps): exits / entries 416.8 / 419.8, 443.8 / 445.0, 447.2 / 448.4, 475.0 / 477.6,
512.2 / 516.6 s. The 1 s depth table (scan) marks 416-420, 442-449, 473-478, 512-516 s as surfaced: every end within
about 1 s; the double breath at 445-447 s (2 s just under the surface) is one surfacing for the sensor.

Masks: 29 frames (00017 9, 00023 16, 00024 4): seafloor painted in 14, seafloor-related `unsure` only in 5 (sediment,
bottom fading as the whale rises), froth hard negatives 2; whale instances 14 in 14 frames (clear 4, hazy 10; one
faint whale of 00017 painted unsure); fish instances 7 (jacks).
Fish calls: 7 frames, 8 jacks (00023 5 frames / 6 fish, 00024 2 / 2), all with body, eye and forked tail at
120-560 px.
