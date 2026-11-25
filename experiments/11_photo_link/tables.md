# 11_photo_link (WP16): do tag video and boat photos line up in time?

Read-only feasibility check, 2026-11-23. Nothing fitted, trained or tuned. Counts only: no photo file names, no
positions, no names of people. Per-photo / per-surfacing rows: `underwater_cache/photo_link/` (photos_<day>.csv,
surfacings_<day>.csv, shift_scan_<day>.csv, summary.json, recordings_<day>.csv, sheet_selection.csv, cg_counts.json).
Code: `photo_link.py` (Q1-Q4, Q6), `make_sheet.py` (Q5), `cg_counts.py` (Q7).

## Data used

| side | source | 2025-02-03 | 2025-01-02 |
|---|---|---|---|
| tag | `underwater_cache/tagdata/<dep>/` states, surfacings, sensors (GNSS time column) | CC-11-96, drone-dropped | CC-09-85, pole-tagged |
| tag video on disk | scan windows tables | 9 recordings (00004-00012), cover 15,678 of 15,904 on-whale s | 9 recordings; on-whale coverage 5,445 s of 13,205 (00010-00012, 00017, 00023-00024) |
| photos | Dinov3-WhaleID `days_all/<day>/safe/{photos,exif,clusters}.json` (EXIF DateTimeOriginal + SubSec) | 202 photos, 1 camera body | 981 photos, 2 bodies (970 + 11) |
| catalogue | `labels_all_days.json` (Lightroom catalogue animal per photo) | 202 labelled | 981 labelled (incl. no-whale) |
| notes | `test4.xlsx` via `moonshot_cache/notes/tables.py` | 3 groups | 6 groups |

The same camera body (same serial) took all 202 photos on 02-03 and 970 of 981 on 01-02. The Lightroom catalogue
(`/datasets/NOAA_whales_data/Test.lrcat`, read-only) holds 981 / 202 photos for these days and none after 2025-04;
no photo of 2025-12-20 is on disk.

## Clocks

| clock | what it is | how known |
|---|---|---|
| tag `t` / UTC | s since the tag CSV's first sample; its UTC stamp is the tag's own clock | `meta.json` |
| tag vs GNSS | tag clock minus GNSS time logged in the same CSV rows: 02-03 +12.36 s (p05-p95 11.98-12.74, n 220,756); 01-02 +9.32 s (9.24-10.28, n 4,508) | sensors_50hz gps_date/gps_time; assumes the GNSS column is UTC (if it is GPS time, both shift by 18 s) |
| camera | EXIF local time of the camera, taken as HST (UTC-10, no DST in Hawaii) | catalogue capture_time = EXIF time on 202 / 981 photos (max diff 0 s); labelled photos fall in their own notes group interval: 02-03 192/192 at 0 h vs 164 at +1 h, 135 at -1 h; 01-02 701/934 at 0 h vs 0 at +1 h, 144 at -1 h |
| notes | boat log start / end times, HST | as above |

Offset camera to tag, from one photographed tag-on event per day (AI by-eye on the photos; sensor event on the tag):

| day | event | tag clock (HST) | photos (camera, HST) | shift = tag - camera |
|---|---|---|---|---|
| 2025-02-03 | drone drop, impact (drops.csv) | 10:14:07.785 | 10:17:00.96 tag in the air above the whale; 10:17:01.30 tag on the back (7-photo burst, catalogue G01A01) | -173.3 +- 0.2 s |
| 2025-01-02 | pole tag, acceleration spike 33 m/s2 (depth leaves 0 m 2 s later) | 12:31:25.0 | 12:34:13.0-17.2: pole placing the tag on G05A04 (9 photos, a person in frame: not on the sheet) | -170.2 +- 0.6 s |

Camera minus GNSS: 02-03 +185.7 s, 01-02 +179.5 s: the camera clock ran 6 s slow over 31 days relative to GNSS
(0.2 s/day), so the two tags' clocks and the camera agree with one consistent picture.

## Q1 time ranges

| | 2025-02-03 | 2025-01-02 |
|---|---|---|
| tag on whale (tag clock, HST) | 10:14:10 - 14:39:15 (4.42 h) | 12:31:28 - 16:11:34 (3.67 h) |
| photos (camera, HST) | 10:11:13 - 14:31:09 | 08:32:16 - 15:42:51 |
| overlap | 257 min | 191 min |
| photos inside on-whale, shift 0 / event shift | 185 / 171 of 202 | 86 / 57 of 981 |
| of those, the tagged group (notes / catalogue) | G01: 104 (A03 77, A01 26, A02 1) | G05: 22 (all A04) |
| tagged-group photographing after tag-on | 105 min span, 32 tag surfacings inside it | 1.6 min (13:25-13:26 camera), 1 surfacing; the boat then went to G06 (MCE, 26 photos) |
| tag video on disk at those times | yes (00004-00012 cover the whole stretch) | no: 13:22 tag clock falls in pieces 00013-00016, not on disk |

## Q2 surfacings of the tagged whale (depth: tag port out of the water, `surfacings.csv`)

| | 2025-02-03 | 2025-01-02 |
|---|---|---|
| surfacings | 80 (18.1 / h), median 3.6 s, 293 s in all | 84 (22.9 / h), median 4.1 s, 317 s in all |
| bouts (< 1 m) | 72 (16.3 / h) | 51 (13.9 / h) |
| photos within 3 / 5 / 15 s, shift 0 | 0 / 0 / 0 of 185 | 1 / 6 / 10 of 86 |
| chance (share of on-whale time x photos) | 9.0 / 12.4 / 23.3 | 4.9 / 6.6 / 13.6 |
| photos within 3 / 5 / 15 s, event shift | 47 / 57 / 80 of 185 (12 / 13 / 15 surfacings) | 17 / 22 / 22 of 86 (2 / 2 / 2 surfacings) |

Without the clock correction nothing lines up (02-03: 0 photos within 15 s where 23 would be chance): the camera
is about 3 min ahead of the tag. With it, 02-03 photos are 3-5x chance near surfacings. Photo minus surfacing start
(event shift, within +-30 s, 5 s bins from -30): 02-03 [0,0,2,0,0,4,32,27,9,11,8,6], median +7.3 s (99 photos);
01-02 [0,0,0,0,0,0,9,13,0,0,0,0], median +5.8 s (22). Photos come 0-10 s after the tag leaves the water (the fin and
fluke show after the blow), which is why a +-3 s window around the tag interval is slightly off-centre.

## Q3 clock offset scan (-600..+600 s, 1 s steps, photos within 3 s of a surfacing)

| | 2025-02-03 | 2025-01-02 |
|---|---|---|
| best shift | -178 s: 62 photos, 15 surfacings | +542 s: 35 photos, 5 surfacings |
| median / MAD over shifts | 7 / 6 | 1 / 1 |
| p95 / p99 | 29 / 45 | 17 / 28 |
| peak / median, robust z | 8.9, 6.2 | 35, 22.9 (median near 0: z meaningless) |
| at the event shift (rounded) | -173: 44 | -170: 17 (local peak -172/-173: 22, 2 surfacings) |
| other peaks >= 15 | many (-219: 46, -212: 45, -185: 45, -498: 37, ...) | 13 (+433: 33, +511: 29, +286: 22, ...) |
| most surfacings hit | -178 (15) | +549 (6) |

02-03: the global maximum is 4.7 s from the event-synced offset, on a broad raised shoulder (-230..-170), because
the escort and calf surface with the mother and the photographer shoots every surfacing. It would have synchronised
the clocks to about 5 s without the drop photos. 01-02: the global maximum (+542) is wrong; 86 photos in a few
bursts give many chance peaks. The data alone do not synchronise that day; the event photo and the 02-03 camera
offset (predicted -176 s, found -170 s) do.

## Q4 which animals are at the surfacings (catalogue labels)

| | 2025-02-03 (notes G01: MCE, size 3, "Tagged? Yes", video "Aerial") | 2025-01-02 (notes G05: CG, size 10 / actual 8, "Tagged? Yes") |
|---|---|---|
| within 5 s, best shift | G01A03 45, G01A01 26, G01A02 1 (72) | G05A04 27, G05A05 6, G05A06 2, G06A03 2 (37) |
| within 5 s, event shift | G01A03 34, G01A01 22, G01A02 1 (57) | G05A04 22 (22) |
| share of each animal's on-whale photos within 5 s (event shift) | A01 22/35, A03 34/82, A02 1/1, G02 / G03 animals 0/62 | A04 22/31, all others 0/55 |
| per-animal scan peak (best shift, count, p99 of its own scan) | A01 -177 (26, p99 13); A03 -219 (38, p99 33) | A04 -173 (22, p99 17); others 2-8 photos |
| clusters (safe clusters) within 3 s, best shift | 62 photos in 35 clusters | 35 photos in 12 clusters |
| notes, animals | A01 adult female (high hook), A02 calf, A03 adult male; no animal comment mentions the tag; roles empty | A04 "Nuclear", adult female, comment "Tagged"; A05 "Principle escort", comment "UW fluke from CATS"; A03 "same animal as G03A01", A08 "same animal as G03A06" |

One group dominates on both days (only the tagged group's animals appear near surfacings). Within it, the notes'
tagged animal dominates on 01-02 (A04, 22 of 22). On 02-03 the drop photos are labelled A01 (the mother) and A01's
own scan has the sharpest peak (26 vs p99 13), but the escort A03 has more photos at the surfacings (34 vs 22), and
on the sheet three photos labelled A03 show the tag (see Q5): either some catalogue labels swap A01 / A03 or the
tagged whale is not the one the drop photos are labelled with. Not resolved here.

## Q5 by eye (AI by-eye, one viewing; sheet `review_sheets/underwater/photo_link_tag_candidates_01.jpg`)

12 photos (no person in any; 9 photos of the 01-02 tagging burst and 1 with a person's helmet were left out):
2 drone-drop photos (02-03, A01: tag in the air, tag on the back), 6 surfacing photos 02-03, 4 surfacing photos 01-02.

| | tag visible | possible | not visible |
|---|---|---|---|
| 02-03 surfacing photos (6) | 3 (all labelled G01A03, clear on zoom) | 2 (labelled G01A01: yellow spot at the waterline, yellow object under splash) | 1 (G01A01) |
| 01-02 surfacing photos (4, all G05A04) | 3 (orange tag behind the blowholes) | 0 | 1 (head-on, back turned) |

Uncertain: the two "possible" 02-03 photos; whether the tagged 02-03 whale is A01 or A03 in catalogue terms.

## Q6 positions (counts only)

| | 2025-02-03 | 2025-01-02 |
|---|---|---|
| tag GNSS | `.ubx` holds raw measurements (UBX RXM-RAWX), no position solution; no epoch with >= 4 satellites while on the whale (0; 1,976 before, 889 after); epochs with >= 1: 1,157 in 20 sessions | raw only; 132 epochs with >= 4 satellites on the whale in 8 sessions (2.2 / h), 0 within 3 s and 2 within 15 s of a surfacing |
| photos with EXIF GPS | 202 of 202 (36 distinct positions) | 970 of 981 (169 distinct; the 11 of the second body have none) |
| trip log / notes | start and end position per group: 3 groups | 6 groups |

No fix on the tag at surfacings from the files as they are; post-processing the raw snapshots with ephemeris data
(not tried) might give some. The photos' positions are the boat's (camera GPS), not the whale's.

## Q7 comp-group clip `2025-12-20 CG.m4v` (95 s)

Cached OWLv2 -> SAM 3 runs, cleaned as displayed (`tracks.load_tracks`: back, flicker, fish-sized and duplicate
outlines removed), whale numbers only. Life = first to last frame of a number inside its run (runs are 15-24 s, so
numbers alive at the run end are cut short). "Top edge" = the outline touches the top 2 % of the frame on its last
frame (the camera looks forward along the back, so the top of the frame is ahead / towards the surface; not checked
by eye). Second table: the same runs after the shipped strict purity guards (WP8c), which end a number rather than
let it move, so numbers are shorter and more numerous (`underwater_cache/photo_link/cg_counts.json`).

| run (s) | whale numbers | most outlined at once | median per frame | median / max of per-second max | seconds with >= 2 / >= 3 | life median / max (s), numbers >= 3 s | ends: top edge / elsewhere / alive at run end |
|---|---|---|---|---|---|---|---|
| 5-22 | 10 | 6 | 3 | 4 / 6 | 15 / 13 of 17 | 13.8 / 17.0, 7 | 2 / 4 / 4 |
| 24-48 | 6 | 5 | 4 | 4 / 5 | 24 / 24 of 24 | 22.7 / 24.0, 6 | 1 / 0 / 5 |
| 61-76 | 12 | 6 | 3 | 3 / 6 | 12 / 12 of 15 | 3.8 / 15.0, 6 | 2 / 4 / 6 |

Purity-guarded (strict):

| run (s) | whale numbers | most at once | median per frame | median / max of per-second max | seconds with >= 2 / >= 3 | life median / max (s), numbers >= 3 s | ends: top edge / elsewhere / alive at run end |
|---|---|---|---|---|---|---|---|
| 5-22 | 19 | 5 | 2 | 3 / 5 | 12 / 10 of 17 | 1.3 / 6.3, 5 | 4 / 11 / 4 |
| 24-48 | 17 | 4 | 3 | 3.5 / 4 | 24 / 23 of 24 | 2.5 / 24.0, 6 | 3 / 10 / 4 |
| 61-76 | 23 | 6 | 3 | 3 / 6 | 11 / 10 of 15 | 1.4 / 4.1, 1 | 5 / 12 / 6 |

AI by-eye reference (WP3, every 15th frame, 191 frames, 180 under water): whales besides the tagged one per frame
0: 46, 1: 7, 2: 14, 3: 60, 4: 50, 5: 14 (max 5, so up to 6 animals with the tagged whale; median 3); faint whales
in addition 0-3. So for 56 s of tracked footage 3-4 whales are outlined most of the time and up to 6 at once; 5 of
28 numbers end at the top edge (cleaned), 12 of 59 (guarded); guarded numbers last a median of 1.3-2.5 s, so a
number is not an animal. That is a group of at least 6 seen at one moment, against the notes' G01 sizes of
8 / 8 / 14 (min / avg / max) and 16 actual.

Photos of 2025-12-20: not on disk (confirmed: no folder, no
catalogue entry after 2025-04-11). Notes: G01 CG 08:10-09:35, size min 8 / avg 8 / max 14, actual 16, "Tagged? Yes",
whale 13 "Tagged with green cats via pole - fluke on C1 frame 1722". The clip's burned-in clock reads 11:29:36 at
frame 0, which falls in G03 (Pair, 10:58-11:31), not in G01: either the clip's clock or date, or its link to G01, is
off by about 2 h (not resolved).
