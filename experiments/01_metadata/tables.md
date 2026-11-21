# 01_metadata: what the tag's own sensor log says about "underwater", "surfaced" and "not on a whale"

Deployment `2025-2-3 Mini-mother 11-96` (camera tag CC-11-96, 2025-02-03, local = UTC-10). One log, one tag, one
day: every number below is one run on this deployment. Videos used: the `.mov` pieces 00004-00008 that were fully
copied at 19:41 (5 of 14 pieces; 00009 onwards were still arriving) and the matching `.m4v` conversions.
All times below are local unless marked UTC. `t` = seconds since the first CSV sample (08:36:52.000 local).

Scripts (run in this order, all from the repo root with `.venv/bin/python`):
`build_cache.py` (one pass over the 3.1 GB CSV, 143 s) -> `analyze_sensors.py` -> `probe_videos.py` ->
`sync_video.py --suffix .mov` / `--suffix .m4v` -> `summarize_sync.py` -> `sound_check.py` -> `video_states.py` ->
`make_plots.py`. Clock reader: `clock_ocr.py` (stop-gap; `uwid.timecode` from WP0 now exists and should replace it;
its digit templates were built from frames of 00004.mov, a test video, which only affects OCR, not any measure here).
Library code: `src/uwid/tagdata.py`. Cache: `underwater_cache/tagdata/2025-2-3 Mini-mother 11-96/` (56 MB).

## 1. Channels

Real rates = value changes per second over the whole log (6.74 h, 9,703,339 rows, no time gaps); a slow
channel that does not change between two samples is undercounted, so read rates as lower bounds.

| CSV column | real rate | what it is / what it gives here |
|---|---|---|
| Accelerometer X/Y/Z | 400 Hz | clips at +-39.2 m/s2 per axis (about 4 g). Gives drone vibration (rms around the 20 ms mean 1.8-9.4 m/s2 in flight vs < 1 otherwise), free fall, impacts, wave motion, handling. |
| Gyroscope X/Y/Z | ~100 Hz (changes every 4 rows) | median |gyro| 23 mrad/s on the whale below 5 m, 159 at the surface, 780 when floating free (table 5). |
| Magnetometer X/Y/Z | ~50 Hz | heading; not used here. |
| Depth (100bar) | 10 Hz | step 0.0311 m (resolution). In air on deck: median -0.062 m, sd 0.048, 1-99 % [-0.156, 0.000] (994 s quiet deck). 27 isolated readings of 1529.54 m (all with the tag in air) are glitches; replaced by interpolation. Drift: air reading during surfacings -0.031 m early, -0.062 m late (slope -0.005 m/h). After recovery (15:08 on) the sensor reads down to -3.36 m (1 % quantile -1.36 m) for minutes: unexplained, not altitude (that would be 3 km); treat depth on deck after recovery as unreliable. |
| Temperature, Temperature (depth), Temp. (magnet.) | ~9 Hz | water 25.9-27.8 C; the tag heats to 34 C in the sun on deck, so a cooling step marks water entry but with a time constant of minutes. `Temperature (depth)` has a -51.2 C glitch. |
| Light intensity 1, 2 | ~8-10 Hz | both clip at 37177 in daylight air (65535 = invalid, ~200 50-Hz bins, 09:39-09:43 on deck). Light 2 is 37177 in 100 % of the samples with the tag exposed on the whale, but already from about 1.1 s before the depth trace says exposed until 1.1 s after (median, n=80): it says "within ~0.5 m of the surface in daylight", not wet/dry. Light 1 below ~200 switches the LED on. |
| Hydrophone [raw] | 0.7 Hz | values 7-47, same distribution in air, floating, at the surface and at depth, no correlation with flow (r = -0.004 with vibration): no information found. |
| GPS 1, GPS 2 | — | date / time of the last GPS snapshot. |
| GPS 3 | 20 Hz | ramps 0-2500 then resets: guess, a GPS acquisition timer in ms. |
| GPS 4 | 0.03 Hz | 0-7: guess, satellites in the last snapshot (but it reads 1 at 20 m depth too). No positions in the CSV. |
| GPS | — | always `F`. The `.ubx` holds 242,521 RXM-RAWX raw-measurement and 242,519 NAV-TIMEGPS messages, 54 RXM-SFRBX, no position messages: positions need post-processing (RTKLIB-type), not done. |
| System error | — | 0x00000000 throughout. |
| BATT V / mA / mAh | ~0.1 Hz | 4.17 -> 3.80 V; -45 mA with the camera off, -340 to -410 mA with it on; 6662 -> 4180 mAh (2482 mAh used). |
| Camera | event | 0 off, 11 starting (about 22 s), 12 recording, 20/21/22 closing a file (20 only on every other roll), then 0 or 11 again. |
| CC status | event | `R--` while the camera controller is recording (from 15 s after Camera=12 until about 10 s after Camera leaves 12), `---` otherwise. The CATS toolbox renames it `CamOn` (importCATSdata.m). |
| CC vid. size [kBytes] | ~0.2 Hz | size of the file being written; resets at each roll. Last logged values 160,468 / 214,986 / 262,534 / 270,359 kB vs .mov sizes 161,657 / 216,220 / 268,035 / 271,915 kB for 00004-00007 (the log stops ~5 s before the file closes): it is the `.mov`. |
| Camera time | 1 Hz | counts recorded seconds: +1 every second (ticks at x.98 s), pauses 23-24 s at every file roll, 23,640 at the end. Resumes 6.87 s after a file's first frame (5 files, spread 0.08 s). Toolbox name `CamTime`. |
| Flags | event | only `---` and `-T-`. `T` on from 08:41:52 (camera start) with 1-2 s drops every ~15 min, at every file roll and halfway between. Guess: camera link / time sync; not wet/dry. |
| LED | event | 1 when Light 1 < ~200 raw (on at 192-199, off at 222-294): 5 times at 88-106 m depth, 0.5-3 min each, plus 1 s on deck at 09:20:22. |
| CPU temp. | 0.2 Hz | 25-36 C. |
| SWS (salt-water switch) | — | listed under "available sensors" in the `.txt` but NOT under "activated sensors" and absent from the CSV: **this log has no wet/dry signal**. Wet/dry here is inferred from depth (and light, motion). |

Sources read: CATS toolbox `CATSMatlabTools/importCATSdata.m` (header mapping: `CC status`->CamOn, `Camera time`->CamTime,
`CC vid. size`->VidSize, `SWS`, Flags / System error / GPS dropped on import), `WorkhorseFunctions/gettagon.m` (tag-on
guess = depth > 1 m, then by hand), `Subfunctions/catsspecific/findsurfacings.m` (surfacings = depth minima within 1 m
of the local minimum), github.com/cadede/CATS-Methods-Materials; Cade et al. 2021, Anim. Biotelemetry 9:34
(doi 10.1186/s40317-021-00256-w): video synchronised by OCR of the clock burned into the lower right of each frame,
a camera-firmware processing delay once found, start-trigger delays and skipped frames as known errors; cameras can be
set to stop below a light threshold; tag-on / off chosen by hand. The meanings of Flags `T`, GPS 3 / 4 and the
Camera state codes are my reading of the data, not documented anywhere I found.

## 2. Timeline from the sensors alone

Rules (`tagdata.tag_states`, thresholds from this log): **flight** = 1 s median of acceleration vibration > 1.5 m/s2
for >= 20 s (drone motors: 1.8-9.4 vs < 1.0 for 95 % of everything else); **floating** = over 10 s: median |gyro|
> 350 mrad/s, median vibration < 0.12, no knock > 16 m/s2, depth max < 0.35 m and median >= air + 0.04 m (port wet);
**whale** = between samples deeper than 1 m not separated by flight / floating; **air** = the rest (deck, hand).
Drops: free fall = |a| (50 ms mean per axis) < 0.35 g for >= 0.1 s, impact = first raw |a| > 2.5 g after it.

| from | to | state | video |
|---|---|---|---|
| 08:36:52 | 09:06:11 | air (deck; camera on from 08:41:54) | piece 1 |
| 09:06:11 | 09:14:59 | drone flight 1 (8.8 min) | 1-2 |
| 09:14:57.22 | 09:14:58.15 | **drop 1**: free fall 0.93 s (about 4.3 m), impact clipped at 4 g; max depth in the next 10 min 0.78 m (handling) | 2 |
| 09:14:59 | 09:17:29 | floating (failed drop, 2.5 min), then recovered | 2 |
| 09:30:34 | 09:35:41 | drone flight 2 (5.1 min) | 2 |
| 09:35:40.63 | 09:35:41.59 | **drop 2**: free fall 0.96 s (about 4.5 m); max depth 10 min 0.53 m | 2 |
| 09:35:47 | 09:38:00 | floating (failed drop, 2.2 min), then recovered | 2 |
| 09:59:30 | 10:01:41 | floating 2.2 min with no flight before it: tag in the water by hand (soak / float test?) | 3 |
| 10:07:33 | 10:14:07 | drone flight 3 (6.6 min) | 3-4 |
| 10:14:06.948 | 10:14:07.785 | **drop 3**: free fall 0.84 s (about 3.4 m), impact; depth > 1 m 2.4 s after impact, 117 m 5 min later | 4 |
| 10:14:07.8 | 14:38:44 | **on the whale** 4 h 24.6 min | 4-12 |
| 14:38:44 | 14:39:15 | tag released at 21.3 m (|gyro| jumps from a 19 mrad/s baseline; the whale was resting at 21-24 m with no motion), rises at 0.68 m/s with rotation | 12 |
| 14:39:15 | 15:04:48 | floating 25.5 min (orientation y-axis up, as in the failed drops) | 12-13 |
| 15:04:48 | 15:21:10 | recovered, deck; camera still recording at log end | 13-14 |

Diving record on the whale (`dives.csv`): 86 excursions deeper than 1 m, 29 deeper than 10 m, max 177.3 m. Two dive
types: long resting dives at 23-31 m (8.9-20.9 min) and short deep dives to 45-177 m (2.3-5.8 min).

## 3. Surfacings (`surfacings.csv`, truth for WP4)

Definition: "surfaced" = the pressure port is out of the water. On the whale the depth histogram has a separate peak
at the deck air level (-0.14 to -0.01 m raw, 292 s in total) and a minimum at +0.05-0.11 m, so the threshold is
air reading + 0.10 m (raw 0.04 m). Re-immersions shorter than 1 s that stay above 1 m are merged (8 of 80 events
have 2-3 parts). A "bout" is the enclosing stretch shallower than 1 m (several breaths per bout).

Result: **80 surfacings in 72 bouts**, duration 3.0 / 3.6 / 4.4 s (5 / 50 / 95 %), none shorter than 1 s; ascent
over the last metre 0.55 m/s, descent 0.65 m/s (medians). 78 lie inside a video file, 2 fall in the 24 s gap
between pieces 6 and 7 (11:42:33, 11:42:46). Per piece: 4:5, 5:11, 6:14, 7:10, 8:10, 9:7, 10:11, 11:8, 12:4.

Columns: `surfacing_id`; `start_t`, `end_t` (s since first sample, threshold crossings interpolated); `start_utc`,
`end_utc`, `start_local`, `end_local`; `duration_s`; `n_parts`; `min_depth_m` (above air); `bout_id`,
`bout_start_t`, `bout_end_t` (1 m crossings); `ascent_mps`, `descent_mps`; `prev_gap_s`, `prev_max_depth_m` (the dive
before); `light2_sat_start_t`, `light2_sat_end_t`; `air_reading_m` (raw depth while exposed: offset drift),
`temp_c`; `piece` (video sequence number), `in_video`, `first_frame_t` (sensor time of that file's first frame),
`pts_exit_pred`, `pts_entry_pred` (expected .mov presentation times of lens-out / lens-in, see 4), `pts_source`
(`burned clock` for files read, else `camera time`), `pic_exit_pts`, `pic_entry_pts` (automatic saturation cue,
files 00004-00008 only; biased, see 4).

Timing accuracy, sensor side: depth is sampled at 10 Hz with a 0.031 m step and crosses the threshold at ~0.6 m/s,
so a crossing is known to about 0.1 s. Against the picture (AI by-eye reading of frame strips, 9 exits and 9
entries spread over 00004-00008, see below): the lens is fully out of the water about **0.3 s after** `start_t`
(range of the 9: +0.0 to +0.5 s, with 0.2-0.5 s of water film / splash before) and the picture is clearly under water
again about **0.4 s before** `end_t` (range -0.2 to -0.6 s, splash starting ~0.8 s before). Use `start_t + 0.3` /
`end_t - 0.4` for "lens in air" and expect +-0.3 s; frames within ~0.5 s of either end are `crossing`.

## 4. Camera and time

Camera in the log: on at 08:41:54 (Camera 11), first piece from about 08:42:08, recording without a stop until the
log ends at 15:21:10; 14 pieces of 29 min 41 s (1780.4-1781.4 s) separated by 23-24 s gaps (1.3 % of the time not
filmed); piece 14 is cut by the end of the log (449 s logged). Files on disk agree: sequence numbers 00004-00008
match pieces 4-8, the `.mov` durations are 1780.4-1781.4 s, sizes match `CC vid. size`. Pieces 00001-00003
(08:42-10:12) and 00009-00014 were not on disk when this ran.

Ways to map sensor time to a video's presentation time (pts), `.mov`:

| method | sensor time of the first frame | accuracy | evidence |
|---|---|---|---|
| (c) burned-in clock - 2.03 s | clock fit per file: clock = a + b pts, rms 0.009 s, drift -0.7 to +2.5 ppm (5 files, 60 probes x 12 frames each) | +-0.05 s | drone drop in 00004: release frame (pts 90.83) and splash frame (91.63) vs accelerometer free-fall start and impact: clock - sensor = 2.05 and 2.01 s; picture fall 0.80 s vs accelerometer 0.84 s |
| (b) `Camera time` resumes - 6.87 s | from the log alone | +-0.1 s (spread of the 5 files 0.08 s) | calibrated on 00004-00008 against (c) |
| (a) file name (local start) | name time | -0.14, -0.05, +0.89, -0.12, -0.10 s off (5 files) | up to ~1 s wrong (00006) |
| Camera state 11 / 12 | first frame = Camera 11 + ~16.2 s = Camera 12 - 5.8 s | ~1 s | 5 files |

The burned clock (`03Feb25 10:22:38.200`, bottom right, 0.1 s steps) is present in both the tag's `.mov` and the
`.m4v` conversions, but the clock is **2.03 s ahead of the sensor log**. The `.m4v` conversions drop / retime frames:
their pts run at -779, -807, +29, -361, +31 ppm against the clock (up to 1.4 s at the end of a file), so `.m4v`
needs its own clock fit (`clock_fits_m4v.csv`) and cannot use (a) or (b).

(d) Physical check, 48 surfacings in 00004-00008, picture side = automatic cue (mean HSV saturation of the frame <
90, underwater frames 150-220, frames in air 20-60), mapped with the clock: clock - depth at exit 1.61 s (q10-q90
0.76-1.99, MAD 0.29), at entry 1.79 s (1.59-2.02, MAD 0.10); per-file medians at entry 1.75-1.92 s, no trend over
2.5 h. The saturation cue fires early at exits (frames looking up at the bright surface from just below are already
desaturated; 6 of the 6 largest outliers checked by eye) - so the automatic cue confirms the offset to ~0.2 s and
shows no drift, and the by-eye reading in 3 gives the picture-vs-depth lags. File-name mapping instead: MAD 0.31 /
0.17 s and one file 0.9 s off.

## 5. Sound

The `.mov` sound track (AAC, mono, 32 kHz) is empty in this deployment: -77.4 dBFS rms, 1-99 % range 1 dB (-77.7 to
-76.7) over each whole 30-min file, peak < 0.006 full scale, identical in air and water (medians -77.47 vs -77.44 dB).
Threshold test per band (20-200 Hz ... 8-16 kHz, 0.25 s frames; threshold chosen on 00004, tested on 00005-00008,
615 air vs 26,848 water frames): balanced accuracy 0.51-0.60, chance 0.5. **Sound cannot separate underwater from
surfaced here**, because nothing was recorded (microphone off or not connected; the .m4v stereo track is the same
noise floor). Whether other deployments record sound is open.

## 6. "Not on a whale" from the sensors

Feature quantiles (5 / 50 / 95 %) per state:

| state | seconds | raw depth (m) | vibration (m/s2) | |gyro| (mrad/s) | light 2 |
|---|---|---|---|---|---|
| air (deck, hand) | 5183 | -0.12 / -0.06 / 0.00 | 0.07 / 0.21 / 1.0 | 22 / 62 / 925 | 1036 / 9029 / 37177 |
| drone flight | 1237 | -0.25 / -0.06 / -0.03 | 1.75 / 3.07 / 9.4 | 26 / 74 / 568 | 2149 / 6495 / 8421 |
| floating free | 1935 | -0.03 / 0.03 / 0.12 | 0.04 / 0.06 / 0.11 | 278 / 782 / 1602 | 189 / 1108 / 17157 |
| whale, port exposed | 292 | -0.12 / -0.06 / 0.00 | 0.04 / 0.10 / 0.85 | 62 / 159 / 321 | 37177 all |
| whale, 0-1 m | 361 | 0.12 / 0.59 / 0.97 | 0.04 / 0.07 / 0.22 | 27 / 77 / 229 | 19825 / 35305 / 37177 |
| whale, > 5 m | 14378 | 12 / 24 / 93 | 0.03 / 0.04 / 0.12 | 16 / 23 / 112 | 6 / 480 / 1125 |

Video recorded per state (sensor time, pieces from `Camera time`, `video_state_seconds.csv` and
`video_state_spans.csv` with pts ranges per piece): total 23,605 s (6.56 h); on the whale 15,674 s (66.4 %); **not on
a whale 7,931 s (33.6 %)**: deck 4,802 s, drone flight 1,175 s, floating 1,954 s. Pieces 1-3 and 13-14 contain no
whale time at all; piece 4 starts with 92 s of flight, piece 12 ends with 265 s floating.

Successful vs failed drop: after a successful drop the depth passes 1 m within 2.4 s and the tag shows the whale's
dive profile; after both failed drops it stayed at -0.03..0.12 m with wave-driven rotation (|gyro| ~780 mrad/s) and the
float orientation until recovered. 3 drops, 3 correct with "deeper than 1 m within 10 min" and with the floating
rule: only n = 3, so the rule is plausible, not measured. Risks not covered by this log: a whale that stays at the
surface for minutes after tagging (whale at the surface: |gyro| 62-321 vs floating 278-1602, overlapping tails;
use 10 s medians and the float orientation), a tag slipping without coming off, night (light rules), a tag that
comes off at the surface (no ascent signature; the floating rule still applies).

What has to come from the picture alone when there is no sensor log (e.g. `2025-12-20 CG.m4v`): underwater /
crossing / air per frame and surfacing times (no depth); whether the tag is on a whale (carrier's back fixed in
frame vs open sea / boat deck / drone legs / hands); drone flight and drop (drone legs in the top of the frame,
horizon, splash); floating free (horizon rolling with waves, waterline across the lens, no carrier back); deck tests;
darkness / LED; and the time base: the burned-in clock gives camera time to 0.1 s but its offset to any other clock
(2.03 s on this tag) is unknown without a sensor log. Diving depth and dive phase cannot be recovered except roughly
from colour and light.
