"""Decode each test clip; compare its first / last frame with the source frames the index names (clock + NCC)."""
import csv, json, sys
import numpy as np
from uwid import video, timecode as tc
gs = tc.load_glyphs()
rows = list(csv.DictReader(open(sys.argv[1])))
res = []
for r in rows:
    clip, src = r["clip_file"], r["source"]
    tab = video.frame_table(clip)
    frames = list(video.iter_frames(clip, fmt="gray"))
    s0, s1 = int(r["start_frame"]), int(r["end_frame"])
    with video.VideoReader(src, fmt="gray") as rs:
        first_src, last_src = rs.frame_by_index(s0), rs.frame_by_index(s1 - 1)
        before = rs.frame_by_index(s0 - 1) if s0 > 0 else None
    ncc = lambda a, b: float(np.corrcoef(a.astype(np.float32).ravel(), b.astype(np.float32).ravel())[0, 1])
    rd = lambda img: (tc.read_clock(img, gs) or tc.ClockReading("-", 0, None, None, 0, [])).text
    out = {"clip": clip.rsplit("/", 1)[1], "mode": r["mode"], "frames_in_clip": len(frames), "index_says": s1 - s0,
           "hidden_in_clip": tab.n_hidden, "clip_first_t": frames[0].t, "clip_duration_s": round(tab.duration, 4),
           "src_start_s": r["start_s"], "wanted_start_s": r["wanted_start_s"],
           "clock_clip_first": rd(frames[0].image), "clock_src_first": rd(first_src.image),
           "clock_clip_last": rd(frames[-1].image), "clock_src_last": rd(last_src.image),
           "ncc_first_vs_src_first": round(ncc(frames[0].image, first_src.image), 5),
           "ncc_first_vs_src_previous": round(ncc(frames[0].image, before.image), 5) if before else None,
           "ncc_last_vs_src_last": round(ncc(frames[-1].image, last_src.image), 5)}
    res.append(out); print(json.dumps(out))
json.dump(res, open("/home/sat3737/NOAA_whales/underwater_cache/wp0_checks/check4_verify_clips.json", "w"), indent=1)
