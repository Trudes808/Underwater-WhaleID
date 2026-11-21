"""Dev clip: what do the 163 frames the edit list hides show on the clock? (decode with ignore_editlist)"""
import av, json
from uwid import timecode as tc
gs = tc.load_glyphs()
p = "/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
c = av.open(p, options={"ignore_editlist": "1"})
v = c.streams.video[0]
rows = []
for i, f in enumerate(c.decode(v)):
    rd = tc.read_clock(f.to_ndarray(format="gray"), gs)
    rows.append((i, round(float(f.time), 4), rd.text if rd else None))
print(len(rows), rows[:2], rows[160:166], rows[-4:])
print("unreadable", sum(r[2] is None for r in rows))
json.dump(rows, open("/home/sat3737/NOAA_whales/underwater_cache/wp0_checks/check1b_dev_hidden_frames.json", "w"))
