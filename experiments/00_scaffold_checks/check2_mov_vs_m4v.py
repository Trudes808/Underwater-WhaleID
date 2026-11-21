"""WP0 check 2: .mov (tag) vs .m4v (conversion) of recording 00004 at three moments: matched frames, shift, PSNR,
sharpness; side-by-side full-resolution crops for the eye."""
import json, sys
import cv2, numpy as np
from uwid import video, timecode as tc, sheets
V = "/home/sat3737/NOAA_whales/underwater_video/2025-2-3 Mini-mother 11-96/Video/CC-11-96-20250203-101236-000-00004"
MOV, M4V = V + ".mov", V + ".m4v"
gs = tc.load_glyphs()
out = {"moments": []}
tiles, labels, caps = [], [], []
rm, rv = video.VideoReader(MOV), video.VideoReader(M4V)
tab_v = video.frame_table(M4V)
for t_mov in (300.0, 900.0, 1500.0):
    fm = rm.frame_at(t_mov)
    cm = tc.read_clock(fm.image, gs).seconds_of_day
    # m4v time of that clock from the m4v clock fit (WP0 check 3), then best pixel match within +-6 frames
    t_guess = (cm - 36758.08653809216) / 0.9992188215988168
    i0 = tab_v.index_at(t_guess)
    gm = cv2.cvtColor(fm.image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    best = None
    for i in range(max(i0 - 6, 0), min(i0 + 7, tab_v.n_frames)):
        fv = rv.frame_by_index(i)
        gv = cv2.cvtColor(fv.image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        rv_clock = tc.read_clock(fv.image, gs)
        a, b = gm[:1040].ravel(), gv[:1040].ravel()  # leave the clock rows out
        ncc = float(np.corrcoef(a, b)[0, 1])
        same_clock = rv_clock is not None and abs(rv_clock.seconds_of_day - cm) < 1e-6
        key = (same_clock, ncc)
        if best is None or key > best[0]:
            best = (key, i, fv, gv)
    _, i, fv, gv = best
    # level mapping m4v = a * mov + b (least squares), then PSNR after removing it
    A = np.vstack([gm[:1040].ravel(), np.ones(gm[:1040].size)]).T
    coef = np.linalg.lstsq(A, gv[:1040].ravel(), rcond=None)[0]
    mse_raw = float(np.mean((gm[:1040] - gv[:1040]) ** 2))
    mse = float(np.mean((coef[0] * gm[:1040] + coef[1] - gv[:1040]) ** 2))
    shift, resp = cv2.phaseCorrelate(gm[:1040], gv[:1040])
    lap = lambda g: float(cv2.Laplacian(g, cv2.CV_32F).var())
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
    rec = {"t_mov": fm.t, "mov_index": fm.index, "mov_clock": tc.read_clock(fm.image, gs).text,
           "m4v_index": i, "t_m4v": fv.t, "m4v_clock": (tc.read_clock(fv.image, gs) or tc.ClockReading("", 0, None, None, 0, [])).text,
           "index_difference": fm.index - i, "psnr_db_after_level_map": round(psnr, 2), "psnr_db_raw": round(10 * np.log10(255 ** 2 / mse_raw), 2), "level_map_gain_offset": [round(float(coef[0]), 4), round(float(coef[1]), 2)], "ncc": round(best[0][1], 4), "shift_px": [round(shift[0], 3), round(shift[1], 3)],
           "shift_response": round(resp, 3), "laplacian_var_mov": round(lap(gm), 2), "laplacian_var_m4v": round(lap(gv), 2),
           "mean_mov": round(float(gm.mean()), 2), "mean_m4v": round(float(gv.mean()), 2)}
    out["moments"].append(rec)
    print(json.dumps(rec))
    # full-resolution 640x360 crop from the middle and the bottom-right corner, side by side
    for (y, x, what) in ((360, 640, "centre"), (720, 1280, "bottom right")):
        tiles += [fm.image[y:y + 360, x:x + 640], fv.image[y:y + 360, x:x + 640]]
        labels += [f"mov #{fm.index}", f"m4v #{i}"]
        caps += [f"{rec['mov_clock']} {what} 1:1", f"{rec['m4v_clock']} {what} 1:1"]
sheets.make_sheet(tiles, labels, caps, cols=4, tile_width=640, title="00004 .mov (left) vs .m4v (right), 1:1 crops",
                  out="/home/sat3737/NOAA_whales/review_sheets/underwater/wp0_mov_vs_m4v_crops.jpg")
json.dump(out, open("/home/sat3737/NOAA_whales/underwater_cache/wp0_checks/check2_mov_vs_m4v.json", "w"), indent=1)
