"""Plots for WP1 (metadata and sensors) -> review_sheets/underwater/metadata_*.png

    .venv/bin/python experiments/01_metadata/make_plots.py "2025-2-3 Mini-mother 11-96"

Needs analyze_sensors.py and sync_video.py outputs. Video frames are mapped to sensor time by the file-name start
time burned-in clock minus the offset measured at the drone drop (tables.md); the clock is 2.03 s ahead of sensor time.
"""
import json
import sys
from pathlib import Path

import av
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid import tagdata as td  # noqa: E402

VIDEO_ROOT = Path("/home/sat3737/NOAA_whales/underwater_video")
OUT = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
# reference palette (dataviz skill, light mode)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "green": "#008300", "violet": "#6250d6", "red": "#e34948", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
     "surface": "#fcfcfb"}
STATE_COL = {"air": "#d9d8d3", "flight": C["violet"], "floating": C["orange"], "whale": C["blue"]}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
                     "ytick.color": C["ink2"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": C["surface"],
                     "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"]})


def to_dt(t, t0_local):
    return t0_local + pd.to_timedelta(np.asarray(t), unit="s")


def main(dep: str):
    out = td.cache_dir(dep)
    meta = td.load_meta(dep)
    t0_local = pd.Timestamp(meta["first_utc"]) + pd.Timedelta(hours=meta["utc_offset_h"])
    df = td.clean(td.load(dep))
    st = pd.read_parquet(out / "states_50hz.parquet")
    seg = pd.read_csv(out / "states.csv")
    surf = pd.read_csv(out / "surfacings.csv")
    drops = pd.read_csv(out / "drops.csv")
    cam = pd.read_csv(out / "camera_log.csv")
    N = json.loads((out / "sensor_numbers.json").read_text())
    fits = pd.read_csv(out / "clock_fits_mov.csv")
    sync = json.loads((out / "sync_summary.json").read_text()) if (out / "sync_summary.json").exists() else {}
    clk_off = sync["clock_minus_sensor_from_drop_s"]["mean"]  # burned clock minus sensor time
    fits["first_frame_t"] = fits.clock_at_pts0_t - clk_off
    t = df.t.to_numpy()
    T = to_dt(t, t0_local)
    d = df.depth_clean.to_numpy() - N["air_depth_pre_deploy"]["median"]
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ whole deployment
    fig, axs = plt.subplots(3, 1, figsize=(15, 8.5), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.2, 1.0]})
    ax = axs[0]
    step = 25
    ax.plot(T[::step], d[::step], color=C["blue"], lw=0.8)
    ax.invert_yaxis()
    ax.set_ylabel("depth (m)")
    for k, r in enumerate(drops.itertuples()):
        ax.axvline(to_dt(r.impact_t, t0_local), color=C["violet"], lw=1.2, ls="--")
        ax.annotate(f"drone drop {k + 1}: {'attached' if r.attached else 'failed, floated'}\n{r.impact_local}",
                    (to_dt(r.impact_t, t0_local), 120 + 25 * (k % 2)), xytext=(-4, 0), textcoords="offset points",
                    fontsize=8, color=C["ink"], ha="right", va="center")
    ax.set_title(f"{dep}: depth, tag state, camera and video files (local time, UTC{meta['utc_offset_h']:+.0f})",
                 loc="left", color=C["ink"])
    # timeline strip
    ax = axs[1]
    rows = {"tag state": 3, "camera recording (log)": 2, "video pieces (log)": 1, "video files on disk": 0}
    for r in seg.itertuples():
        ax.barh(rows["tag state"], (r.end_t - r.start_t) / 86400, left=to_dt(r.start_t, t0_local), height=0.7,
                color=STATE_COL[r.state], edgecolor="none")
    rec = (df.camera.to_numpy() == 12)
    for a, b in td.runs(rec):
        ax.barh(rows["camera recording (log)"], (t[b - 1] - t[a]) / 86400, left=T[a], height=0.7, color=C["aqua"])
    for r in cam.itertuples():
        ax.barh(rows["video pieces (log)"], (r.cam12_end_t - r.cam12_start_t) / 86400, left=to_dt(r.cam12_start_t, t0_local),
                height=0.7, color=C["ink2"])
        ax.text(to_dt((r.cam12_start_t + r.cam12_end_t) / 2, t0_local), rows["video pieces (log)"], f"{r.piece:05d}",
                ha="center", va="center", fontsize=7, color="white")
    for r in fits.itertuples():
        s0 = r.first_frame_t
        ax.barh(rows["video files on disk"], r.duration_s / 86400, left=to_dt(s0, t0_local), height=0.7, color=C["green"])
        ax.text(to_dt(s0 + r.duration_s / 2, t0_local), 0, f"{r.seq:05d}", ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks(list(rows.values()), list(rows.keys()))
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in STATE_COL.values()]
    ax.legend(handles, list(STATE_COL.keys()), ncol=4, loc="upper left", bbox_to_anchor=(0, 1.32), frameon=False, fontsize=8)
    # motion: drone vibration
    ax = axs[2]
    hf = td.rolling_median(df.acc_hf.to_numpy(np.float64), 50)
    ax.semilogy(T[::step], hf[::step], color=C["violet"], lw=0.7)
    ax.axhline(1.5, color=C["ink2"], lw=0.8, ls=":")
    ax.set_ylabel("acc. vibration\n(m/s², 1 s median)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    fig.savefig(OUT / "metadata_deployment.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------ one-hour zoom
    a_t = drops[drops.attached].impact_t.iloc[0] - 120
    b_t = a_t + 3600
    m = (t >= a_t) & (t <= b_t)
    fig, axs = plt.subplots(3, 1, figsize=(15, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    axs[0].plot(T[m], d[m], color=C["blue"], lw=0.8)
    axs[0].invert_yaxis()
    axs[0].set_ylabel("depth (m)")
    ss = surf[(surf.start_t > a_t) & (surf.end_t < b_t)]
    for r in ss.itertuples():
        axs[0].axvspan(to_dt(r.start_t, t0_local), to_dt(r.end_t, t0_local), color=C["orange"], alpha=0.5, lw=0)
    axs[0].set_title(f"One hour after the drone drop: depth with sensor surfacings (orange, n={len(ss)}), light and rotation",
                     loc="left", color=C["ink"])
    axs[1].semilogy(T[m], df.light2_clean.to_numpy()[m], color=C["yellow"], lw=0.7)
    axs[1].axhline(td.LIGHT_SAT, color=C["ink2"], lw=0.8, ls=":")
    axs[1].set_ylabel("light 2 (raw)")
    axs[2].plot(T[m], td.rolling_median(df.gmag.to_numpy(), 50)[m], color=C["green"], lw=0.7)
    axs[2].set_ylabel("|gyro| (mrad/s)")
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    fig.savefig(OUT / "metadata_zoom_1h.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------ single surfacings
    ev = pd.read_csv(out / "sync_events_mov.csv")
    frames = pd.read_parquet(out / "sync_frames_mov.parquet")
    audio = pd.read_parquet(out / "audio_levels.parquet") if (out / "audio_levels.parquet").exists() else None
    good = ev.dropna(subset=["pic_exit_pts"])
    pick = good.iloc[np.linspace(0, len(good) - 1, min(6, len(good))).round().astype(int)]
    for i, e in enumerate(pick.itertuples()):
        s = surf[surf.surfacing_id == e.surfacing_id].iloc[0]
        f = fits[fits.file == e.file].iloc[0]
        vid0 = f.first_frame_t                # sensor time of pts 0 (burned clock - offset)
        w0, w1 = s.start_t - 6, s.end_t + 6
        m = (t >= w0) & (t <= w1)
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(6, 8, height_ratios=[2, 1, 1, 1, 1.6, 1.6])
        axd = fig.add_subplot(gs[0, :])
        axd.plot(T[m], d[m], color=C["blue"], lw=1.2, drawstyle="steps-post")
        axd.invert_yaxis()
        axd.set_ylabel("depth (m)")
        axd.axhline(0.10, color=C["ink2"], ls=":", lw=0.8)
        for ax in [axd]:
            ax.axvspan(to_dt(s.start_t, t0_local), to_dt(s.end_t, t0_local), color=C["orange"], alpha=0.25, lw=0)
        axd.set_title(f"Surfacing {int(s.surfacing_id)} at {s.start_local} local, {e.file}: sensor exposure (orange band) "
                      f"vs picture in air (green band)", loc="left", color=C["ink"])
        axl = fig.add_subplot(gs[1, :], sharex=axd)
        axl.plot(T[m], df.light1_clean.to_numpy()[m], color=C["yellow"], lw=1, label="light 1")
        axl.plot(T[m], df.light2_clean.to_numpy()[m], color=C["magenta"], lw=1, label="light 2")
        axl.set_ylabel("light (raw)")
        axl.legend(loc="upper left", fontsize=7, frameon=False)
        fk = frames[(frames.file == e.file) & (frames.surfacing_id == e.surfacing_id)]
        axs_ = fig.add_subplot(gs[2, :], sharex=axd)
        axs_.plot(to_dt(vid0 + fk.pts, t0_local), fk.sat, color=C["green"], lw=1)
        axs_.axhline(90, color=C["ink2"], ls=":", lw=0.8)
        axs_.set_ylabel("picture\nsaturation")
        axa = fig.add_subplot(gs[3, :], sharex=axd)
        if audio is not None:
            ak = audio[(audio.file == e.file) & (audio.pts >= (w0 - vid0)) & (audio.pts <= (w1 - vid0))]
            axa.plot(to_dt(vid0 + ak.pts, t0_local), ak.rms_db, color=C["violet"], lw=1)
        axa.set_ylabel("sound\n(dBFS)")
        for ax in [axl, axs_, axa, axd]:
            if not np.isnan(e.pic_exit_pts):
                ax.axvspan(to_dt(vid0 + e.pic_exit_pts, t0_local), to_dt(vid0 + e.pic_entry_pts, t0_local),
                           color=C["green"], alpha=0.15, lw=0)
            ax.axvspan(to_dt(s.start_t, t0_local), to_dt(s.end_t, t0_local), color=C["orange"], alpha=0.12, lw=0)
        axa.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        for ax in [axd, axl, axs_]:
            plt.setp(ax.get_xticklabels(), visible=False)
        # frame strip: 16 frames evenly over the window
        times = np.linspace(w0 + 2, w1 - 2, 16)
        ims = grab(VIDEO_ROOT / dep / "Video" / e.file, times - vid0)
        for k, (tt, im) in enumerate(zip(times, ims)):
            ax = fig.add_subplot(gs[4 + k // 8, k % 8])
            ax.imshow(im[..., ::-1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(str(to_dt(tt, t0_local))[11:21], fontsize=7, color=C["ink2"])
        fig.tight_layout()
        fig.savefig(OUT / f"metadata_surfacing_{i + 1}.png", dpi=110)
        plt.close(fig)
    print("plots written to", OUT)


def grab(video, pts_list):
    c = av.open(str(video))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    out = []
    for p in pts_list:
        c.seek(int(max(p, 0) / s.time_base), stream=s, backward=True)
        im = None
        for fr in c.decode(s):
            if fr.time >= p:
                im = fr.to_ndarray(format="bgr24", width=320, height=180)
                break
        out.append(im if im is not None else np.zeros((180, 320, 3), np.uint8))
    c.close()
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-2-3 Mini-mother 11-96")
