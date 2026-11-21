"""Read the clock burned into the picture, e.g. ``20Dec25 11:29:36.600`` at the bottom right.

The overlay is a fixed-pitch font on a black box at a fixed place, so a glyph-template reader is enough:

1. :class:`ClockLayout` cuts the text line into ``n_chars`` character cells (default: 1920x1080 footage of the
   CC camera tags, 20 cells of 8 x 13 px starting at x=1756, y=1062).
2. :func:`bootstrap_glyphs` learns the templates from one video whose first-frame text is known: all cells of all
   frames are clustered, clusters seen in the known frame get their character, and the remaining digits are named
   from the fact that the clock advances (in the tenths and seconds cells each new value is the old one plus one).
   Templates are stored in the cache (``underwater_cache/timecode/glyphs.json``), not in the repo: they are crops of
   footage. Default source: the development clip, so nothing is fitted on a test video.
3. :func:`read_clock` classifies every cell by normalised correlation and parses the text. The month comes from the
   glyphs when its letters are known, else from a date hint (the file name); the time of day needs digits only.
   Returns None when the frame has no readable clock.
4. :func:`read_video_clock` reads a whole video (or a sampling of it), fits clock time against presentation time
   with a robust line fit and flags unreadable frames and outliers.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import names, paths
from . import video as _video

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TEXT_RE = re.compile(r"^(\d\d)([A-Za-z]{3})(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})$")
# character classes per cell for the format "DDMonYY HH:MM:SS.mmm"
FORMAT = "dd" + "LLL" + "dd" + " " + "dd:dd:dd.ddd"
BLANK_STD = 6.0          # a cell whose grey-level std is below this is blank (the space)
DEV_CLIP = "2025-12-20 CG.m4v"
DEV_CLIP_TEXT0 = "20Dec25 11:29:36.600"   # first shown frame of the development clip


@dataclass
class ClockLayout:
    """Where the text line sits, in pixels of a ``ref_width`` x ``ref_height`` frame (scaled for other sizes)."""

    x0: int = 1756
    y0: int = 1062
    pitch: int = 8
    cell_h: int = 13
    n_chars: int = 20
    ref_width: int = 1920
    ref_height: int = 1080

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def cells(self, gray: np.ndarray) -> np.ndarray:
        """Cut the character cells of a grey frame: array (n_chars, cell_h, pitch) float32."""
        h, w = gray.shape[:2]
        x1, y1 = self.x0 + self.pitch * self.n_chars, self.y0 + self.cell_h
        if (w, h) != (self.ref_width, self.ref_height):
            import cv2
            sx, sy = w / self.ref_width, h / self.ref_height
            strip = gray[int(round(self.y0 * sy)):int(round(y1 * sy)), int(round(self.x0 * sx)):int(round(x1 * sx))]
            if strip.size == 0:
                return np.zeros((self.n_chars, self.cell_h, self.pitch), np.float32)
            strip = cv2.resize(strip, (x1 - self.x0, self.cell_h), interpolation=cv2.INTER_AREA)
        else:
            strip = gray[self.y0:y1, self.x0:x1]
        if strip.shape != (self.cell_h, self.pitch * self.n_chars):
            return np.zeros((self.n_chars, self.cell_h, self.pitch), np.float32)
        return strip.reshape(self.cell_h, self.n_chars, self.pitch).transpose(1, 0, 2).astype(np.float32)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return (image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114).astype(np.float32)


def _normalise(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten, remove mean, scale to unit norm. Returns (vectors, std per cell)."""
    v = cells.reshape(len(cells), -1).astype(np.float32)
    std = v.std(axis=1)
    v = v - v.mean(axis=1, keepdims=True)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-6), std


@dataclass
class GlyphSet:
    """Templates (unit vectors) with the character each one stands for; a character may have several."""

    chars: list[str]
    vectors: np.ndarray          # (n_templates, cell_h * pitch)
    layout: ClockLayout = field(default_factory=ClockLayout)
    meta: dict = field(default_factory=dict)

    def classify(self, cells: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Best character and score per cell, and the score of the best *different* character (margin)."""
        v, std = _normalise(cells)
        sims = v @ self.vectors.T                      # (n_cells, n_templates)
        chars = np.array(self.chars)
        best_i = sims.argmax(axis=1)
        best_c = chars[best_i]
        best_s = sims[np.arange(len(v)), best_i]
        other = np.where(chars[None, :] == best_c[:, None], -1.0, sims).max(axis=1)
        out_c = []
        for i, c in enumerate(best_c):
            if std[i] < BLANK_STD:
                out_c.append(" ")
                best_s[i], other[i] = 1.0, 0.0
            else:
                out_c.append(str(c))
        return out_c, best_s, other

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"chars": self.chars, "shape": [self.layout.cell_h, self.layout.pitch],
                "vectors": np.round(self.vectors, 5).tolist(), "layout": self.layout.to_dict(), "meta": self.meta}
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "GlyphSet":
        data = json.loads(Path(path).read_text())
        return cls(chars=data["chars"], vectors=np.array(data["vectors"], np.float32),
                   layout=ClockLayout(**data["layout"]), meta=data.get("meta", {}))


def default_glyph_path() -> Path:
    return paths.cache_dir("timecode") / "glyphs.json"


def load_glyphs(path: Path | None = None, auto_bootstrap: bool = True) -> GlyphSet:
    """Load the glyph templates; if they do not exist yet, learn them from the development clip."""
    path = Path(path) if path else default_glyph_path()
    if path.is_file():
        return GlyphSet.load(path)
    dev = paths.footage_dir() / DEV_CLIP
    if not auto_bootstrap or not dev.is_file():
        raise FileNotFoundError(f"no clock glyphs at {path}; run `uwid timecode --bootstrap` on the development clip")
    gs = bootstrap_glyphs(dev, DEV_CLIP_TEXT0)
    gs.save(path)
    return gs


# ----------------------------------------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class ClockReading:
    text: str
    seconds_of_day: float
    datetime: _dt.datetime | None
    month_source: str | None      # "glyph", "hint" or None
    min_digit_score: float
    scores: list[float]


def parse_text(text: str, date_hint: _dt.date | None = None) -> tuple[float, _dt.datetime | None, str | None] | None:
    m = TEXT_RE.match(text)
    if not m:
        return None
    dd, mon, yy, hh, mi, ss, ms = m.groups()
    hh, mi, ss, ms = int(hh), int(mi), int(ss), int(ms)
    if hh > 23 or mi > 59 or ss > 59:
        return None
    sod = hh * 3600 + mi * 60 + ss + ms / 1000.0
    month = MONTHS.index(mon.capitalize()) + 1 if mon.capitalize() in MONTHS else None
    src = "glyph" if month else None
    day, year = int(dd), 2000 + int(yy)
    if month is None and date_hint is not None:
        month, src = date_hint.month, "hint"
    dt = None
    if month is not None:
        try:
            dt = _dt.datetime(year, month, day, hh, mi, ss, ms * 1000)
        except ValueError:
            dt = None
    return sod, dt, src


def read_clock(image: np.ndarray, glyphs: GlyphSet | None = None, date_hint: _dt.date | None = None,
               min_score: float = 0.6) -> ClockReading | None:
    """Read the burned-in clock of one frame (RGB or grey). None when there is no readable clock.

    Digit, colon and dot cells must match a template with normalised correlation >= ``min_score``. Month letters
    that match no known template (or a weak one) are left to ``date_hint``."""
    glyphs = glyphs or load_glyphs()
    cells = glyphs.layout.cells(to_gray(image))
    chars, scores, _ = glyphs.classify(cells)
    text_chars = []
    digit_scores = []
    for c, s, kind in zip(chars, scores, FORMAT):
        if kind == "L":
            text_chars.append(c if (c.isalpha() and s >= min_score) else "?")
            continue
        if kind == "d":
            if not c.isdigit() or s < min_score:
                return None
            digit_scores.append(float(s))
        elif c != kind or (kind != " " and s < min_score):
            return None
        text_chars.append(c)
    text = "".join(text_chars)
    month_txt = text[2:5]
    if "?" in month_txt or month_txt.capitalize() not in MONTHS:
        text = text[:2] + "Xxx" + text[5:]
    parsed = parse_text(text, date_hint)
    if parsed is None:
        return None
    sod, dt, src = parsed
    if src == "hint" and dt is not None:
        text = text[:2] + MONTHS[dt.month - 1] + text[5:]   # month_source says it came from the hint
    return ClockReading(text=text, seconds_of_day=sod, datetime=dt, month_source=src,
                        min_digit_score=min(digit_scores) if digit_scores else 0.0, scores=[float(s) for s in scores])


# ----------------------------------------------------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------------------------------------------------

def _cluster(vectors: np.ndarray, thr: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Greedy clustering of unit vectors (assign to the nearest centroid with correlation >= thr, else new)."""
    cents: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    counts: list[int] = []
    labels = np.empty(len(vectors), np.int64)
    C = np.zeros((0, vectors.shape[1]), np.float32)
    for i, v in enumerate(vectors):
        if len(C):
            s = C @ v
            j = int(s.argmax())
            if s[j] >= thr:
                labels[i] = j
                sums[j] += v
                counts[j] += 1
                if counts[j] in (2, 4, 8, 16, 64, 256, 1024):
                    c = sums[j] / np.linalg.norm(sums[j])
                    C[j] = c
                continue
        sums.append(v.copy())
        counts.append(1)
        C = np.vstack([C, v[None]])
        labels[i] = len(C) - 1
    cents = [s / np.linalg.norm(s) for s in sums]
    return labels, np.array(cents, np.float32)


def bootstrap_glyphs(path, known_text: str, frame_index: int = 0, layout: ClockLayout | None = None,
                     max_frames: int = 6000, cluster_thr: float = 0.9) -> GlyphSet:
    """Learn glyph templates from video ``path`` whose shown frame ``frame_index`` displays ``known_text``."""
    layout = layout or ClockLayout()
    if len(known_text) != layout.n_chars:
        raise ValueError(f"known text has {len(known_text)} characters, layout has {layout.n_chars}")
    all_cells = []
    for k, fr in enumerate(_video.iter_frames(path, fmt="gray")):
        all_cells.append(layout.cells(fr.image))
        if k + 1 >= max_frames:
            break
    cells = np.stack(all_cells)                       # (F, n_chars, h, w)
    F, P = cells.shape[:2]
    vecs, std = _normalise(cells.reshape(F * P, layout.cell_h, layout.pitch))
    blank = std < BLANK_STD
    labels = np.full(F * P, -1, np.int64)
    lab_nb, cents = _cluster(vecs[~blank], cluster_thr)
    labels[~blank] = lab_nb
    labels = labels.reshape(F, P)
    names_: dict[int, str] = {}
    for p, ch in enumerate(known_text):
        lab = labels[frame_index, p]
        if lab >= 0 and ch != " ":
            names_.setdefault(int(lab), ch)
    # name the remaining digit clusters from the advance of the clock: tenths (17), seconds units (15), tens (14)
    cycles = {17: 10, 15: 10, 12: 10, 14: 6}
    for _ in range(20):
        votes: dict[int, dict[str, int]] = {}
        for p, mod in cycles.items():
            seq = labels[:, p]
            runs = []  # (label, length)
            for lab in seq:
                if runs and runs[-1][0] == lab:
                    runs[-1][1] += 1
                else:
                    runs.append([int(lab), 1])
            runs = [r for r in runs if r[1] >= 2 and r[0] >= 0]
            for (a, _), (b, _) in zip(runs, runs[1:]):
                if a == b:
                    continue
                if a in names_ and names_[a].isdigit() and b not in names_:
                    votes.setdefault(b, {}).setdefault(str((int(names_[a]) + 1) % mod), 0)
                    votes[b][str((int(names_[a]) + 1) % mod)] += 1
                if b in names_ and names_[b].isdigit() and a not in names_:
                    votes.setdefault(a, {}).setdefault(str((int(names_[b]) - 1) % mod), 0)
                    votes[a][str((int(names_[b]) - 1) % mod)] += 1
        if not votes:
            break
        for lab, v in votes.items():
            names_[lab] = max(v.items(), key=lambda kv: kv[1])[0]
    counts = np.bincount(labels[labels >= 0], minlength=len(cents))
    keep = [j for j in range(len(cents)) if j in names_ and counts[j] >= 3]
    gs = GlyphSet(chars=[names_[j] for j in keep], vectors=cents[keep], layout=layout,
                  meta={"source": str(path), "known_text": known_text, "frame_index": frame_index, "n_frames": F,
                        "n_clusters": int(len(cents)), "n_unnamed_clusters": int(len(cents) - len(names_)),
                        "cells_in_unnamed_clusters": int(sum(counts[j] for j in range(len(cents)) if j not in names_)),
                        "created": _dt.datetime.now().isoformat(timespec="seconds")})
    gs.meta["chars_known"] = "".join(sorted(set(gs.chars)))
    return gs


# ----------------------------------------------------------------------------------------------------------------
# whole video
# ----------------------------------------------------------------------------------------------------------------

def robust_line_fit(x: np.ndarray, y: np.ndarray, inlier_s: float = 0.15, iters: int = 10) -> dict:
    """Fit y = a + b x, dropping points further than max(inlier_s, 4 MAD) from the line, repeatedly."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return {"slope": None, "intercept": None, "n_inliers": int(len(x))}
    inl = np.ones(len(x), bool)
    # start from slope 1 and the median offset, which is right for a clock that runs at real time
    b, a = 1.0, float(np.median(y - x))
    for _ in range(iters):
        r = y - (a + b * x)
        mad = float(np.median(np.abs(r[inl] - np.median(r[inl])))) * 1.4826
        new = np.abs(r) <= max(inlier_s, 4 * mad)
        if new.sum() < 2:
            break
        inl = new
        b, a = np.polyfit(x[inl], y[inl], 1)
    r = y - (a + b * x)
    return {"slope": float(b), "intercept": float(a), "n_inliers": int(inl.sum()), "n_outliers": int((~inl).sum()),
            "max_abs_residual_inliers_s": float(np.abs(r[inl]).max()), "max_abs_residual_all_s": float(np.abs(r).max()),
            "rms_residual_inliers_s": float(np.sqrt(np.mean(r[inl] ** 2))), "residuals": r, "inliers": inl}


def read_video_clock(path, fps: float | None = None, glyphs: GlyphSet | None = None,
                     date_hint: _dt.date | None = None, start: float = 0.0, end: float | None = None,
                     progress: bool = False) -> dict:
    """Read the clock on every frame (or ``fps`` frames per second) of a video and fit it against presentation time.

    Returns ``rows`` (index, t, text, clock seconds of day or None, residual, flag) and ``summary``."""
    glyphs = glyphs or load_glyphs()
    if date_hint is None:
        date_hint = names.date_hint(path)
    rows = []
    it = _video.iter_frames(path, fps=fps, start=start, end=end, fmt="gray")
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc=Path(path).name, unit="frame")
    first_dt = None
    for fr in it:
        rd = read_clock(fr.image, glyphs, date_hint)
        if rd is not None and first_dt is None:
            first_dt = rd.datetime
        rows.append({"index": fr.index, "t": fr.t, "text": rd.text if rd else None,
                     "clock_s": rd.seconds_of_day if rd else None,
                     "datetime": rd.datetime.isoformat(timespec="milliseconds") if rd and rd.datetime else None,
                     "month_source": rd.month_source if rd else None,
                     "score": rd.min_digit_score if rd else None})
    ok = [r for r in rows if r["clock_s"] is not None]
    # unwrap midnight
    off = 0.0
    prev = None
    for r in ok:
        if prev is not None and r["clock_s"] + off < prev - 43200:
            off += 86400.0
        r["clock_s"] += off
        prev = r["clock_s"]
    summary: dict = {"path": str(path), "n_frames": len(rows), "n_read": len(ok),
                     "share_read": len(ok) / len(rows) if rows else 0.0, "fps_sampled": fps,
                     "date_hint": date_hint.isoformat() if date_hint else None,
                     "month_sources": sorted({r["month_source"] for r in ok if r["month_source"]})}
    if len(ok) >= 2:
        t = np.array([r["t"] for r in ok])
        c = np.array([r["clock_s"] for r in ok])
        dif = np.diff(c)
        summary["n_backward_steps"] = int((dif < -1e-6).sum())
        summary["monotonic"] = bool((dif >= -1e-6).all())
        fit = robust_line_fit(t, c)
        res, inl = fit.pop("residuals"), fit.pop("inliers")
        summary["fit"] = fit
        for r, e, i in zip(ok, res, inl):
            r["residual_s"] = float(e)
            r["flag"] = "" if i else "outlier"
        summary["first_text"] = ok[0]["text"]
        summary["first_t"] = ok[0]["t"]
        summary["last_text"] = ok[-1]["text"]
        summary["clock_at_t0"] = _fmt_sod(fit["intercept"])
        summary["first_datetime"] = first_dt.isoformat(timespec="milliseconds") if first_dt else None
    for r in rows:
        if r["clock_s"] is None:
            r["flag"] = "unreadable"
            r["residual_s"] = None
    info = names.parse_tag_video_name(path)
    if info and len(ok) >= 2:
        name_sod = (info["start_local"] - info["start_local"].replace(hour=0, minute=0, second=0)).total_seconds()
        summary["name_start"] = info["start_local"].isoformat()
        summary["clock_at_t0_minus_name_start_s"] = float(summary["fit"]["intercept"] - name_sod)
        summary["first_clock_minus_name_start_s"] = float(ok[0]["clock_s"] - name_sod)
    return {"summary": summary, "rows": rows}


def _fmt_sod(s: float | None) -> str | None:
    if s is None:
        return None
    s = s % 86400
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def clock_at(path, times: Sequence[float], glyphs: GlyphSet | None = None) -> list[ClockReading | None]:
    """Clock readings of the frames nearest to ``times`` (random access)."""
    try:
        glyphs = glyphs or load_glyphs()
    except FileNotFoundError:
        return [None] * len(times)
    hint = names.date_hint(path)
    out = []
    with _video.VideoReader(path, fmt="gray") as r:
        for t in times:
            fr = r.frame_at(t)
            out.append(read_clock(fr.image, glyphs, hint) if fr is not None else None)
    return out
