"""Read the clock burned into the picture, e.g. ``20Dec25 11:29:36.600`` at the bottom right.

Two layouts are known (:data:`FORMATS`), and the reader picks the one a video carries by itself
(:func:`glyphs_for`):

- ``DDMonYY``: ``20Dec25 11:29:36.600``, tag CC-11-96 and the development clip, 1920x1080, fixed pitch, tenths;
- ``DD/MM/YY``: ``02/01/25 15:45:02.475``, tag CC-09-85, 3840x2160, proportional font (digits of equal width),
  milliseconds (in 10 ms steps, stamped with about +-15 ms jitter against the frame times).

The overlay is white text on a black box at a fixed place, so a glyph-template reader is enough:

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

A video whose clock none of the stored glyph sets reads gets its own: :func:`find_layout` locates the black box
(always dark over frames spread through the video) and the characters in it (columns of white text), matches them
to a format in :data:`FORMATS`, and :func:`bootstrap_from_name` learns the digit templates from the video itself,
naming each cell from the clock the file name implies (start time + presentation time; only cells whose digit is
certain under the remaining uncertainty are used, which shrinks from 4 s to the measured fit in three passes).
Templates are stored in the cache as ``glyphs_<format>_<W>x<H>.json``.
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
DMY_RE = re.compile(r"^(\d\d)/(\d\d)/(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})$")
# character classes per cell for the format "DDMonYY HH:MM:SS.mmm"
FORMAT = "dd" + "LLL" + "dd" + " " + "dd:dd:dd.ddd"
# known layouts: cell classes (d digit, L letter, others literal) and the finest step the clock shows
FORMATS = {"DDMonYY": {"pattern": FORMAT, "resolution_s": 0.1},
           "DD/MM/YY": {"pattern": "dd/dd/dd dd:dd:dd.ddd", "resolution_s": 0.001}}
BLANK_STD = 6.0          # a cell whose grey-level std is below this is blank (the space)
DEV_CLIP = "2025-12-20 CG.m4v"
DEV_CLIP_TEXT0 = "20Dec25 11:29:36.600"   # first shown frame of the development clip


@dataclass
class ClockLayout:
    """Where the text line sits, in pixels of a ``ref_width`` x ``ref_height`` frame (scaled for other sizes).

    Fixed-pitch layouts (``centers`` None): ``n_chars`` cells of ``pitch`` x ``cell_h`` from (x0, y0). Proportional
    layouts: one cell of ``cell_w`` x ``cell_h`` centred on each of ``centers`` (x, one per character of the
    format), read at every shift up to ``shift`` px (the centres come from the text's columns, to about 1 px)."""

    x0: int = 1756
    y0: int = 1062
    pitch: int = 8
    cell_h: int = 13
    n_chars: int = 20
    ref_width: int = 1920
    ref_height: int = 1080
    fmt: str = "DDMonYY"
    centers: list | None = None
    cell_w: int = 0
    shift: int = 0
    date_order: str = "DMY"           # DD/MM/YY: day first (CC-09-85) or "MDY"

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @property
    def pattern(self) -> str:
        return FORMATS[self.fmt]["pattern"]

    @property
    def resolution_s(self) -> float:
        return FORMATS[self.fmt]["resolution_s"]

    def cell_stack(self, gray: np.ndarray) -> np.ndarray:
        """Cells at every shift: array (n_shifts, n_chars, cell_h, cell_w) float32 (one shift for fixed pitch)."""
        if self.centers is None:
            return self.cells(gray)[None]
        import cv2
        h, w = gray.shape[:2]
        s, cw, ch = self.shift, self.cell_w, self.cell_h
        lefts = [int(round(c - cw / 2)) for c in self.centers]
        X0, X1 = min(lefts) - s, max(lefts) + cw + s
        Y0, Y1 = self.y0 - s, self.y0 + ch + s
        if (w, h) != (self.ref_width, self.ref_height):
            sx, sy = w / self.ref_width, h / self.ref_height
            strip = gray[max(int(round(Y0 * sy)), 0):int(round(Y1 * sy)), max(int(round(X0 * sx)), 0):int(round(X1 * sx))]
            if strip.size == 0:
                return np.zeros(((2 * s + 1) ** 2, len(lefts), ch, cw), np.float32)
            strip = cv2.resize(strip, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_AREA)
        else:
            strip = gray[max(Y0, 0):min(Y1, h), max(X0, 0):min(X1, w)]
        if strip.shape != (Y1 - Y0, X1 - X0):      # the box touches the frame edge: pad with black
            if strip.size == 0:
                return np.zeros(((2 * s + 1) ** 2, len(lefts), ch, cw), np.float32)
            top, left = max(-Y0, 0), max(-X0, 0)
            strip = np.pad(strip, ((top, Y1 - Y0 - strip.shape[0] - top), (left, X1 - X0 - strip.shape[1] - left)))
        strip = strip.astype(np.float32)
        out = np.empty(((2 * s + 1) ** 2, len(lefts), ch, cw), np.float32)
        k = 0
        for dy in range(-s, s + 1):
            for dx in range(-s, s + 1):
                for i, L in enumerate(lefts):
                    x = L - X0 + dx
                    out[k, i] = strip[s + dy:s + dy + ch, x:x + cw]
                k += 1
        return out

    def cells(self, gray: np.ndarray) -> np.ndarray:
        """Cut the character cells of a grey frame: array (n_chars, cell_h, pitch) float32 (unshifted)."""
        if self.centers is not None:
            st = self.cell_stack(gray)
            return st[len(st) // 2]
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
        """Best character and score per cell, and the score of the best *different* character (margin).

        ``cells`` (n, h, w) or, for shifted reading, (n_shifts, n, h, w): the best shift per cell and template."""
        if cells.ndim == 4:
            S, n = cells.shape[:2]
            v, std = _normalise(cells.reshape(S * n, *cells.shape[2:]))
            sims = (v @ self.vectors.T).reshape(S, n, -1).max(axis=0)
            std = std.reshape(S, n)[S // 2]
            v = v[:n]
        else:
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
    if glyphs.layout.centers is not None:
        return _read_proportional(image, glyphs, date_hint, min_score)
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


def parse_dmy(text: str, order: str = "DMY") -> tuple[float, _dt.datetime | None, str | None] | None:
    """Parse ``02/01/25 15:45:02.475`` (day first; ``order="MDY"`` for month first)."""
    m = DMY_RE.match(text)
    if not m:
        return None
    a, b, yy, hh, mi, ss, ms = (int(x) for x in m.groups())
    if hh > 23 or mi > 59 or ss > 59:
        return None
    day, month = (a, b) if order == "DMY" else (b, a)
    try:
        dt = _dt.datetime(2000 + yy, month, day, hh, mi, ss, ms * 1000)
    except ValueError:
        dt = None
    return hh * 3600 + mi * 60 + ss + ms / 1000.0, dt, "glyph" if dt else None


def _read_proportional(image: np.ndarray, glyphs: GlyphSet, date_hint, min_score: float) -> ClockReading | None:
    """Proportional layouts: only digit cells are classified (separators sit at fixed places and are not read)."""
    lay = glyphs.layout
    chars, scores, _ = glyphs.classify(lay.cell_stack(to_gray(image)))
    out, digit_scores = [], []
    for c, sc, kind in zip(chars, scores, lay.pattern):
        if kind == "d":
            if not c.isdigit() or sc < min_score:
                return None
            digit_scores.append(float(sc))
            out.append(c)
        else:
            out.append(kind)
    text = "".join(out)
    parsed = parse_dmy(text, lay.date_order) if lay.fmt == "DD/MM/YY" else parse_text(text, date_hint)
    if parsed is None:
        return None
    sod, dt, src = parsed
    return ClockReading(text=text, seconds_of_day=sod, datetime=dt, month_source=src,
                        min_digit_score=min(digit_scores) if digit_scores else 0.0, scores=[float(x) for x in scores])


def render(sod: float, date: _dt.date, fmt: str, date_order: str = "DMY") -> str:
    """The text a clock of format ``fmt`` shows at ``sod`` seconds of the day ``date`` (for naming glyphs)."""
    sod = sod % 86400
    ms = int(np.floor(round(sod * 1000, 6))) % 1000
    whole = int(np.floor(round(sod, 6)))
    hh, rem = divmod(whole, 3600)
    mi, ss = divmod(rem, 60)
    if fmt == "DD/MM/YY":
        a, b = (date.day, date.month) if date_order == "DMY" else (date.month, date.day)
        return f"{a:02d}/{b:02d}/{date.year % 100:02d} {hh:02d}:{mi:02d}:{ss:02d}.{ms:03d}"
    return f"{date.day:02d}{MONTHS[date.month - 1]}{date.year % 100:02d} {hh:02d}:{mi:02d}:{ss:02d}.{ms // 100}00"


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
# a clock layout not seen before: find it, learn its glyphs from the video and its file name
# ----------------------------------------------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True as (first, last) indices."""
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return []
    cuts = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[cuts + 1]])
    ends = np.concatenate([idx[cuts], [idx[-1]]])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _signature(pattern: str) -> list[str]:
    """Per space-separated token: 'w' (digit, letter, slash) or 'n' (colon, dot) per character."""
    return ["".join("n" if c in ":." else "w" for c in tok) for tok in pattern.split(" ")]


def find_layout_in_frames(frames: Sequence[np.ndarray], dark: int = 40, bright: int = 128) -> tuple[ClockLayout | None, dict]:
    """Locate a burned-in clock in grey frames spread through one video.

    The text box is dark in every frame (its maximum over the frames stays below ``dark``) and holds white text
    (maximum above ``bright``). Candidates are box-shaped dark regions (filled >= 50 %, wider than 4x their height);
    inside, the columns with white text are cut into characters, gaps much wider than the others are spaces, and
    narrow characters (under half the median width) are ``:`` / ``.``. The pattern of wide / narrow per word is
    matched against :data:`FORMATS`. Returns (layout or None, details)."""
    import cv2
    A = np.stack([np.asarray(f, np.uint8) for f in frames])
    H, W = A.shape[1:]
    mx = A.max(0)
    info: dict = {"frame_size": [W, H], "n_frames": len(A)}
    n, lab, st, _ = cv2.connectedComponentsWithStats((mx < dark).astype(np.uint8), connectivity=4)
    cands = []
    for j in range(1, n):
        x, y, w, h, area = (int(v) for v in st[j])
        if area < 2e-4 * W * H or w < 4 * h or area < 0.5 * w * h or h < 8:
            continue
        box = mx[y:y + h, x:x + w]
        txt = box > bright
        rows = np.nonzero(txt.any(1))[0]
        if len(rows) < 5:
            continue
        runs = _runs(txt[rows[0]:rows[-1] + 1].any(0))
        if len(runs) < 8:
            continue
        cands.append((len(runs), (x, y, w, h), rows, runs))
    info["n_candidates"] = len(cands)
    for _, (x, y, w, h), rows, runs in sorted(cands, key=lambda c: -c[0]):
        widths = np.array([b - a + 1 for a, b in runs], float)
        gaps = np.array([runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)], float)
        med_gap = float(np.median(gaps)) if len(gaps) else 0.0
        space = gaps > max(2.5 * med_gap, med_gap + 6)
        narrow = widths < 0.5 * np.median(widths)
        sig, cur = [], ""
        for i in range(len(runs)):
            cur += "n" if narrow[i] else "w"
            if i < len(gaps) and space[i]:
                sig.append(cur)
                cur = ""
        sig.append(cur)
        for name, f in FORMATS.items():
            if _signature(f["pattern"]) != sig:
                continue
            centers, k = [], 0
            for tok_i, tok in enumerate(f["pattern"].split(" ")):
                if tok_i:            # the space: middle of the gap
                    centers.append((runs[k - 1][1] + runs[k][0]) / 2.0 + x)
                for _c in tok:
                    a, b = runs[k]
                    centers.append((a + b) / 2.0 + x)
                    k += 1
            cw = int(round(np.median(widths[~narrow]))) + 2
            y0 = int(y + rows[0] - 1)
            ch = int(rows[-1] - rows[0] + 3)
            lay = ClockLayout(x0=int(x), y0=y0, pitch=cw, cell_h=ch, n_chars=len(f["pattern"]), ref_width=W,
                              ref_height=H, fmt=name, centers=[float(c) for c in centers], cell_w=cw, shift=1)
            info.update(box=[x, y, w, h], format=name, n_runs=len(runs), cell=[cw, ch])
            return lay, info
        info.setdefault("unmatched_signatures", []).append(" ".join(sig))
    return None, info


def find_layout(path, n_frames: int = 16) -> tuple[ClockLayout | None, dict]:
    """:func:`find_layout_in_frames` on ``n_frames`` grey frames spread evenly through the video."""
    tab = _video.frame_table(path)
    times = np.linspace(min(1.0, tab.duration / 4), max(tab.duration - 1.0, 0.0), n_frames)
    frames = []
    with _video.VideoReader(path, fmt="gray") as r:
        for t in times:
            fr = r.frame_at(float(t))
            if fr is not None:
                frames.append(fr.image)
    if len(frames) < 3:
        return None, {"why": "fewer than 3 frames decoded"}
    lay, info = find_layout_in_frames(frames)
    info["path"] = str(path)
    return lay, info


def _certain(lo: float, hi: float, date: _dt.date, fmt: str, order: str, step: float = 0.004) -> str:
    """Characters that are the same for every clock value in [lo, hi] ('?' where they may differ)."""
    xs = np.append(np.arange(lo, hi, step), hi)
    texts = [render(x, date, fmt, order) for x in xs]
    first = texts[0]
    return "".join(c if all(t[i] == c for t in texts) else "?" for i, c in enumerate(first))


def bootstrap_from_name(path, layout: ClockLayout, n_probes: int = 80, burst: int = 8, passes: int = 3,
                        window: tuple[float, float] = (-1.0, 3.0), min_group: int = 3, log=None) -> GlyphSet:
    """Learn digit templates for ``layout`` from video ``path`` itself.

    The clock text is known roughly from the file name: clock = name start + presentation time + offset, offset
    within ``window`` (the names carry whole seconds). In each pass, every digit cell whose character is the same
    for every offset in the window (date, hours, minutes; later seconds and tenths) is named; the mean of each
    (cell position, character) group becomes a template; all sampled frames are then read with them, a line is
    fitted, and the window shrinks to the fit's offset +- 4 MAD (at least +-0.05 s)."""
    sod0 = names.start_seconds_of_day(path)
    date = names.date_hint(path)
    if sod0 is None or date is None:
        raise ValueError(f"{Path(path).name}: the file name gives no start time, cannot learn the clock from it")
    order = layout.date_order
    tab = _video.frame_table(path)
    t_probe = np.linspace(0.5, max(tab.duration - 1.0, 0.6), n_probes)
    ts, stacks = [], []
    with _video.VideoReader(path, fmt="gray") as r:
        for t in t_probe:
            i0 = tab.index_at(float(t))
            for i in range(i0, min(i0 + burst, tab.n_frames)):
                fr = r.frame_by_index(i)
                if fr is None:
                    continue
                ts.append(fr.t)
                stacks.append(layout.cell_stack(fr.image))
    ts = np.array(ts)
    S = np.stack(stacks)                     # (F, n_shifts, n_chars, h, w)
    mid = S[:, S.shape[1] // 2]              # unshifted cells
    pat = layout.pattern
    slots = [i for i, k in enumerate(pat) if k == "d"]
    lo, hi = window
    history = []
    gs = None
    for p in range(passes):
        groups: dict[tuple[int, str], list[np.ndarray]] = {}
        for f, t in enumerate(ts):
            cert = _certain(sod0 + t + lo, sod0 + t + hi, date, layout.fmt, order)
            for i in slots:
                if cert[i] != "?":
                    groups.setdefault((i, cert[i]), []).append(mid[f, i])
        chars, vecs = [], []
        for (i, c), cl in sorted(groups.items()):
            if len(cl) < min_group:
                continue
            v, _ = _normalise(np.mean(cl, axis=0)[None])
            chars.append(c)
            vecs.append(v[0])
        gs = GlyphSet(chars=chars, vectors=np.array(vecs, np.float32), layout=layout)
        sod = []
        for f in range(len(ts)):
            chs, sc, _ = gs.classify(S[f])
            txt = "".join(c if k == "d" else k for c, k in zip(chs, pat))
            ok = all(chs[i].isdigit() and sc[i] >= 0.6 for i in slots)
            pr = (parse_dmy(txt, order) if layout.fmt == "DD/MM/YY" else parse_text(txt, date)) if ok else None
            sod.append(pr[0] if pr else np.nan)
        sod = np.array(sod)
        good = np.isfinite(sod)
        rec = {"pass": p, "window": [round(lo, 3), round(hi, 3)], "n_templates": len(chars),
               "digits_known": "".join(sorted(set(chars))), "share_read": float(good.mean())}
        if good.sum() >= 5:
            fit = robust_line_fit(ts[good], sod[good] - sod0)
            res = fit["residuals"][fit["inliers"]]
            off = float(np.median(sod[good] - sod0 - ts[good]))
            mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826
            rec.update(offset_s=off, slope=fit["slope"], rms_s=fit["rms_residual_inliers_s"])
            lo, hi = off - max(0.05, 4 * mad), off + max(0.05, 4 * mad)
        history.append(rec)
        if log:
            log(f"  clock glyphs pass {p}: {rec}")
    gs.meta = {"source": str(path), "method": "file name + presentation time", "n_frames": int(len(ts)),
               "passes": history, "share_read": history[-1]["share_read"],
               "created": _dt.datetime.now().isoformat(timespec="seconds")}
    gs.meta["chars_known"] = "".join(sorted(set(gs.chars)))
    return gs


_GLYPH_CHOICE: dict = {}


def glyph_path_for(layout: ClockLayout) -> Path:
    slug = layout.fmt.replace("/", "")
    return paths.cache_dir("timecode") / f"glyphs_{slug}_{layout.ref_width}x{layout.ref_height}.json"


def stored_glyph_sets() -> list[GlyphSet]:
    out = []
    try:
        out.append(load_glyphs())
    except FileNotFoundError:
        pass
    d = paths.cache_dir("timecode")
    for p in sorted(d.glob("glyphs_*.json")) if d.is_dir() else []:
        try:
            out.append(GlyphSet.load(p))
        except Exception:
            continue
    return out


def glyphs_for(path, learn: bool = True, n_test: int = 6, log=None) -> GlyphSet | None:
    """The glyph set that reads this video's clock: the stored set reading most of ``n_test`` frames (at least
    half); else, with ``learn``, a new one (:func:`find_layout` + :func:`bootstrap_from_name`), stored in the cache
    when it reads at least half of its frames. None when no clock can be found or learned."""
    key = str(Path(path).resolve())
    if key in _GLYPH_CHOICE:
        return _GLYPH_CHOICE[key]
    tab = _video.frame_table(path)
    times = np.linspace(min(1.0, tab.duration / 4), max(tab.duration - 1.0, 0.0), n_test)
    with _video.VideoReader(path, fmt="gray") as r:
        frames = [fr.image for fr in (r.frame_at(float(t)) for t in times) if fr is not None]
    hint = names.date_hint(path)
    best, best_n = None, 0
    for gs in stored_glyph_sets():
        lay = gs.layout
        if frames and abs(frames[0].shape[1] / frames[0].shape[0] - lay.ref_width / lay.ref_height) > 0.02:
            continue
        n = sum(read_clock(f, gs, hint) is not None for f in frames)
        if n > best_n:
            best, best_n = gs, n
    if best is not None and best_n >= max(1, len(frames) / 2):
        _GLYPH_CHOICE[key] = best
        return best
    if not learn:
        return None
    lay, info = find_layout(path)
    if lay is None:
        if log:
            log(f"  {Path(path).name}: no burned-in clock found ({info})")
        _GLYPH_CHOICE[key] = None
        return None
    try:
        gs = bootstrap_from_name(path, lay, log=log)
    except ValueError as exc:
        if log:
            log(f"  {exc}")
        _GLYPH_CHOICE[key] = None
        return None
    gs.meta["layout_search"] = info
    if gs.meta["share_read"] >= 0.5:
        gs.save(glyph_path_for(lay))
        if log:
            log(f"  learned clock glyphs {glyph_path_for(lay).name} from {Path(path).name}: "
                f"{gs.meta['share_read']:.0%} of {gs.meta['n_frames']} frames read")
        _GLYPH_CHOICE[key] = gs
        return gs
    _GLYPH_CHOICE[key] = None
    return None


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


def _iter_bursts(path, every_s: float, burst: int, start: float = 0.0, end: float | None = None):
    """``burst`` consecutive frames every ``every_s`` seconds (keyframe seeks; cheap on long 4K files)."""
    tab = _video.frame_table(path)
    end = tab.duration if end is None else min(end, tab.duration)
    with _video.VideoReader(path, fmt="gray") as r:
        for t in np.arange(start, end, every_s):
            i0 = tab.index_at(float(t))
            for i in range(i0, min(i0 + burst, tab.n_frames)):
                fr = r.frame_by_index(i)
                if fr is not None:
                    yield fr


def read_video_clock(path, fps: float | None = None, glyphs: GlyphSet | None = None,
                     date_hint: _dt.date | None = None, start: float = 0.0, end: float | None = None,
                     progress: bool = False, every_s: float | None = None, burst: int = 10) -> dict:
    """Read the clock on every frame (or ``fps`` frames per second, or ``burst`` consecutive frames every
    ``every_s`` seconds) of a video and fit it against presentation time. Without ``glyphs`` the glyph set is
    chosen (or learned) by :func:`glyphs_for`.

    Returns ``rows`` (index, t, text, clock seconds of day or None, residual, flag) and ``summary``."""
    glyphs = glyphs or glyphs_for(path)
    if date_hint is None:
        date_hint = names.date_hint(path)
    rows = []
    if glyphs is None:
        return {"summary": {"path": str(path), "n_frames": 0, "n_read": 0, "share_read": 0.0,
                            "why": "no clock layout found or learned"}, "rows": rows}
    if every_s:
        it = _iter_bursts(path, every_s, burst, start, end)
    else:
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
                     "sampling": f"{burst} frames every {every_s} s" if every_s else None,
                     "clock_format": glyphs.layout.fmt,
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
        glyphs = glyphs or glyphs_for(path)
    except FileNotFoundError:
        return [None] * len(times)
    if glyphs is None:
        return [None] * len(times)
    hint = names.date_hint(path)
    out = []
    with _video.VideoReader(path, fmt="gray") as r:
        for t in times:
            fr = r.frame_at(t)
            out.append(read_clock(fr.image, glyphs, hint) if fr is not None else None)
    return out
