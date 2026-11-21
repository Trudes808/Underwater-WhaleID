"""Data locations: footage (read only), cache (derived data) and review sheets.

Defaults come from ``configs/paths.toml`` in the repo; each value can be overridden with an environment variable
(``UWID_FOOTAGE_DIR``, ``UWID_CACHE_DIR``, ``UWID_SHEETS_DIR``). ``UWID_PATHS_CONFIG`` points at another toml file.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.toml"

_FALLBACK = {
    "footage_dir": "/home/sat3737/NOAA_whales/underwater_video",
    "cache_dir": "/home/sat3737/NOAA_whales/underwater_cache",
    "sheets_dir": "/home/sat3737/NOAA_whales/review_sheets/underwater",
}
_ENV = {"footage_dir": "UWID_FOOTAGE_DIR", "cache_dir": "UWID_CACHE_DIR", "sheets_dir": "UWID_SHEETS_DIR"}


@lru_cache(maxsize=None)
def _config(config_path: str | None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    values = dict(_FALLBACK)
    if path.is_file():
        with open(path, "rb") as fh:
            values.update(tomllib.load(fh).get("paths", {}))
    return values


def get(name: str) -> Path:
    """Return the configured directory ``name`` (footage_dir, cache_dir, sheets_dir)."""
    env = os.environ.get(_ENV[name])
    if env:
        return Path(env).expanduser()
    return Path(_config(os.environ.get("UWID_PATHS_CONFIG"))[name]).expanduser()


def footage_dir() -> Path:
    return get("footage_dir")


def cache_dir(topic: str | None = None, create: bool = False) -> Path:
    """Cache directory, or its ``topic`` sub folder (created on request)."""
    path = get("cache_dir") if topic is None else get("cache_dir") / topic
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def sheets_dir(create: bool = False) -> Path:
    path = get("sheets_dir")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def is_inside_footage(path: str | os.PathLike) -> bool:
    """True when ``path`` lies inside the read-only footage folder (used to refuse writes there)."""
    try:
        Path(path).resolve().relative_to(footage_dir().resolve())
        return True
    except ValueError:
        return False


def check_writable_target(path: str | os.PathLike) -> Path:
    """Raise if ``path`` would be written inside the footage folder or the repo's source tree."""
    p = Path(path).resolve()
    if is_inside_footage(p):
        raise PermissionError(f"refusing to write inside the read-only footage folder: {p}")
    return p
