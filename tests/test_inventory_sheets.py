import json
import os
import time

import numpy as np
from PIL import Image

from conftest import make_video
from uwid import inventory, sheets
from uwid.cli import main


def _footage(data_dirs):
    root = data_dirs["footage"]
    dep = root / "2025-2-3 Test deployment"
    (dep / "Video").mkdir(parents=True)
    (dep / "Data").mkdir()
    stem = "CC-11-96-20250203-101236-000-00004"
    make_video(dep / "Video" / f"{stem}.mov", n=60, fps=30, gop=30, movflags="frag_keyframe+empty_moov",
               audio=True, audio_channels=1)
    make_video(dep / "Video" / f"{stem}.m4v", n=59, fps=30, gop=120, audio=True)
    (dep / "Video" / f"._{stem}.mov").write_bytes(b"\0" * 4096)
    (dep / "Video" / ".DS_Store").write_bytes(b"\0" * 10)
    (dep / "Data" / "20250203-164014-CC-11-96.csv").write_text("a,b\n1,2\n")
    make_video(root / "2025-12-20 dev.m4v", n=30, fps=30, gop=30)
    old = time.time() - 3600
    for p in root.rglob("*"):
        os.utime(p, (old, old))
    return root, dep, stem


def test_inventory(data_dirs):
    root, dep, stem = _footage(data_dirs)
    make_video(root / "fresh.mp4", n=10)  # just written: must be skipped as "still being copied"
    inv = inventory.build_inventory(root, n_tiles=6, log=lambda *a: None)
    out = data_dirs["cache"] / "inventory"
    assert (out / "inventory.json").is_file() and (out / "inventory.md").is_file()
    assert [s["path"].endswith("fresh.mp4") for s in inv["skipped"]] == [True]
    deps = {os.path.basename(d["folder"]): d for d in inv["deployments"]}
    d = deps["2025-2-3 Test deployment"]
    assert [s["name"] for s in d["sidecars"]] == ["20250203-164014-CC-11-96.csv"]
    rec = d["recordings"][0]
    assert set(rec["forms"]) == {"mov", "m4v"}
    mov = rec["forms"]["mov"]
    assert mov["fragmented"] and mov["decode"]["n_decoded"] == 60 and mov["audio"]["channels"] == 1
    assert mov["name_fields"]["seq"] == 4
    assert rec["comparison"]["frame_difference"] == 1
    assert inv["debris_ignored"] == 2
    sheet = data_dirs["sheets"] / f"inventory_{stem}_mov.jpg"
    assert sheet.is_file()
    # a second run reuses the entries
    msgs = []
    inventory.build_inventory(root, n_tiles=6, log=msgs.append)
    assert any(m.startswith("reuse") for m in msgs)


def test_cli_probe_and_sheet(data_dirs, capsys):
    p = make_video(data_dirs["footage"] / "x.mp4", n=45, fps=30, gop=15)
    assert main(["probe", str(p), "--decode"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["frames"]["n_frames_shown"] == 45 and out["decode"]["n_decoded"] == 45
    assert main(["sheet", str(p), "-n", "5", "--no-clock"]) == 0
    assert (data_dirs["sheets"] / "sheet_x.jpg").is_file()


def test_make_sheet_with_masks(tmp_path):
    imgs = [np.full((90, 160, 3), 40 * i, np.uint8) for i in range(5)]
    masks = [None, np.ones((90, 160), bool), None, np.eye(90, 160, dtype=np.uint8) * 2, None]
    out = tmp_path / "s.jpg"
    sheets.make_sheet(imgs, ["0:00.00"] * 5, ["cap"] * 5, masks, cols=3, tile_width=160, title="t", out=out)
    im = Image.open(out)
    assert im.size[0] == 3 * 164 + 4
