"""WP15: single-image 3D meshes with TripoSG (VAST-AI, MIT licence) for prepared RGBA inputs (frame + tracker outline).

Runs in the separate env `underwater_cache/envs/triposg` (never the repo .venv), under the GPU lock:

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 900 nice -n 10 \
        underwater_cache/envs/triposg/bin/python experiments/10_mesh3d/run_triposg.py --inputs DIR --out DIR

Resumable: an input whose mesh exists is skipped; stops starting new inputs after --budget-s seconds (default 240 s,
so one locked piece stays under 5 minutes including model loading). Each mesh gets a JSON with seconds, the torch
peak, and nvidia-smi's reading for this process. GPU memory is capped at 9 GB (torch per-process fraction).
TripoSG's own background remover (RMBG-1.4) is NOT used: the alpha channel is the tracker's outline. The flash
decoder needs the `diso` CUDA extension (no nvcc here), so the hierarchical decoder + skimage marching cubes is used.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
TRIPOSG = Path("/home/sat3737/NOAA_whales/underwater_cache/third_party/TripoSG")
WEIGHTS = Path("/home/sat3737/NOAA_whales/underwater_cache/weights/TripoSG")
CAP_GB = 9.0

sys.modules.setdefault("diso", types.SimpleNamespace(DiffDMC=None))   # only the flash decoder needs it
sys.path[:0] = [str(TRIPOSG), str(TRIPOSG / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402


class SmiPeak(threading.Thread):
    """Polls nvidia-smi every 0.5 s for this process's used memory (MiB)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop, self.pid = 0, False, os.getpid()

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    p, m = [x.strip() for x in line.split(",")]
                    if int(p) == self.pid:
                        self.peak = max(self.peak, int(m))
            except Exception:
                pass
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="folder of *.png RGBA inputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=240.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--octree", type=int, default=9, help="hierarchical octree depth (9 = 512^3, the default)")
    ap.add_argument("--order", default="", help="optional CSV with an `id` column: process inputs in that order")
    ap.add_argument("--keep-faces", type=int, default=0,
                    help="storage only: quadric decimation of the saved mesh to this many faces (0 = keep all); "
                         "100k faces differ from the full mesh by 0.010 RMS-radius chamfer, the same as resampling")
    args = ap.parse_args()
    t_start = time.time()
    print(f"run_triposg: process start {time.strftime('%H:%M:%S')}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(Path(args.inputs).glob("*.png")) if not (out / f"{p.stem}.json").exists()]
    if args.order:
        import csv
        rank = {r["id"]: k for k, r in enumerate(csv.DictReader(open(args.order)))}
        todo = sorted([p for p in todo if p.stem in rank], key=lambda p: rank[p.stem])
    print(f"run_triposg: start {time.strftime('%H:%M:%S')}, {len(todo)} to do", flush=True)
    if not todo:
        (out / "ALL.DONE").touch()
        return
    torch.set_num_threads(8)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, CAP_GB / total))
    smi = SmiPeak()
    smi.start()

    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from image_process import prepare_image

    t0 = time.time()
    pipe = TripoSGPipeline.from_pretrained(str(WEIGHTS)).to("cuda", torch.float16)
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f} s, torch alloc {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)

    for k, p in enumerate(todo):
        if k > 0 and time.time() - t_start > args.budget_s:   # at least one per piece
            print("budget reached, stopping this piece", flush=True)
            break
        torch.cuda.reset_peak_memory_stats()
        smi.peak = 0
        t1 = time.time()
        img = prepare_image(str(p), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None)
        with torch.no_grad():
            res = pipe(image=img, generator=torch.Generator(device="cuda").manual_seed(args.seed),
                       num_inference_steps=args.steps, guidance_scale=args.guidance,
                       use_flash_decoder=False, hierarchical_octree_depth=args.octree)
        torch.cuda.synchronize()
        gen_s = time.time() - t1
        v, f = res.samples[0]
        info = dict(input=p.name, model="TripoSG (VAST-AI/TripoSG, MIT), fp16, hierarchical decoder",
                    seed=args.seed, steps=args.steps, guidance=args.guidance, octree=args.octree,
                    seconds=round(gen_s, 2), model_load_s=round(load_s, 1),
                    torch_max_allocated_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                    torch_max_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 3),
                    smi_peak_mib=smi.peak, ok=v is not None)
        if v is not None:
            import trimesh
            mesh = trimesh.Trimesh(v.astype(np.float32), np.ascontiguousarray(f), process=False)
            info.update(vertices=int(len(v)), faces=int(len(f)))
            if args.keep_faces and len(f) > args.keep_faces:
                mesh = mesh.simplify_quadric_decimation(face_count=args.keep_faces)
                info.update(saved_faces=int(len(mesh.faces)))
            mesh.export(out / f"{p.stem}.ply")
        (out / f"{p.stem}.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info), flush=True)
        torch.cuda.empty_cache()
    smi.stop = True
    left = [p for p in sorted(Path(args.inputs).glob("*.png")) if not (out / f"{p.stem}.json").exists()]
    if args.order:
        left = [p for p in left if p.stem in rank]
    if not left:
        (out / "ALL.DONE").touch()
    print(f"run_triposg: end {time.strftime('%H:%M:%S')}, left {len(left)}", flush=True)


if __name__ == "__main__":
    main()
