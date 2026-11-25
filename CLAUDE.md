# Working rules for this repo

Read `PLAN.md` first: it says what phase we are in and what is parked. These rules apply to every person and agent
working here.

## Where things live

| what | where | notes |
|---|---|---|
| footage and tag sensor files | `/home/sat3737/NOAA_whales/underwater_video/` | READ ONLY. Never write, move, rename or delete anything there. |
| derived data (frames, features, masks, predictions, tables built from the footage) | `/home/sat3737/NOAA_whales/underwater_cache/<topic>/` | outside the repo on purpose |
| images for the user to look at (contact sheets, overlays, plots) | `/home/sat3737/NOAA_whales/review_sheets/underwater/` | JPEG or PNG, named `<experiment>_<what>.jpg` |
| code, small text results, docs | this repo | |

Nothing made from the footage goes into the repo: no frames, crops, sheets, videos, sensor tables, GPS positions,
model weights. The repo may be published later. Small result tables (`tables.md`, metrics as JSON) and prompt / key
lists are fine. Do not upload footage or frames to any outside service.

## Other projects on this machine

`Dinov3-WhaleID`, `Whale-Image-Data-Factory`, `Whale-Review-Web`, `moonshot_cache`, `review_sheets/<other folders>`
belong to other running work. Read them and copy code from them if useful; never edit them, never run `pip install`
in their environments, never kill or signal a process you did not start, never touch Unreal.

Python: this repo's `.venv` once the file `.venv/READY` exists. Before that you may RUN (not install into)
`/home/sat3737/NOAA_whales/Dinov3-WhaleID/.venv/bin/python`. No system ffmpeg and no sudo on this machine: ffmpeg
comes from the `imageio-ffmpeg` package in `.venv`.

## Shared machine limits

- One 20 GB GPU, shared with another project that is rendering and training right now. Every GPU command runs as
  `flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 900 <command>`. The `timeout 900` is required: on
  2026-11-23 a job whose decode child had died slept on a pipe and held the lock for 1 h 43 min, blocking the other
  project. Waiting for the lock is normal. Split longer jobs into resumable pieces that cache what they computed. A
  parent that reads from a child process must notice the child's death, and a job writes a log line before its first
  step. Start one copy of a job; check `ps` for a queued copy before starting another. Before a GPU
  job read `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`: the job must fit in the free memory minus
  1.5 GB and never use more than 9 GB (half precision, small batches). On out-of-memory, wait and retry. Never take
  `ue_render.lock`.
- CPU is shared too: at most 8 threads per job (`OMP_NUM_THREADS=8`, `torch.set_num_threads(8)`), `nice -n 10` for
  anything longer than a minute.
- Disk: `/home` had 356 GB free on 2026-11-20 19:15, with more footage being copied in and another project writing
  renders. Check `df -h /home` before anything large. Budget: `.venv` under 8 GB; ask the orchestrator before one
  download over 3 GB; `underwater_cache` under 20 GB in total. Decode frames on the fly; never dump a whole video as
  images.
- Long jobs: start them detached (`nohup ... &`) and have them write a done-marker file; a watcher inside an agent
  dies when the session restarts.

## Git

No commits, no remotes, no pushes: the user commits. Do not create files the user would not want in a commit
(scratch goes to the session scratchpad or `underwater_cache`).

## Evidence

- Split by VIDEO, never by frame: neighbouring frames are near copies. Development video = `2025-12-20 CG.m4v`.
  Every other video is a test video unless `PLAN.md` says otherwise. Nothing is fitted or tuned on a test video; a
  re-fit that includes test videos is leave-one-video-out and is reported as that.
- Labels made by eye by an AI agent are "AI by-eye labels", not ground truth. Say how many frames, and which cases
  were uncertain.
- One run is one run. Say what was fitted on what, give counts next to percentages, and report what did not work.
- A file that is still being copied must not be processed: its size has to be unchanged for 60 seconds.

## Records

Each piece of work ends with an entry at the top of `experiments/LOG.md` (newest first):
`## YYYY-MM-DD HH:MM  <experiment folder>: <one-line result>` followed by what was run, on which data, the numbers,
the verdict, the caveats and the paths of outputs. Detailed tables go in `experiments/<folder>/tables.md`.

## Code layout

`src/uwid/` is the package (`pip install -e .`), `experiments/NN_name/` holds one experiment each (scripts +
`tables.md`), `tests/` uses small synthetic videos made on the fly. Experiments import shared code from `uwid`; code
used by two experiments moves into `uwid`.
