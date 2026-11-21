# Models, data sets and prior work for phase 1 (WP2)

Web research, 2026-11-20. Tags: **[read]** = on the cited page today; **[search]** = only in a search-result summary of
that page, not opened; **[recall]** = my memory (to mid 2026), not re-checked; **[inference]** = my reading.
Nothing from our footage left this machine. No model was run; every speed or memory figure here is someone else's.

## Recommendations

1. Test in this order (each loads in well under 9 GB in half precision going by parameter count; none measured here):
   1. **OWLv2-base boxes, then SAM 2.1 masks** (both cached, open, Apache-2.0): baseline with no download.
   2. **LLMDet-large** `iSEE-Laboratory/llmdet_large` (open, Apache-2.0, 344 M params, 1.4 GB). Runs with
      `AutoModelForZeroShotObjectDetection`. Best LVIS zero-shot score of the open text-prompted box models in
      transformers (50.2 AP LVIS val). Pass its boxes to SAM 2.1 for masks.
   3. **SAM 3** `facebook/sam3` (**gated**; `Sam3Model` / `Sam3VideoModel` in transformers): text or example box in,
      masks with track ids across video out. It is the current best text-prompted segmenter. SAM 3.1 (2026-03-27) is a
      faster video tracker that drops in for it, also gated.
   4. **YOLOE-26** (open weights, AGPL-3.0, ultralytics >= 8.4.0, 4-55 M params, has `model.track`): text- or
      example-prompted masks at YOLO speed. It is the cheap candidate for whole folders and for fine-tuning once
      WP3 labels exist.
   For fish (question 2), also try the **Community Fish Detector** RF-DETR (open, Apache-2.0, released 2026).
   For **seafloor**:
   - The cached Mask2Former ADE / COCO-panoptic models already have `sand`, `sea`, `rock` and `water` classes.
   - `EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024` (open, 0.34 GB) has reef classes plus `sand` and
     `background` (water).
   - Test whether the carrier's back is called seafloor (section 1b).
   I found nothing newer than SAM 3 / 3.1 that does text-prompted video segmentation. The 2026 papers I found build
   on SAM 3. I found no published result for any of these models on whales underwater, so ours will be the first.
2. Fish data: **UIIS10K** (`LiamLian0727/UIIS10K`, 0.76 GB, Apache-2.0 on the card, masks) and **Brackish**
   (`moondream/brackish_underwater`, 0.54 GB, CC BY 4.0, boxes, murky fixed-camera frames). For whale against fish:
   **Open Images V7** val+test, which has 199 Whale, 200 Dolphin, 203 Shark and 1,328 Fish images, with boxes and
   masks, under 1 GB. These must be cut down by eye to underwater shots. No public underwater set has whales and
   fish together in quantity.
3. **Changes the plan:** a public **CATS humpback deployment**, CC0, is on Zenodo (record 4902325, Goldbogen lab).
   It has 9 tag `.mov` files of 0.57-1.75 GB plus the sensor CSV, `movieTimes.mat` and PRH. One or two files
   (under 3 GB) would give WP1/WP4/WP5 a test deployment that is not ours and has sensor truth.
4. To unlock SAM 3 (worth asking for), the user: (1) logs in at huggingface.co; (2) presses "request access" on
   `facebook/sam3` and `facebook/sam3.1`, which Meta approves by hand; (3) makes a read token; (4) runs
   `hf auth login` here. Unofficial ungated copies also exist (`jetjodh/sam3`, `unsloth/sam3.1`). The SAM License
   allows redistribution, but using a copy is the user's decision.

## 1. Models

### 1a. General text-prompted detectors and segmenters

| model (HF id) | output | licence / access | size | how to run | memory, speed (stated by others) |
|---|---|---|---|---|---|
| SAM 3 `facebook/sam3` | text phrase or example boxes give masks plus ids for every match; video tracking | SAM License; **gated, manual approval** | 860 M params; repo 10.3 GB | transformers `Sam3Model`, `Sam3VideoModel`, `Sam3TrackerModel`, `Sam3TrackerVideoModel` | about 2 GiB peak, 116 ms per 1024 px image on an RTX 5060 Ti (community figure, image only); built for 1008 px |
| SAM 3.1 `facebook/sam3.1` | as SAM 3; "Object Multiplex" tracks objects jointly | SAM License; **gated** | repo 7.0 GB | official repo; transformers support **not confirmed** | 7x faster at 128 objects; 16 to 32 frames/s on an H100 |
| LLMDet `iSEE-Laboratory/llmdet_{tiny,base,large}` | boxes from text | Apache-2.0, open | large 344 M / 1.38 GB; base 233 M / 0.94 GB | transformers (MM-Grounding-DINO class) | not stated |
| MM-Grounding-DINO `openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg` | boxes from text | Apache-2.0, open | 344 M / 2.75 GB | transformers `MMGroundingDinoForObjectDetection` | not stated |
| Grounding DINO `IDEA-Research/grounding-dino-base` | boxes from text | Apache-2.0, open | 233 M / 2.8 GB | transformers | not stated |
| OWLv2 `google/owlv2-base-patch16-ensemble` (cached), `-large-patch14-ensemble` | boxes from text, or from an example image region | Apache-2.0, open | large 438 M / 3.5 GB | transformers `Owlv2ForObjectDetection` | not stated |
| YOLOE-26 n/s/m/l/x (`yoloe-26*-seg.pt`) | instance masks from text, from example boxes, or with no prompt (4,585 built-in names); `track` | AGPL-3.0 or paid Enterprise licence; open (GitHub release v8.4.0) | 3.9-55 M params; text encoder `mobileclip2_b.ts` 254 MB | `pip install ultralytics>=8.4.0` | no frames/s table; LVIS 24.7 (n) to 40.6 (x) AP with text prompts |
| Florence-2 `florence-community/Florence-2-large` | `<OPEN_VOCABULARY_DETECTION>`, `<CAPTION_TO_PHRASE_GROUNDING>`, `<REFERRING_EXPRESSION_SEGMENTATION>` | MIT, open | 777 M / 1.55 GB | transformers `Florence2ForConditionalGeneration` | not stated |
| OmDet-Turbo `omlab/omdet-turbo-swin-tiny-hf` | boxes from text, built for speed | Apache-2.0, open | 115 M / 0.92 GB | transformers | real-time claimed [recall] |
| Rex-Omni `IDEA-Research/Rex-Omni` | boxes and points from a 3B multimodal LLM | IDEA License 1.0 plus Qwen research licence | 4.1 B / 8.1 GB | own code | too large for 9 GB at 16 bit [inference]; skip |
| Qwen3-VL-2B/4B, Molmo2-4B | answers "is there a whale?" for a frame; points | Apache-2.0, open | 2.1 B / 4.4 B / 4.9 B params | transformers | 4B models are about 9 GB in bf16 [inference] |

Notes:
- **SAM 3.** Hugging Face page and transformers docs [read]; official repo [read]; licence [read]; 3.1 release notes
  [read]; memory figure [search].
  - The official repo needs PyTorch >= 2.7 and CUDA >= 12.6. We have torch 2.6 / CUDA 12.4, so we use the
    transformers classes. That they run on torch 2.6 is not checked.
  - In the transformers video session, frames are held on a device we choose (`video_storage_device="cpu"`).
  - Streaming mode (one frame at a time) turns off the "hotstart" filters that drop duplicate and unmatched tracks,
    and the docs warn of more false positives in that mode. For 30-minute files we would run windows of a few
    hundred frames [inference].
  - Lowering `image_size` saves memory, but the docs warn it may cost accuracy.
  - Meta describes video use on "short videos" [search].
  - Licence: anyone may use and redistribute it, with the licence attached. Military, ITAR, nuclear and weapons uses
    are banned.
  - `unsloth/sam3.1` says it is in transformers `Sam3Model` format, which is image only [read]. Whether the 3.1
    video tracker runs in transformers is unknown.
  - URLs: <https://huggingface.co/facebook/sam3>, <https://github.com/facebookresearch/sam3>,
    <https://raw.githubusercontent.com/facebookresearch/sam3/main/RELEASE_SAM3p1.md>,
    <https://huggingface.co/docs/transformers/main/model_doc/sam3_video>,
    <https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE>.
- **LLMDet and MM-Grounding-DINO.** LVIS numbers are copied from the official repos into the transformers docs
  [read]:
  - LLMDet-large: 56.6 AP LVIS minival, 50.2 AP LVIS val.
  - MM-GDINO-tiny: 41.4 AP minival, 31.9 AP val.
  - <https://huggingface.co/docs/transformers/main/model_doc/mm-grounding-dino>
- **YOLOE-26.** Ultralytics docs [read]:
  - Licence: production use means AGPL compliance or an Enterprise licence. For us, AGPL code would be a dependency
    of the repo, not part of it.
  - Prompts: example-box prompts and the no-prompt mode do not need the text encoder.
  - Paper: arXiv 2602.00168. Docs: <https://docs.ultralytics.com/models/yoloe>
- **Sizes, gating and licence fields** came from the Hugging Face API today
  (`/api/models/<id>?expand[]=gated&expand[]=usedStorage`). "Repo size" is all files in the repo, so it can include
  copies in several formats.
- **Few labelled frames.**
  - SAM 3 accepts example boxes as well as text (transformers docs) [read].
  - YOLOE takes `visual_prompts` boxes [read].
  - OWLv2 has image-guided detection, i.e. one example [recall].
  - The DINOv3 patch-feature classifier in PLAN needs no download.

### 1b. Seafloor as its own class (sand, reef, bottom fading into haze)

Seafloor is "stuff" (a region with no countable instances). Instance-style text detectors (OWLv2, Grounding DINO /
LLMDet, YOLOE) tend to return one big box or nothing for prompts like "sea floor" or "sand" [inference]. The models
below are the ones that have seafloor-like classes or can mask a region from text.

| model | seafloor-like output | status here |
|---|---|---|
| Mask2Former Swin-L ADE20K semantic (cached) | trained classes `sand` (46), `sea` (26), `water` (21), `rock` (34), `floor` (3), `animal` (126) | label ids read today from the cached `config.json`. Trained on above-water scenes, so underwater use is untested |
| Mask2Former Swin-L COCO panoptic (cached) | `sand`, `sea`, `water-other`, `rock-merged`, `dirt-merged` | as above |
| SAM 3 | text prompts "sea floor", "sandy seabed", "coral reef". The transformers docs say it outputs instance **and semantic** masks | gated. Whether it does stuff classes well underwater is unknown |
| `EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024` (Apache-2.0, 0.34 GB, open). Also `EPFL-ECEO/coralscapesv2-dinov3-vitb-lora-dpt` (0.44 GB), which needs DINOv3 ViT-B/16; ours is a local file, so it may load without the gated download [inference] | 39 Coralscapes classes incl. `sand`, `rubble`, `unknown hard substrate`, many coral classes, `fish`, `human`, `background` (water), `dark` | open. Trained on diver-height Red Sea reef video frames (CoralscapesV2: 2,433 frames, 55 sites), not on views from a whale's back |
| Florence-2 `<REFERRING_EXPRESSION_SEGMENTATION>` | polygon for a phrase | open; untested on stuff |
| DINOv3 patch-feature classifier (PLAN) | any class we label, including seafloor, open water, carrier's back | no download. Probably the most direct way to separate the carrier's back from the seabed, given a few labelled frames [inference] |

**Confusion to test.** The tagged whale's back fills the bottom of every frame. It is a grey, textured, lit surface
close to the lens, much like sand or rock below. Every seafloor model above has to be scored on whether it calls the
carrier's back "seafloor" (and "rock" / "sand").

Cues that may help, all [inference]:
- the carrier's back is fixed in image position and moves with the camera;
- the seabed slides through the frame, sits behind haze, and appears only at depth;
- the depth sensor can rule out seafloor near the surface, but only if the water depth is known.

The WP3 masks should give the carrier's back its own class so this can be measured.
Sources:
- <https://huggingface.co/EPFL-ECEO/coralscapesv2-dinov3-vitb-lora-dpt> [read]
- <https://huggingface.co/datasets/EPFL-ECEO/coralscapes> [read]

### 1c. Marine-specific models

| model | what it does | access | relevance |
|---|---|---|---|
| Community Fish Detector, <https://github.com/WildHackers/community-fish-detector> | one class, "fish". RF-DETR medium/small 1024 px (Apache), nano 640 px, YOLOv12x (AGPL, deprecated). Trained on 1.9 M images with 935 k fish boxes from 17 open data sets. Release tag 2026.07.06 | GitHub releases, open | **best fish-only candidate** for question 2. The page says nothing on whether whales were ever boxed as fish [read] |
| FathomNet Megalodon `FathomNet/megalodon` (YOLOv8x, CC BY 4.0), `FathomNet/megalodon-2024-yolov11` (one class "object", 0.12 GB) | finds "any animal" in MBARI imagery, mostly deep sea | HF, open | low. FathomNet has almost no living cetaceans: the API counts today were "whale carcass" 29 boxes, *Zalophus californianus* 34, no Cetacea concept [read] |
| MegaFishDetector `FathomNet/megafishdetector` (YOLOv5, MIT, 0.55 GB) | fish boxes | HF, open | fallback to the Community Fish Detector [search] |
| MarineInst (ECCV 2024), stage-1 weights | instance masks with no class names | GitHub `zhengziqiang/MarineInst` | low: no whale / fish labels [search] |
| USIS-SAM (ICML 2024) | salient instance masks, 7 classes, SAM ViT-H base | Google Drive / Baidu, <https://github.com/LiamLian0727/USIS10K> [read] | low |
| MARIS (CVPR 2026), Earth2Ocean / AquaOV255 (CVPR 2026) | underwater open-vocabulary segmentation. The MARIS baselines are FC-CLIP and MAFT+; SAM 3 is not tested | MARIS weights only on Baidu [read], <https://github.com/LiBingyu01/MARIS> | watch only; no mammal classes named |
| TurtleCamClassifier `priyammaz/TurtleCamClassifier` (ViT, 86 M, MIT) | classifies turtle-camera frames as breathing (head above the surface) or apnea | HF, open | a baseline idea for question 4, not a model to reuse |

## 2. Public data for fish and whale-against-fish

| data set | contents | size | licence | download without login |
|---|---|---|---|---|
| **UIIS10K** | 10,048 underwater images; 10 classes, from the annotation file: fish, reptiles, arthropoda, corals, mollusk, plants, ruins, garbage, human, robots. Test split: 4,126 fish masks and 2,484 coral masks over 2,010 images. COCO polygon masks; **no mammal class, no sea-floor class** | 0.76 GB (zip 394 MB) | Apache-2.0 on the HF card; the paper names none | HF `LiamLian0727/UIIS10K` (not gated) |
| **UIIS** | 4,628 images; 7 classes: fish, reefs, aquatic plants, wrecks/ruins, human divers, robots, **sea-floor**. Val: 691 images, 2,040 fish, 1,241 reefs, 149 sea-floor instances. Instance masks | 1.14 GB | Apache-2.0 (HF card) | HF `LiamLian0727/UIIS` |
| **USIS10K** | 10,632 images; 7 classes: wrecks/ruins, fish, reefs, aquatic plants, human divers, robots, **sea-floor**. Val: 1,594 images, 1,354 fish, 846 reefs, 78 sea-floor. Salient-instance masks, so seafloor is masked only where salient | 0.88 GB (zip 460 MB) | Apache-2.0 (HF card) | HF `LiamLian0727/USIS10K` |
| **SUIM** | 1,525 + 110 images; 8 **semantic** classes (every pixel labelled): background **waterbody**, human divers, plants / sea-grass, wrecks, robots, **reefs and invertebrates**, fish and vertebrates, **sea-floor and rocks**. RGB-coded masks, plus binary masks for the test set | not stated | not stated | single Google Drive file linked from <http://irvlab.cs.umn.edu/resources/suim-dataset> [read] |
| Coralscapes / CoralscapesV2 | 2,075 (v1) / 2,433 (v2) Red Sea reef video frames, 1024x2048; 39 semantic classes incl. `sand`, `rubble`, `unknown hard substrate`, `fish`, `background` | v1 5.9 GB, v2 220 GB | Apache-2.0 | HF `EPFL-ECEO/coralscapes` (deprecated), `josauder/coralscapesV2`; too big for us, use the trained models instead |
| **Brackish** | about 12,444 frames from fixed cameras in a brackish strait; crab, fish, jellyfish, shrimp, small_fish, starfish; boxes | 0.54 GB | CC BY 4.0 | HF `moondream/brackish_underwater` |
| **Open Images V7** (subset) | Whale, Dolphin, Fish, Shark, Sea lion, Harbor seal and Sea turtle all have boxes, and all are among the 350 mask classes. Val+test image counts (v5 box files): Whale 52+147, Dolphin 61+139, Fish 356+972, Shark 57+146. Only 24 images hold both Whale and Fish. Mostly Flickr photos, many above water | under 1 GB for these classes [inference] | annotations CC BY 4.0, images listed as CC BY 2.0 [recall] | CSVs at `storage.googleapis.com/openimages/...` (fetched today); images from the public S3 bucket `open-images-dataset` with the official `downloader.py` [recall] |
| UWS (Kabir et al., IROS 2023) | 576 internet images with pixel masks; 21 animal classes incl. **whale, dolphin, seal, shark, turtle, penguin, stingray**; about 30 whale images (bar chart); **no generic fish class** | small [inference] | not stated | Google Drive file, <https://github.com/Imran2205/uwsnet> |
| NDD20 | 4,402 above- and below-water images of two dolphin species, masks | about 2 GB (third-party figure) | CC BY-NC-SA 4.0 | <https://doi.org/10.25405/data.ncl.c.4982342> (my fetch got HTTP 403) |
| WebUOT-238-Test | 238 underwater tracking videos, 192 k frames, boxes plus a language prompt; WebUOT-1M has a "mammal (except humans)" superclass, but the category list is not shown | 3.6 GB | CC BY 4.0 on the card, "academic research" | HF `Voxel51/WebUOT-238-Test` via FiftyOne |
| MUOT-3M | 3,030 videos, 3 M frames, boxes and masks, 677 classes | 216-257 GB | CC BY-NC-ND 4.0 | HF, too big |
| DivingWithPenguins | penguin-borne camera video; 602 frames with boxes for penguin, fish and bubble; 188 predation events | small | not stated | <https://github.com/Kejia928/PenguinProject> |
| FathomNet | public API, no login; see 1c: practically no cetaceans | | | <https://database.fathomnet.org> |
| DeepFish, Fish4Knowledge, OzFish, MarineInst20M, Community Fish Detection Dataset (LILA) | fish only, or too large | | | not checked today [recall] |

Sources:
- UIIS10K classes: <https://arxiv.org/html/2505.15581v4> [read].
- HF sizes and licences came from the HF API today [read].
- Open Images counts and class lists were computed today from
  `storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv`,
  `.../v5/classes-segmentation.txt` and `.../v5/{validation,test}-annotations-bbox.csv` [read].
- UWS paper: <https://analytics.drake.edu/~reza/alimoor_reza_files/fewshot_underwater_iros23.pdf> [read].
- NDD20 licence and count: [search].

Category names and counts for UIIS, USIS10K and UIIS10K were read today from the annotation JSON files on HF. For
the two zips I fetched only the JSON members, with HTTP range requests; no images were downloaded.

**Seafloor data.** Use **SUIM**: it is semantic, so every pixel is waterbody, seafloor/rocks, reef, fish or other.
That makes it the closest match to our "water / seafloor / whale / fish / carrier" split. Add **UIIS** (sea-floor and
reefs as instance masks, 1.14 GB on HF). No public set has a camera looking down from a whale's back. Hawaiian sand
and haze are judged on our own frames.

**Recommendation.** For fish, use UIIS10K. Its Fish class is varied underwater imagery with masks, so it can measure
fish recall and also false "whale" calls on fish. Add Brackish as the hazy, low-contrast case.

For whale against fish:
- Use the Open Images V7 val+test images of Whale, Dolphin, Shark and Fish.
- Keep only the underwater ones. An AI by-eye pass will do, or CLIP followed by a check.
- Add the UWS whale / dolphin / shark images.
- Our own tag video stays the main whale test.

## 3. Prior work on animal-borne camera video

**CATS toolbox: Cade et al. 2021, Animal Biotelemetry 9:34.**
<https://doi.org/10.1186/s40317-021-00256-w>, MATLAB code and wiki at
<https://github.com/wgough/CATS-Methods-Materials>. The paper exists [search]; the article site sent me to a login,
so the detail below is from the wiki [read].

How the toolbox lines up video and sensor time:
- Video start times come from the file names, in **local time**: `CATS-XX-YYYYMMDD-HHMMSS-ZZZ-AAAAA.mov`. Our
  `CC-11-96-20250203-101236-000-00004` fits that pattern. I take ZZZ to be milliseconds [inference].
- The wiki says the clock is "typically visible in the bottom right-hand corner of the screen". Cell 1
  (`useVideoTimestamps`) reads it on every frame and interpolates frames it cannot read.
- That on-screen time "may be 1-2 s delayed in current CATS tags versions", so the file-name time is often the more
  accurate one.
- So the tag's own `.mov` should carry the burned-in clock, not only the conversions. This is not checked on our
  files.
- Newer tags run three clocks: diary (sensors), audio and camera. Audio starts 1-5 s after the data. On the
  deep-rated tags each video file is 1-2 s off. Clock drift "does not seem to be problematic" over 1-2 days.
- The toolbox offers three ways to align the video:
  - **"timestamps"**: trusts the file names.
  - **"surfacings"**: matches surfacings seen on video to depth peaks.
  - **"audio"** (recommended): lines up the camera's own weak audio track with the hydrophone, using impulsive
    sounds "such as the tag breaking the surface". It writes corrected start times into `vidDN`.
  - Our `.mov` files carry 32 kHz mono sound and the CSV has a hydrophone column, so the audio method applies here.
- Other saved variables: `vidDurs`, `camon` / `audon` (when the camera and audio were on) and `timedif`.
- None of the wiki pages I read documents the CSV columns `Camera`, `Camera time`, `CC status` or `CC vid. size`.

How it finds tag-on and the surface:
- **Tag on/off is picked by hand** on depth and acceleration plots. The start is an acceleration spike that changes
  the signal. The end is zero depth or a straight rise in depth (tag released).
- The depth zero is chosen by eye between the calibrated and uncalibrated trace.
- Wiki pages: `Syncing-Video-and-Data-(MainCATSprhTool-Script-Cell-4)`,
  `Notes-on-synching-newer-tags-with-downloadable-audio-files`, `Raw-Video-And-Audio-Processing`,
  `Tag-On-and-Tag-Off-Times-(MainCATSprhTool-Script-Cell-5)`.

**Public CATS deployment mn200312-58: humpback, CC0, Gough & Cade, 2021.** <https://zenodo.org/records/4902325>
[read through the Zenodo API]
- Contents: 21.7 GB in all, including:
  - nine `.mov` files of 0.58-1.75 GB, e.g. `CATS_58-20200312-154936-339-00002.mov` (576 MB);
  - 58 CSV parts of about 134 MB;
  - `.bin`, `.ubx`;
  - `movieTimes.mat` (the toolbox's video times), `Info.mat`, a 10 Hz PRH and NetCDF;
  - a tag `.txt` with `utc_offset=-3`.
- Direct link pattern: `https://zenodo.org/records/4902325/files/<name>?download=1`.
- Its sensor list has Accelerometer, Gyroscope, Magnetometer, GPS, Pitot, Depth (100bar), two-channel Light and
  Speed, but **no "SWS"**. Our tag's `.txt` lists one, so our tag is a newer build.
- `movieTimes.mat` gives a ready check for our sync code [inference].

Other prior work:

| work | animal / camera | what was automated | result | URL |
|---|---|---|---|---|
| Zhang, Yang, Lang, McInnes, Sherley, Burghardt 2023 (CV4Animals extended abstract), "Diving with Penguins" | African penguin, back-mounted camera | YOLOv5 for penguins (carrier and others), fish and **bubbles**; fish-presence classifier for whole frames; behaviour LSTM over 11-frame snippets with optical flow | penguin 98.0 mAP50, fish 73.3, bubble 72.0; fish presence about 91 %; predation 83 % ("not yet reliable") | <https://arxiv.org/abs/2308.07267> [read] |
| Robinson et al. 2026, J. Exp. Biol. 229(11) jeb251688, TurtleCam | green turtle, 1080p 30 fps camera | classifies each frame as "nostrils above the surface" or not, i.e. **surfacing from the picture**; ResNet-50 and ViT at 224 px; balanced classes; 6-frame running mean; runs under 15 frames dropped | ViT: recall 99.1 %, precision 63.3 %, leave-one-deployment-out over 18 turtles. Breath count right in 12 of 18 videos. Errors when the camera is up but the head is down, or the reverse | <https://doi.org/10.1242/jeb.251688> [read]; models and data on HF |
| Kirkham et al. 2026, PLOS ONE 21(9) e0352346 | North Atlantic right whale, Ocean Alliance camera tags (23 whales) | video (4 tags with the mouth in view) used as labels to train a **sensor** feeding classifier | 95 % agreement | <https://doi.org/10.1371/journal.pone.0352346> [search] |
| Chinstrap penguin prey capture, R. Soc. Open Sci. 2024 | penguin accelerometer and depth, with video | sensor deep learning, video as truth | R² about 0.86 | <https://doaj.org/article/9d3f50a31eaa46c7b67f65e6ed007b5e> [search] |
| UCT thesis 2021, "Detection and isolation of prey capture events in animal-borne images" | penguins | prey present yes/no, then a YOLO detector | 60 % mAP | <https://open.uct.ac.za/items/14348407-9ac2-476f-afbc-ff6d4b2bf9c1> [search] |
| Korpela et al. 2020, Commun. Biol. | seabirds | on-board ML that switches video on from accelerometer data | | [recall] |

I found **nothing that detects other whales automatically in whale-borne video**. The cetacean tag-video work I found
uses video by hand, as labels for sensor models.

**"Is the camera in the water": what people use**

| signal | how it is used | weak points | source |
|---|---|---|---|
| depth (pressure) | standard: surface = depth near 0. The CATS toolbox picks the zero offset by eye; the "surfacings" sync uses depth peaks | a tag floating or lying in air also reads about 0 m; ignores waves and splash; tag on deck = air pressure | CATS wiki [read] |
| salt-water / wet-dry switch | conductivity between two contacts. Wildlife Computers: dry reads 255, seawater about 20, "dry if > 150" by default; used to send only at the surface and to log haul-out | fouling makes it read wet forever (or dry forever) | <https://wildlifecomputers.com/blog/determine-biofouling-through-data/> [search] |
| light | two light channels in both CATS sensor lists; no use for wet/dry found in the docs I read | | tag `.txt` [read] |
| sound | surface breaks show as impulsive sounds on both the camera audio and the hydrophone, used for sync; flow noise rises before and after surfacings (DTAG) | | CATS wiki [read]; DTAG figure [search] |
| picture | TurtleCam frame classifier (above) | low precision without time context | JEB 2026 [read] |

## 4. Telling from picture or sound that the lens is at or above the surface

I found **no dedicated method or public data set** for "lens above / at / below the water" on animal-borne video.
What exists:
- **TurtleCam (above).** A frame classifier for "above the surface" is feasible: 99 % recall from ImageNet-pretrained
  backbones on 18 deployments. Precision needs temporal smoothing and event rules (they use a running mean and a
  minimum run length). Its two failure cases are our `crossing` label.
- **Waterline in the picture.** Waterproof-camera patents detect the waterline as a low-contrast band between the
  above- and below-water parts of a split frame (US 9277127, US 8379112). <https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9277127> [search]
- **Water on the lens after exit.** The lens-raindrop literature (synthetic adherent drops with masks, Hao et al.
  ICCVW 2019; Raindrop Clarity; windshield-drop detection, arXiv 2104.05078) could supply a "film or drops on the
  lens" feature or augmentation [search].
- **Sound.**
  - CATS uses the surface-break impulse in the camera audio [read].
  - A hydrophone in air should lose most of its signal level and change spectrum [inference; not found in print].
  - Flow noise rises before and after surfacings [search].
- **Suggested route for WP4** [inference]. Build three cheap per-frame features and fuse them over time with
  hysteresis:
  - picture: colour statistics (blue/green cast gone, sky), DINOv3 or CLIP features with a linear head, a
    bright-band or horizon cue;
  - audio level and spectral shape;
  - depth and light from the CSV.
  Then score them against depth crossings on the Zenodo deployment as well as ours.

## 5. Parked topics (references only)

1. Cheeseman et al. 2021, Mammalian Biology: Happywhale automated matching of humpback fluke undersides
   <https://cascadiaresearch.org/files/s42991-021-00180-9.pdf> [search]
2. Franklin et al. 2020, J. Cetacean Res. Manage.: Hervey Bay humpbacks; ventral flukes, dorsal fins and lateral
   body marks as complementary tags <https://cascadiaresearch.org/publications/Franklin_etal_2020JCetResManagement/> [search]
3. NOAA AFSC: why fluke photos; trailing edge to confirm matches
   <https://webarchive.library.unt.edu/web/20161226054819/http://www.afsc.noaa.gov/ABL/Humpback/WhyPhoto.htm> [search]
4. Kaggle Happywhale 2022 whale and dolphin ID (above-water photos); HF mirror
   `GATE-engine/happy-whale-dolphin-classification`, 53 GB [read API]
5. MegaDescriptor (already cached) and MiewID: multi-species re-ID backbones [recall]
6. Goldbogen et al. 2017, Anat. Rec.: CATS-type video tags used to study moving morphology (flippers, body)
   <https://doi.org/10.1002/ar.23650> [search]
7. Baieri, Cicciarella, Krützen, Rodolà, Zuffi, IJCV 2026 (arXiv 2504.15782): parametric dolphin model fitted to drone
   video by analysis-by-synthesis with a water transmission model; mass and volume <https://arxiv.org/abs/2504.15782> [search]
8. SMAL / 3D-Fauna: general animal shape models fitted to images [recall]

## What I could not verify

- Any number for any model on **underwater whales**, or on our kind of footage. None was found.
- Whether transformers 5.20 runs SAM 3 on torch 2.6 / CUDA 12.4, how much GPU memory `Sam3VideoModel` needs at
  1080p, and whether SAM 3.1's video tracker is in transformers.
- Whether the Community Fish Detector's 17 source sets box marine mammals as "fish", or leave them out.
- Whether the "Fish" class of UIIS10K / USIS10K / SUIM contains any whales or dolphins.
- How SAM 3, ADE / COCO Mask2Former or the Coralscapes models handle seafloor seen from above through blue haze.
- SUIM's licence and download size.
- Whether WebUOT-1M or MUOT-3M has whale sequences that can be downloaded one at a time.
- How many of the Open Images Whale images are underwater.
- The Cade et al. 2021 article text (login wall); the PLOS ONE right whale article (only news and search summaries);
  NDD20 (HTTP 403).
- What our CSV columns `Camera`, `Camera time`, `CC status` and `CC vid. size` mean, and what the "SWS" sensor
  reads. Not documented in the CATS wiki pages I read. WP1 should test them against the video.
- That the Zenodo deployment's video shows other whales: it is a humpback tag, but I opened no video.
