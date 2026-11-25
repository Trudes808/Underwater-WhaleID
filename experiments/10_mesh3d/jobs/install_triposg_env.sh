#!/bin/bash
# Builds the separate TripoSG environment for WP15 (never the repo's .venv).
# Env: underwater_cache/envs/triposg ; code: underwater_cache/third_party/TripoSG (commit fc5c409).
set -u
C=/home/sat3737/NOAA_whales/underwater_cache
ENV=$C/envs/triposg
LOG=$C/mesh3d/install_triposg.log
mkdir -p $C/mesh3d
rm -f $C/mesh3d/install_triposg.DONE $C/mesh3d/install_triposg.FAILED
echo "$(date) start" > $LOG
export OMP_NUM_THREADS=8 MAX_JOBS=8 PIP_NO_CACHE_DIR=1
( rm -rf $ENV && /home/sat3737/.local/bin/virtualenv -q -p /usr/bin/python3.12 $ENV &&
  $ENV/bin/pip install -q --upgrade pip &&
  $ENV/bin/pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 &&
  $ENV/bin/pip install -q "diffusers==0.32.2" "transformers==4.49.0" "peft==0.14.0" "huggingface_hub<1.0" \
      einops opencv-python-headless trimesh omegaconf scikit-image "numpy<2.3" jaxtyping typeguard scipy pillow accelerate pandas &&
  $ENV/bin/python -c "import torch, diffusers, transformers, trimesh, skimage; print('ok', torch.__version__, torch.cuda.is_available(), diffusers.__version__, transformers.__version__)"
) >> $LOG 2>&1 && { echo "$(date) done" >> $LOG; du -sh $ENV >> $LOG; touch $C/mesh3d/install_triposg.DONE; } || { echo "$(date) FAILED" >> $LOG; touch $C/mesh3d/install_triposg.FAILED; }
