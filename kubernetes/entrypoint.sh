#!/bin/sh

set -e

# copy workspace from nas
# mkdir -p /workspace
# cp -r /mnt/nas5/suhyeon/sources/watermark-anything/. /workspace/

# copy workspace from private git repo
GIT_REPO="StableGuard"
GIT_BRANCH="exp/rebuttal"
TOKEN_FILE="/mnt/nas5/suhyeon/tokens/github_token.txt"
HF_TOKEN_FILE="/mnt/nas5/suhyeon/tokens/hf_token.txt"

export GITHUB_TOKEN=$(cat "$TOKEN_FILE")
export HF_TOKEN=$(cat "$HF_TOKEN_FILE")

GIT_REPO_URL="https://${GITHUB_TOKEN}@github.com/SuhyeonHa/${GIT_REPO}.git"

rm -rf "${GIT_REPO}"
git clone --branch "${GIT_BRANCH}" "${GIT_REPO_URL}"
cd "${GIT_REPO}"
echo "Clone complete."

# packages
pip install -U bitsandbytes
pip install git+https://github.com/huggingface/diffusers.git
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/scraed/LanPaint.git

# python eval_AGE.py
# python -m locmark.main
# python analysis_layers.py
# bash scripts/eval.sh
# python -m locmark.train_decoder
bash scripts/eval_rebuttal.sh