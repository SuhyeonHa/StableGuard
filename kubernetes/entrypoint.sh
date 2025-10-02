#!/bin/sh

set -e

# copy workspace from nas
# mkdir -p /workspace
# cp -r /mnt/nas5/suhyeon/sources/watermark-anything/. /workspace/

# copy workspace from private git repo
GIT_REPO="StableGuard"
GIT_BRANCH="main"
TOKEN_FILE="/mnt/nas5/suhyeon/tokens/github_token.txt"
WORKSPACE = "/workspace"

export GITHUB_TOKEN=$(cat "$TOKEN_FILE")

GIT_REPO_URL="https://${GITHUB_TOKEN}@github.com/SuhyeonHa/${GIT_REPO}.git"

rm -rf "${CLONE_DIR}"
mkdir -p "${CLONE_DIR}"
git clone --branch "${GIT_BRANCH}" "${GIT_REPO_URL}" "${CLONE_DIR}"
cd "${CLONE_DIR}"
echo "Clone complete."

python eval_AGE.py