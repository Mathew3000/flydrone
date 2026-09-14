#!/usr/bin/env bash
# Clones gym-pybullet-drones @ v1.0.0 into ../simulator and applies the
# Python 3.10 compatibility patch documented in PATCHES.md.
# simulator/ is gitignored (third-party vendor code, ~28MB) -- run this
# once after cloning the flydrone repo, or whenever simulator/ is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d simulator ]; then
  echo "simulator/ already exists -- delete it first if you want a clean re-clone."
  exit 1
fi

git clone --branch v1.0.0 --depth 1 https://github.com/utiasDSL/gym-pybullet-drones.git simulator
rm -rf simulator/.git

# Patch 1 (see PATCHES.md #1): collections.Mapping -> collections.abc.Mapping
sed -i 's/collections\.Mapping/collections.abc.Mapping/' simulator/gym_pybullet_drones/envs/BaseAviary.py

echo "Done. Now: pip install -r requirements.txt (pins numpy==1.23.5, see PATCHES.md #2)"
