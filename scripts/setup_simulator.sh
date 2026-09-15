#!/usr/bin/env bash
# Clones gym-pybullet-drones @ v1.0.0 into ../simulator and applies the
# compatibility patches documented in PATCHES.md.
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

# Portable in-place sed: `sed -i <expr>` is GNU-only -- on BSD/macOS sed, -i
# consumes the next argument as a backup suffix, so that form aborts the
# script (set -e) and leaves an unpatched clone behind. Writing to a temp file
# and moving it works identically on both. Each patch is verified to have
# actually changed something, so a silently non-matching pattern (e.g. after
# an upstream reformat) fails loudly instead of surfacing much later as an
# AttributeError at runtime.
patch_file() {
  local file="$1" expr="$2" desc="$3"
  sed "$expr" "$file" > "$file.tmp"
  if cmp -s "$file" "$file.tmp"; then
    rm -f "$file.tmp"
    echo "ERROR: $desc did not match anything in $file -- check PATCHES.md." >&2
    exit 1
  fi
  mv "$file.tmp" "$file"
  echo "patched: $desc"
}

# Patch 1 (see PATCHES.md #1): collections.Mapping -> collections.abc.Mapping
patch_file simulator/gym_pybullet_drones/envs/BaseAviary.py \
  's/collections\.Mapping/collections.abc.Mapping/' \
  "PATCHES.md #1 (collections.abc.Mapping)"

# Patch 2 (see PATCHES.md #2): np.int -> int (removed in NumPy >= 1.24)
patch_file simulator/gym_pybullet_drones/envs/VisionAviary.py \
  's/dtype=np\.int$/dtype=int/' \
  "PATCHES.md #2 (dtype=int)"

# Patch 3 (see PATCHES.md #3): the drone camera's up vector follows the drone
patch_file simulator/gym_pybullet_drones/envs/BaseAviary.py \
  's/cameraUpVector=\[0, 0, 1\],/cameraUpVector=np.dot(rot_mat, np.array([0, 0, 1])),/' \
  "PATCHES.md #3 (camera up vector follows the drone)"

echo "Done. Now: pip install -r requirements.txt"
