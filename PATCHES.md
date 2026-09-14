# Patches to vendored gym-pybullet-drones (v1.0.0)

`simulator/` is a vendored copy of gym-pybullet-drones @ tag `v1.0.0`
(MIT license, https://github.com/utiasDSL/gym-pybullet-drones), chosen over
the current main branch because that now requires Python >=3.12 (we're on
3.10). It is gitignored (third-party code, ~28MB) and reproduced by
`scripts/setup_simulator.sh`, which clones the tag and applies both patches
below automatically.

1. `gym_pybullet_drones/envs/BaseAviary.py` (~line 909):
   `collections.Mapping` -> `collections.abc.Mapping`
   (`collections.Mapping` was removed in Python 3.10; it moved to
   `collections.abc` back in 3.3.)

2. `gym_pybullet_drones/envs/VisionAviary.py` (~line 130):
   `dtype=np.int` -> `dtype=int`
   (`np.int` was a deprecated alias for the builtin `int`, removed in
   NumPy 1.24. This was the only `np.int`/`np.float`/`np.bool` occurrence
   found in the code paths we use (`VisionAviary`/`BaseAviary`); a repo-wide
   grep found nothing else.)

With both patches applied, the vendored code runs fine on a current numpy
(>=2) and scipy, which matters because the connectome/ code (medulla
encoder's optical flow via opencv, the LIF network via scipy.sparse) needs a
modern numpy too -- an early version of this project pinned numpy==1.23.5
instead of patching VisionAviary.py, which then hard-conflicted with
opencv-python-headless needing numpy>=2. Patching was the better fix; no
version pin needed for numpy anymore.

No other changes were made to the vendored simulator. If gym-pybullet-drones
gets upgraded to the current (Python 3.12+) main branch later, both patches
above become unnecessary and this file should be deleted.
