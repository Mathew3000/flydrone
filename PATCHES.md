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

3. `gym_pybullet_drones/envs/BaseAviary.py` (~line 592, in `_getDroneImages`):
   `cameraUpVector=[0, 0, 1]` -> `cameraUpVector=np.dot(rot_mat, np.array([0, 0, 1]))`

   The drone's onboard camera had its up vector pinned to the WORLD vertical,
   so the image never rolled with the aircraft. Measured before the patch:
   rolling the drone 0.8 rad (46 degrees) changed the mean pixel value by 2.9
   grey levels -- i.e. not at all -- where an equivalent pitch changed it by 44
   and a yaw by 73. Roll was simply invisible to the camera.

   That makes any roll-related vision experiment impossible, which is what M4
   needs: the vertical system (T4/T5 -> VS) reads roll off the rotation of the
   visual field, and there was no rotation to read. `rot_mat` is already
   computed two lines above for the camera target, so the fix reuses it.

   This is a behaviour change, not only a compatibility fix like #1 and #2, and
   it is worth being explicit about: it does not affect M0-M3 or M5, all of
   which keep the drone level or rotate it about the vertical axis only, where
   the drone's up vector and the world's coincide.

No other changes were made to the vendored simulator. If gym-pybullet-drones
gets upgraded to the current (Python 3.12+) main branch later, both patches
above become unnecessary and this file should be deleted.
