# Patches to vendored gym-pybullet-drones (v1.0.0)

`simulator/` is a vendored copy of gym-pybullet-drones @ tag `v1.0.0`
(MIT license, https://github.com/utiasDSL/gym-pybullet-drones), chosen over the
current main branch because that now requires Python >=3.12 (we're on 3.10).

Two small compatibility patches were applied on top of the v1.0.0 tag to make
it run on Python 3.10 + a numpy version that still has scipy/gym support:

1. `gym_pybullet_drones/envs/BaseAviary.py` line ~909:
   `collections.Mapping` -> `collections.abc.Mapping`
   (`collections.Mapping` was removed in Python 3.10; it moved to
   `collections.abc` back in 3.3.)

2. `requirements.txt` pins `numpy==1.23.5`, not because of an incompatibility
   with this code specifically, but because the vendored code still uses
   `np.int` / `np.float` style aliases in a few places, which were removed in
   NumPy 1.24. 1.23.5 is the last release that still has them. If we patch
   those call sites out later (swap for plain `int`/`float`), we can drop this
   pin and move to a current numpy.

No other changes were made to the vendored simulator. If gym-pybullet-drones
gets upgraded to the current (Python 3.12+) main branch later, both patches
above become unnecessary and this file should be deleted.
