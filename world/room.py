"""
A patterned 3D room for the drone to actually see something in.

Why this exists: VisionAviary's default world is a bare ground plane under an
empty sky. A forward-looking camera in that world returns an essentially
uniform image, so the Farneback optical flow in connectome.medulla_encoder is
~0 everywhere, the T4/T5 populations get no drive, and the connectome's decoded
output collapses to exactly 0.0 as soon as the drone stops climbing (confirmed
in both M2 scripts: thr_out/yaw_out go to 0.000 at steady state and the yaw
setpoint freezes). The closed loop is then only nominally closed -- the PID
holds the drone up and the fly contributes nothing.

The walls are vertical bars on purpose. This is the fly optomotor paradigm: a
tethered fly inside a striped drum turns to follow the drum's motion, and that
response is driven by exactly the T4/T5 -> lobula plate tangential cell
(HS/H1/H2) pathway this project wires up. Vertical bars produce strong
horizontal luminance gradients, which is what a yaw rotation sweeps across the
retina.

Bar widths and shades are randomized (fixed seed) rather than a regular
grating: a periodic pattern is ambiguous to a correlation-based flow estimator
(a shift of exactly one period is indistinguishable from no shift), which shows
up as flow that flips sign between frames. Randomizing keeps the same dominant
spatial frequency without that ambiguity.

Two rendering constraints shaped this module, both measured rather than
assumed:

  - The bars are built as separate box bodies, not painted on a wall with a
    texture. PyBullet's GEOM_BOX UV mapping stretches a texture across a thin
    slab in a way that does not track the slab's proportions, so a vertical-bar
    image came out as horizontal smears on the large faces. Geometry sidesteps
    UV mapping entirely and makes the bar count an exact, controllable
    spatial frequency.
  - Shades stay well clear of black. Rendered brightness is roughly albedo x
    lighting, and an interior vertical face here comes out at ~0.69 of full
    (measured: an untextured white wall renders at mean 176/255 from the
    drone's camera). A 0.08/0.92 bar pair would land at 14/162 -- the dark bars
    crush to near-black, which is what made the first version of this room look
    unlit despite being correctly textured.

Everything is built through the plain pybullet API against an already running
client, so this does NOT require touching the vendored simulator (see
PATCHES.md -- simulator/ stays unpatched beyond the two documented fixes).
Call build_room() AFTER env.reset(): reset() calls p.resetSimulation(), which
would wipe anything added before it.
"""
from __future__ import annotations

import os

import numpy as np
import pybullet as p
from PIL import Image

TEXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "files", "textures")

BAR_DARK = 0.34
BAR_LIGHT = 0.95


def make_mosaic_texture(path: str, size: int = 512, cells: int = 16, seed: int = 1,
                        low: int = 95, high: int = 255) -> str:
    """Write a random-grayscale-block texture for the floor and return its path.

    Bars would be the wrong pattern underneath the drone: they carry gradient
    across one axis only, so translation along the bars produces no flow at
    all. A 2D mosaic has structure in both directions, which is what the
    ventral flow field needs (the fly uses ventral flow for altitude and ground
    speed -- see docs/projektplan.md section 4).

    Generated rather than vendored as a binary; files/ is gitignored.
    """
    rng = np.random.default_rng(seed)
    block = rng.integers(low, high + 1, size=(cells, cells)).astype(np.uint8)
    img = np.kron(block, np.ones((size // cells, size // cells), dtype=np.uint8))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.stack([img] * 3, axis=-1)).save(path)
    return path


def _box(client: int, half_extents, position, shade: float) -> int:
    rgba = [shade, shade, shade, 1.0]
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba,
                                 physicsClientId=client)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                       physicsClientId=client)
    return p.createMultiBody(baseMass=0,  # static
                             baseCollisionShapeIndex=collision,
                             baseVisualShapeIndex=visual,
                             basePosition=position,
                             physicsClientId=client)


def _bar_edges(size: float, n_bars: int, rng) -> np.ndarray:
    """Split a wall of length `size` into n_bars segments of unequal width."""
    widths = rng.uniform(0.6, 1.4, size=n_bars)
    widths *= size / widths.sum()
    return np.concatenate([[0.0], np.cumsum(widths)])


def build_room(client: int, size: float = 8.0, height: float = 6.0, n_bars: int = 14,
               pillars: bool = True, plane_id: int | None = None, seed: int = 0) -> dict:
    """Build a square bar-walled room of `size` x `size` metres on the origin.

    `height` defaults to 6 m, taller than it looks like it needs to be: the
    camera sits at ~1 m with a 60 deg vertical FOV, so at 4 m from a wall it
    looks up to z ~= 3.3 m. A 3 m wall leaves a band of empty sky across the top
    of every frame -- a large, uniform, texture-free region that contributes
    nothing but dilutes the mean flow the encoder computes over the whole image.

    Pass `plane_id` (VisionAviary exposes it as env.PLANE_ID) to also replace
    the ground plane's stock blue checkerboard with a mosaic texture. Worth
    doing: the floor is the best-lit surface in the scene and fills the lower
    third of the camera frame, and the mosaic roughly doubles the horizontal
    gradient there over the stock checkerboard (measured ~37 vs ~15 grey
    levels/px).

    Returns {"walls": [...], "pillars": [...], "floor_texture": id|None} so a
    caller can move things later (e.g. rotating the walls to reproduce the
    moving-drum version of the optomotor experiment).
    """
    rng = np.random.default_rng(seed)
    half, thickness, hh = size / 2.0, 0.05, height / 2.0

    floor_texture_id = None
    if plane_id is not None:
        floor_path = make_mosaic_texture(os.path.join(TEXTURE_DIR, f"mosaic_{seed}.png"),
                                         seed=seed + 1)
        floor_texture_id = p.loadTexture(floor_path, physicsClientId=client)
        p.changeVisualShape(plane_id, -1, textureUniqueId=floor_texture_id,
                            physicsClientId=client)

    walls = []
    # (axis the wall runs along, fixed coordinate of the wall plane)
    for along_x, offset in [(True, half), (True, -half), (False, half), (False, -half)]:
        edges = _bar_edges(size, n_bars, rng)
        for i in range(n_bars):
            lo, hi = edges[i] - half, edges[i + 1] - half
            centre, half_len = (lo + hi) / 2.0, (hi - lo) / 2.0
            shade = (BAR_DARK if i % 2 == 0 else BAR_LIGHT) + rng.uniform(-0.05, 0.05)
            if along_x:
                walls.append(_box(client, [half_len, thickness, hh], [centre, offset, hh], shade))
            else:
                walls.append(_box(client, [thickness, half_len, hh], [offset, centre, hh], shade))

    pillar_ids = []
    if pillars:
        # Off-centre and asymmetric on purpose: identical pillars in a symmetric
        # layout would give the left and right hemifields the same input, i.e.
        # zero yaw signal by construction.
        for (x, y, w, h, shade) in [(2.2, 1.4, 0.18, 2.0, BAR_LIGHT),
                                    (-1.9, 2.6, 0.25, 1.4, BAR_DARK),
                                    (0.8, -2.8, 0.15, 2.4, BAR_LIGHT)]:
            pillar_ids.append(_box(client, [w, w, h / 2.0], [x, y, h / 2.0], shade))

    return {"walls": walls, "pillars": pillar_ids, "floor_texture": floor_texture_id}


def build_drum(client: int, radius: float = 3.5, height: float = 6.0, n_bars: int = 24,
               plane_id: int | None = None, seed: int = 0) -> dict:
    """Build a cylindrical striped drum centred on the origin -- the rotatable
    version of build_room(), for the optomotor experiment.

    A square room cannot be rotated honestly: its corners would swing toward
    and away from the drone, so wall distance (and therefore flow magnitude)
    would change with drum angle and confound the direction signal. Bars on a
    circle keep every bar equidistant, so rotation changes only the direction
    of motion, which is the one variable the experiment is about.

    Bars are evenly spaced here rather than randomized as in build_room(). The
    periodic-pattern ambiguity that randomization avoids is a problem when the
    flow estimator must recover an unknown shift; in this experiment the drum
    speed is set by us and kept well below one bar per frame, and even spacing
    makes the stimulus a clean grating like the real optomotor drum.

    Returns a dict that rotate_drum() consumes.
    """
    floor_texture_id = None
    if plane_id is not None:
        floor_path = make_mosaic_texture(os.path.join(TEXTURE_DIR, f"mosaic_{seed}.png"),
                                         seed=seed + 1)
        floor_texture_id = p.loadTexture(floor_path, physicsClientId=client)
        p.changeVisualShape(plane_id, -1, textureUniqueId=floor_texture_id,
                            physicsClientId=client)

    angles = np.linspace(0.0, 2.0 * np.pi, n_bars, endpoint=False)
    # Bars cover half the circumference; the gaps between them are the dark
    # half of the grating, so contrast comes from bar-vs-gap (the drum has no
    # backing wall -- the sky shows through, which is bright and uniform).
    half_width = (2.0 * np.pi * radius / n_bars) * 0.25

    bars = []
    for i, theta in enumerate(angles):
        shade = BAR_DARK if i % 2 == 0 else BAR_LIGHT
        # Local +x radial, +y tangential after a rotation of theta about z.
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, half_width, height / 2.0],
                                     rgbaColor=[shade, shade, shade, 1.0], physicsClientId=client)
        body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                                 basePosition=[radius * np.cos(theta), radius * np.sin(theta),
                                               height / 2.0],
                                 baseOrientation=p.getQuaternionFromEuler([0, 0, theta]),
                                 physicsClientId=client)
        bars.append(body)

    return {"bars": bars, "angles": angles, "radius": radius, "height": height,
            "floor_texture": floor_texture_id}


def rotate_drum(client: int, drum: dict, angle: float) -> None:
    """Rotate the whole drum to absolute `angle` (radians about z).

    Visual-only bodies with baseMass=0, so resetBasePositionAndOrientation is
    the right call: there is no physics state to keep consistent, and the drone
    is not supposed to be pushed by the drum -- only to see it.
    """
    radius, height = drum["radius"], drum["height"]
    for body, theta0 in zip(drum["bars"], drum["angles"]):
        theta = theta0 + angle
        p.resetBasePositionAndOrientation(
            body,
            [radius * np.cos(theta), radius * np.sin(theta), height / 2.0],
            p.getQuaternionFromEuler([0, 0, theta]),
            physicsClientId=client,
        )


def spawn_object(client: int, half_size: float = 0.5, position=(8.0, 0.0, 1.0),
                 shade: float = BAR_DARK, textured: bool = True, seed: int = 7) -> int:
    """A movable box, for looming stimuli. Returns its body id.

    No collision shape: it is meant to be driven through space with
    move_object() on a collision course with the drone, and a physical
    collision would end the trial with a crash rather than a measurement. The
    drone is supposed to see it, not be hit by it.

    Textured by default so its surface carries motion as it expands. An
    untextured box only produces flow at its silhouette edges, which for a
    correlation detector sampling 2 px apart is a thin signal.
    """
    visual_kwargs = dict(halfExtents=[half_size] * 3, rgbaColor=[shade, shade, shade, 1.0])
    visual = p.createVisualShape(p.GEOM_BOX, physicsClientId=client, **visual_kwargs)
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                             basePosition=list(position), physicsClientId=client)
    if textured:
        path = make_mosaic_texture(os.path.join(TEXTURE_DIR, f"object_{seed}.png"),
                                   cells=8, seed=seed, low=60, high=255)
        p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(path, physicsClientId=client),
                            physicsClientId=client)
    return body


def move_object(client: int, body: int, position) -> None:
    p.resetBasePositionAndOrientation(body, list(position), [0, 0, 0, 1],
                                      physicsClientId=client)
