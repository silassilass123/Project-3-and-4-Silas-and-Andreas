# -*- coding: utf-8 -*-
"""
MOD300: Assignment 3 — Monte Carlo and Random Walk Simulations
Reorganized utilities with explicit Topic/Task headers.

This file collects all helper functions used in:
- Topic 1 (Monte Carlo volume of union of spheres / DNA)
- Topic 2 (Accessible volume via random walks)

Contributors: Silas Hamran and Andreas Turøy Krag
Date: 2025-10-30
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# Topic 1 - Task 0: Define simulation box (make_box, box_volume)
def make_box(lower, upper) -> Dict[str, np.ndarray]:
    """Make a 3D axis-aligned box as a dict with 'min' and 'max'.

    Args:
        lower: Iterable of 3 numbers for the lower (x, y, z) corner.
        upper: Iterable of 3 numbers for the upper (x, y, z) corner.

    Returns:
        Dict with:
          - "min": np.ndarray of shape (3,)
          - "max": np.ndarray of shape (3,)
    """
    lower = np.array(lower, dtype=float).reshape(3,)
    upper = np.array(upper, dtype=float).reshape(3,)
    assert np.all(upper > lower), "upper has to be bigger than lower"
    return {"min": lower, "max": upper}


def box_volume(box: Dict[str, np.ndarray]) -> float:
    """Compute the volume of the box."""
    return float(np.prod(box["max"] - box["min"]))



# Topic 1 - Task 1: Random point uniformly in box
def random_point_in_box(box, rng=None) -> np.ndarray:
    """Sample a random point (x, y, z) uniformly inside the box."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(box["min"], box["max"])


# Topic 1 - Task 2: Random sphere fully inside box
def random_sphere_in_box(box, r_min=1.0, r_max=5.0, rng=None) -> Dict[str, np.ndarray]:
    """Create a random sphere that fully fits inside the box."""
    if rng is None:
        rng = np.random.default_rng()
    radius = float(rng.uniform(r_min, r_max))
    low = box["min"] + radius
    high = box["max"] - radius
    if np.any(high <= low):
        raise ValueError("Box too small for chosen radius range.")
    center = rng.uniform(low, high)
    return {"center": center, "radius": radius}


# Topic 1 - Task 3: Point-in-sphere test
def point_in_sphere(point, sphere, eps=1e-9) -> bool:
    """Check if a point is inside (or on) a sphere."""
    p = np.asarray(point, dtype=float)
    c = np.asarray(sphere["center"], dtype=float)
    r = float(sphere["radius"])
    return float(np.dot(p - c, p - c)) <= r * r + eps


# Topic 1 - Task 5: Estimate π via Monte Carlo
def estimate_pi_mc(N, rng=None):
    """Estimate π using N random points inside [-1,1]×[-1,1]."""
    if rng is None:
        rng = np.random.default_rng()
    x = rng.uniform(-1, 1, N)
    y = rng.uniform(-1, 1, N)
    inside = (x**2 + y**2) <= 1.0
    return 4 * np.mean(inside)

# Topic 1 – Task 7

def point_in_any_sphere(point, spheres):
    """Return True if the point is inside at least one sphere."""
    for s in spheres:
        if point_in_sphere(point, s):
            return True
    return False


def mc_fraction_union_of_spheres(box, spheres, N, rng):
    """Monte Carlo estimate of inside-fraction for a union of spheres."""
    count = 0
    for _ in range(N):
        p = random_point_in_box(box, rng)
        if point_in_any_sphere(p, spheres):
            count += 1
    return count / N


def analytic_volume_sum(spheres):
    """Naive analytic sum of sphere volumes (ignoring overlap)."""
    vol = 0.0
    for s in spheres:
        r = float(s["radius"])
        vol += (4/3) * np.pi * r**3
    return vol

# Topic 1 - Task 8: Load DNA atoms & assign radii
ATOM_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80}

def atom_radius(symbol: str) -> float:
    """Return a rough van der Waals radius in Å; default 1.5 Å if unknown."""
    return float(ATOM_RADII.get(symbol.upper(), 1.5))

def load_dna_atoms(filename="dna_coords.txt") -> List[Dict[str, float]]:
    """Load a simple DNA coordinate file with lines: SYMBOL x y z (Å).

    Returns:
        List of dicts with keys: symbol, x, y, z, radius
    """
    atoms = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            symbol = parts[0]
            x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
            atoms.append({"symbol": symbol, "x": x, "y": y, "z": z, "radius": atom_radius(symbol)})
    return atoms

def atoms_to_spheres(atoms: List[Dict[str, float]]) -> List[Dict[str, np.ndarray]]:
    """Convert atom dicts to sphere dicts (center, radius)."""
    spheres = []
    for a in atoms:
        center = np.array([a["x"], a["y"], a["z"]], dtype=float)
        spheres.append({"center": center, "radius": float(a["radius"])})
    return spheres


# Topic 1 - Task 9: Build DNA bounding box (with padding)
def box_from_atoms(atoms, padding=2.0) -> Dict[str, np.ndarray]:
    """Build a box that encloses all atoms with a little extra margin."""
    xs = np.array([a["x"] for a in atoms])
    ys = np.array([a["y"] for a in atoms])
    zs = np.array([a["z"] for a in atoms])
    lower = np.array([xs.min()-padding, ys.min()-padding, zs.min()-padding], float)
    upper = np.array([xs.max()+padding, ys.max()+padding, zs.max()+padding], float)
    return {"min": lower, "max": upper}


# Topic 1 - Task 7 & Task 10 (support): Fast collision checks
# - vectorized point-vs-many-spheres and uniform-grid accelerator
def prepare_spheres_arrays(spheres: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of sphere dicts to array form (centers, radii²)."""
    centers = np.array([s["center"] for s in spheres], dtype=float)
    radii2 = np.array([float(s["radius"])**2 for s in spheres], dtype=float)
    return centers, radii2

def any_collision_single_point(point: np.ndarray, centers: np.ndarray, radii2: np.ndarray) -> bool:
    diff = centers - point
    d2 = np.einsum("ij,ij->i", diff, diff)
    return bool(np.any(d2 <= radii2))

def any_collision_points(points: np.ndarray, centers: np.ndarray, radii2: np.ndarray, sphere_chunk: int = 2048) -> np.ndarray:
    """Vectorized collision test for many points against many spheres."""
    M = points.shape[0]
    out = np.zeros(M, dtype=bool)
    N = centers.shape[0]
    for i in range(0, N, sphere_chunk):
        c = centers[i:i+sphere_chunk]
        r2 = radii2[i:i+sphere_chunk]
        diff = points[:, None, :] - c[None, :, :]
        d2 = np.einsum("mki,mki->mk", diff, diff)
        out |= np.any(d2 <= r2[None, :], axis=1)
        if out.all():
            break
    return out

def build_sphere_grid(spheres, cell_size=None):
    """Build a simple uniform grid mapping cell -> sphere indices."""
    centers = np.array([s["center"] for s in spheres], float)
    radii = np.array([float(s["radius"]) for s in spheres], float)
    radii2 = radii**2

    bbox_min = centers.min(axis=0) - radii.max()
    bbox_max = centers.max(axis=0) + radii.max()

    if cell_size is None:
        cell_size = float(np.median(radii) * 2.0)  # median diameter
        if cell_size <= 0:
            cell_size = 1.0

    grid = {}
    mins = np.floor((centers - radii[:,None] - bbox_min) / cell_size).astype(int)
    maxs = np.floor((centers + radii[:,None] - bbox_min) / cell_size).astype(int)

    for i in range(centers.shape[0]):
        x0,y0,z0 = mins[i]; x1,y1,z1 = maxs[i]
        for xi in range(x0, x1+1):
            for yi in range(y0, y1+1):
                for zi in range(z0, z1+1):
                    key = (xi, yi, zi)
                    grid.setdefault(key, []).append(i)
    for k,v in grid.items():
        grid[k] = np.asarray(v, dtype=int)

    return {
        "origin": bbox_min,
        "cell_size": float(cell_size),
        "grid": grid,
        "centers": centers,
        "radii2": radii2,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }

def any_collision_points_grid(points, grid_index, sphere_chunk: int = 4096):
    """Collision test using the grid to limit candidate spheres."""
    pts = np.asarray(points, float)
    origin = grid_index["origin"]; cell_size = grid_index["cell_size"]
    centers = grid_index["centers"]; radii2 = grid_index["radii2"]; grid = grid_index["grid"]

    keys = np.floor((pts - origin) / cell_size).astype(int)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    M = pts.shape[0]
    out = np.zeros(M, dtype=bool)

    for ui, cell in enumerate(uniq):
        mask = (inv == ui)
        idxs = grid.get(tuple(cell), None)
        if idxs is None or idxs.size == 0:
            continue
        sub_pts = pts[mask]

        for s0 in range(0, idxs.size, sphere_chunk):
            sidx = idxs[s0:s0+sphere_chunk]
            c = centers[sidx]
            r2 = radii2[sidx]
            diff = sub_pts[:, None, :] - c[None, :, :]
            d2 = np.einsum("mki,mki->mk", diff, diff)
            out_sub = (d2 <= r2[None, :]).any(axis=1)
            out[mask] |= out_sub
            if out[mask].all():
                break
    return out


# Topic 2 - Task 1: Generate 3D random walkers
def generate_random_walkers(box, n_walkers, n_steps, step_std=1.0, rng=None) -> np.ndarray:
    """Generate multiple 3D random walks starting from random box positions.

    Steps are i.i.d. Normal(0, step_std) per axis.
    Returns an array of shape (n_walkers, n_steps+1, 3).
    """
    if rng is None:
        rng = np.random.default_rng()
    starts = rng.uniform(low=box["min"], high=box["max"], size=(n_walkers, 3))
    steps = rng.normal(0, step_std, size=(n_walkers, n_steps, 3))
    displacements = np.cumsum(steps, axis=1)
    positions = np.empty((n_walkers, n_steps + 1, 3), dtype=float)
    positions[:, 0, :] = starts
    positions[:, 1:, :] = starts[:, None, :] + displacements
    return positions


# Topic 2 - Task 5 (support): Exit test and single-walk routine
def is_outside_box(point, box, tol=0.0) -> bool:
    """Return True if the point is at or beyond the box boundary."""
    p = np.asarray(point, float)
    return bool(np.any(p < box["min"] + tol) or np.any(p > box["max"] - tol))

def random_walk_escape(start, box, dna_spheres,
                       n_steps=400, step_std=1.0, rng=None) -> Tuple[bool, bool]:
    """Random walk that stops if it hits DNA or exits the box
    Returns:
        (escaped, hit_dna)
    """
    if rng is None:
        rng = np.random.default_rng()
    centers, radii2 = prepare_spheres_arrays(dna_spheres)
    pos = np.asarray(start, float)

    if any_collision_single_point(pos, centers, radii2):
        return False, True

    for i in range(int(n_steps)):
        pos += rng.normal(0.0, step_std, size=3)
        if any_collision_single_point(pos, centers, radii2):
            return False, True
        if is_outside_box(pos, box, tol=0.0):
            return True, False
    return False, False


# Topic 2 - Task 5: Estimate accessible volume by random-walk Monte Carlo
def estimate_accessible_volume(box, dna_spheres,
                               n_starts=200,
                               n_steps=400,
                               step_std=1.0,
                               rng=None):
    """Estimate accessible volume via random-walk Monte Carlo (fast).

    Returns:
        (frac, stats) where:
          - frac in [0,1] is the escaped fraction
          - stats is a dict: {"escaped", "hit_dna", "stuck", "total"}
    """
    if rng is None:
        rng = np.random.default_rng()

    use_grid = len(dna_spheres) >= 600  # heuristic threshold
    if use_grid:
        grid_index = build_sphere_grid(dna_spheres, cell_size=max(2*float(step_std), 1.0))
        collision_fn = lambda P: any_collision_points_grid(P, grid_index)
    else:
        centers, radii2 = prepare_spheres_arrays(dna_spheres)
        collision_fn = lambda P: any_collision_points(P, centers, radii2)

    batch = min(2048, max(128, n_starts // 4 or 128))
    remaining = n_starts

    escaped = hit_dna = stuck = 0
    while remaining > 0:
        m = min(batch, remaining)
        starts = rng.uniform(box["min"], box["max"], size=(m, 3))

        pos = starts.copy()
        alive = np.ones(m, dtype=bool)
        esc = np.zeros(m, dtype=bool)
        hit = np.zeros(m, dtype=bool)

        # start collision
        start_hit = collision_fn(pos)
        hit |= start_hit
        alive &= ~start_hit

        for i in range(int(n_steps)):
            if not np.any(alive):
                break
            idx = np.where(alive)[0]
            pos[idx] += rng.normal(0.0, step_std, size=(idx.size, 3))
            # boundary first (cheap)
            out = (pos[idx] < (box["min"])).any(axis=1) | (pos[idx] > (box["max"])).any(axis=1)
            esc[idx] |= out
            alive[idx] &= ~out
            # only collide those still alive
            idx2 = np.where(alive)[0]
            if idx2.size:
                coll = collision_fn(pos[idx2])
                hit[idx2] |= coll
                alive[idx2] &= ~coll

        escaped += int(esc.sum())
        hit_dna += int(hit.sum())
        stuck   += int((~esc & ~hit).sum())
        remaining -= m

    frac = escaped / n_starts if n_starts > 0 else 0.0
    return frac, {"escaped": escaped, "hit_dna": hit_dna, "stuck": stuck, "total": n_starts}
