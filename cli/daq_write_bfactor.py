#!/usr/bin/env python3
"""
Write DAQ-style scores into B-factors of a structure model (PDB or mmCIF),
using the same scoring logic as the ChimeraX DAQcolor plugin.

Expected points file:
  points_AA_ATOM_SS_swap.npy  with shape (N, 32):
    0:3   xyz
    3:23  AA probabilities (20)
    23:29 atom-type probabilities (6) ["Other","N","CA","C","O","CB"]
    29:32 SS3 probabilities (unused here)

Example:
  python daq_write_bfactor.py \
      -i model.cif \
      -p points_AA_ATOM_SS_swap.npy \
      -m aa_score \
      -o model.daq.b.cif
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

AA20 = [
    "ALA","VAL","PHE","PRO","MET","ILE","LEU","ASP","GLU","LYS",
    "ARG","SER","THR","TYR","HIS","CYS","ASN","TRP","GLN","GLY"
]
AA_INDEX = {aa:i for i, aa in enumerate(AA20)}
ATOM_TYPES6 = ["Other","N","CA","C","O","CB"]  # 6 classes (index 0..5)

def _knn_idx(db_pts: np.ndarray, q_pts: np.ndarray, k: int = 8,
             radius: Optional[float] = None, chunk: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (dist, idx) with shapes (M,k) for M queries.
    Matches the ChimeraX plugin logic: if radius is set, neighbors beyond radius become inf and idx=0.
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(db_pts)
        if radius is None:
            dist, idx = tree.query(q_pts, k=k)
        else:
            dist, idx = tree.query(q_pts, k=k, distance_upper_bound=float(radius))
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        return dist, idx
    except Exception:
        # NumPy fallback (slower but dependency-free)
        Nq = q_pts.shape[0]
        out_idx = np.empty((Nq, k), dtype=np.int32)
        out_dist = np.empty((Nq, k), dtype=np.float32)
        for s in range(0, Nq, chunk):
            e = min(Nq, s + chunk)
            q = q_pts[s:e]
            diff = q[:, None, :] - db_pts[None, :, :]
            d2 = np.einsum("mpc,mpc->mp", diff, diff)
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            sub = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(sub, axis=1)
            idx = np.take_along_axis(part, order, axis=1)
            dist = np.sqrt(np.take_along_axis(sub, order, axis=1))
            if radius is not None:
                mask = dist > float(radius)
                idx[mask] = 0
                dist[mask] = np.inf
            out_idx[s:e] = idx
            out_dist[s:e] = dist
        return out_dist, out_idx

def _aggregate(pts: np.ndarray, feat: np.ndarray, q: np.ndarray,
               k: int = 1, radius: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    pts:  (N,3)
    feat: (N,C)
    q:    (M,3)

    Returns:
      feat_nn:      (M,C)  feature at nearest valid neighbor (or 0)
      has_neighbor: (M,)   bool
    """
    dist, idx = _knn_idx(pts, q, k=k, radius=radius)
    N, C = feat.shape
    M = q.shape[0]

    valid = (idx >= 0) & (idx < N) & np.isfinite(dist)

    best_pos = np.argmin(dist, axis=1)
    rows = np.arange(M)

    has_neighbor = valid.any(axis=1)
    best_is_valid = has_neighbor & valid[rows, best_pos]

    safe_idx = np.zeros(M, dtype=np.int64)
    safe_idx[best_is_valid] = idx[rows, best_pos][best_is_valid]

    out = feat[safe_idx].copy()
    out[~best_is_valid] = 0.0
    return out, has_neighbor

def _window_average_scal(chain_ids: np.ndarray, resnums: np.ndarray,
                         scal: np.ndarray, half_window: int = 9) -> np.ndarray:
    """
    Chain-aware sliding window average over residue numbers [n-half_window, n+half_window].
    NaN/inf are ignored in averaging; output is NaN if no finite values in window.
    """
    scal = np.asarray(scal, dtype=np.float32)
    R = scal.shape[0]
    out = np.full(R, np.nan, dtype=np.float32)

    for i in range(R):
        c = chain_ids[i]
        n = resnums[i]
        mask = (chain_ids == c) & (resnums >= n - half_window) & (resnums <= n + half_window)
        vals = scal[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            out[i] = float(vals.mean())
    return out

def _load_points(points_npy: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.load(points_npy)
    if arr.ndim != 2 or arr.shape[1] != 32:
        raise ValueError(f"Expected (N,32) numpy file; got {arr.shape} from {points_npy}")
    pts = arr[:, :3].astype(np.float32)
    aa = arr[:, 3:23].astype(np.float32)
    atom = arr[:, 23:29].astype(np.float32)
    return pts, aa, atom

def _metric_to_residue_scores(metric: str,
                             aa_mean: np.ndarray,
                             atom_mean: np.ndarray,
                             resnames3: List[str]) -> np.ndarray:
    metric = metric.strip()
    if metric == "aa_score":
        idx = np.array([AA_INDEX.get(rn, -1) for rn in resnames3], dtype=int)
        scal = np.full((len(resnames3),), np.nan, dtype=np.float32)
        valid = idx >= 0
        if np.any(valid):
            rows = np.nonzero(valid)[0]
            scal[rows] = aa_mean[rows, idx[valid]]
        return scal
    if metric.startswith("aa_conf:"):
        aa3 = metric.split(":", 1)[1].upper()
        if aa3 not in AA_INDEX:
            raise ValueError(f"Unknown AA for aa_conf: {aa3}. Choose one of {AA20}.")
        return aa_mean[:, AA_INDEX[aa3]]
    if metric == "atom_score":
        j = ATOM_TYPES6.index("CA")
        return atom_mean[:, j]
    raise ValueError(f"Unknown metric: {metric}. Use aa_score, atom_score, or aa_conf:ALA (etc).")

def _read_structure(path: Path):
    """
    Read structure with gemmi (PDB or mmCIF). Returns (st, fmt).
    fmt in {"pdb","cif"} for output decision.
    """
    try:
        import gemmi  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "This script requires 'gemmi' for PDB/mmCIF IO. "
            "Install: pip install gemmi"
        ) from e

    ext = path.suffix.lower()
    if ext in [".cif", ".mmcif"]:
        st = gemmi.read_structure(str(path))
        return st, "cif"
    if ext in [".pdb", ".ent"]:
        st = gemmi.read_structure(str(path))
        return st, "pdb"
    # try gemmi autodetect anyway
    st = gemmi.read_structure(str(path))
    # decide output based on content/extension
    return st, ("cif" if ext in [".cif", ".mmcif"] else "pdb")

def _collect_residues(st) -> Tuple[List, np.ndarray, np.ndarray, List[str], List[List]]:
    """
    Collect protein residues and per-residue atom lists.

    Returns:
      residues: list of gemmi.Residue (references)
      chain_ids: (R,) array of chain IDs (strings)
      resnums: (R,) array of residue numbers (int)
      resnames3: list of residue names (3-letter, upper)
      atoms_by_res: list of list[gemmi.Atom] per residue (references)
    """
    residues = []
    chain_ids = []
    resnums = []
    resnames3 = []
    atoms_by_res = []

    for model in st:
        for chain in model:
            for res in chain:
                # accept standard amino acids (3-letter) only
                name = (res.name or "").upper()
                if name not in AA_INDEX:
                    continue
                # require at least one atom
                if len(res) == 0:
                    continue
                residues.append(res)
                chain_ids.append(chain.name)
                # gemmi seqid.num is integer residue number; insertion code ignored for windowing
                try:
                    rn = int(res.seqid.num)
                except Exception:
                    rn = 0
                resnums.append(rn)
                resnames3.append(name)
                atoms_by_res.append([a for a in res])
    return residues, np.array(chain_ids, dtype=object), np.array(resnums, dtype=int), resnames3, atoms_by_res

def _residue_coords(atoms_by_res: List[List], atom_name: str = "CA") -> np.ndarray:
    """
    One 3D coordinate per residue.
    If atom_name is present, use it; else average all atom coordinates in residue.
    """
    coords = []
    for atoms in atoms_by_res:
        chosen = None
        for a in atoms:
            if (a.name or "").strip().upper() == atom_name.upper():
                chosen = a
                break
        if chosen is not None:
            p = chosen.pos  # gemmi.Position
            coords.append((float(p.x), float(p.y), float(p.z)))
        else:
            # average all atoms
            xs = [float(a.pos.x) for a in atoms]
            ys = [float(a.pos.y) for a in atoms]
            zs = [float(a.pos.z) for a in atoms]
            if len(xs) == 0:
                coords.append((0.0, 0.0, 0.0))
            else:
                coords.append((sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)))
    return np.asarray(coords, dtype=np.float32)

def _write_structure(st, out_path: Path, fmt: str):
    import gemmi  # type: ignore
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pdb":
        st.write_pdb(str(out_path))
    else:
        # gemmi: write_structure handles mmCIF
        doc = st.make_mmcif_document()
        doc.write_file(str(out_path))

def main():
    ap = argparse.ArgumentParser(
        description="Write DAQ-style scores into B-factors of PDB/mmCIF using points_AA_ATOM_SS_swap.npy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-i", "--input", required=True, type=Path, help="Input structure file (.pdb/.cif/.mmcif)")
    ap.add_argument("-o", "--output", required=True, type=Path, help="Output structure file (.pdb/.cif/.mmcif)")
    ap.add_argument("-p", "--points", required=True, type=Path,
                    help="Points file (N,32), e.g., points_AA_ATOM_SS_swap.npy")
    ap.add_argument("-m", "--metric", default="aa_score",
                    help="Metric: aa_score | atom_score | aa_conf:ALA (etc)")
    ap.add_argument("--atom-name", default="CA",
                    help="Residue coordinate atom name (typically CA). If missing, averages residue atoms.")
    ap.add_argument("-k", type=int, default=1, help="kNN k")
    ap.add_argument("--radius", type=float, default=3.0, help="kNN radius cutoff (Angstrom). Use <=0 to disable.")
    ap.add_argument("--half-window", type=int, default=9, help="Window averaging half width (n±half_window). Use 0 to disable.")
    ap.add_argument("--no-window", action="store_true", help="Disable window averaging regardless of --half-window.")
    ap.add_argument("--nan-fill", type=float, default=0.0, help="Value written to B-factor when score is NaN/inf.")
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.points.exists():
        raise FileNotFoundError(args.points)

    st, inferred_fmt = _read_structure(args.input)
    fmt = "cif" if args.output.suffix.lower() in [".cif", ".mmcif"] else "pdb"

    residues, chain_ids, resnums, resnames3, atoms_by_res = _collect_residues(st)
    if len(residues) == 0:
        raise RuntimeError("No standard amino-acid residues found in the input structure.")

    q = _residue_coords(atoms_by_res, atom_name=args.atom_name)

    pts, aa, atom = _load_points(args.points)

    radius = None if args.radius is None or args.radius <= 0 else float(args.radius)
    aa_mean, has_nbr = _aggregate(pts, aa, q, k=int(args.k), radius=radius)
    atom_mean, has_nbr2 = _aggregate(pts, atom, q, k=int(args.k), radius=radius)
    has_nbr = has_nbr & has_nbr2  # conservative

    scal = _metric_to_residue_scores(args.metric, aa_mean, atom_mean, resnames3)

    if not args.no_window and int(args.half_window) > 0:
        scal = _window_average_scal(chain_ids, resnums, scal, half_window=int(args.half_window))

    # Replace non-finite (including residues without neighbors) with nan_fill
    scal_for_b = np.asarray(scal, dtype=np.float32).copy()
    bad = ~np.isfinite(scal_for_b)
    if np.any(bad):
        scal_for_b[bad] = float(args.nan_fill)

    # Write to B-factors: per-atom, per-residue repeated
    # Note: gemmi stores B-factor as Atom.b_iso
    for res_atoms, score in zip(atoms_by_res, scal_for_b.tolist()):
        for a in res_atoms:
            a.b_iso = float(score)

    _write_structure(st, args.output, fmt)

    # Simple report
    n_total = len(scal_for_b)
    n_finite = int(np.isfinite(scal).sum())
    n_no_nbr = int((~has_nbr).sum())
    print(f"Wrote B-factors for {n_total} residues to: {args.output}")
    print(f"Metric={args.metric}  k={args.k}  radius={'None' if radius is None else radius}  window={'off' if args.no_window else args.half_window}")
    print(f"Finite scores (pre-fill): {n_finite}/{n_total}  residues without neighbor (within radius): {n_no_nbr}")

if __name__ == "__main__":
    main()
