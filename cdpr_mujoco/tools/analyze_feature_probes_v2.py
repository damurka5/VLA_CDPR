#!/usr/bin/env python3
"""
tools/analyze_feature_probes_v2.py

Robust analyzer for probes with variable-length tensors (e.g., LLM seq length varies).
- Computes cosine on a pooled vector representation.
- Also prints max_abs_diff vs ref.

Pooling rules:
- If tensor is (B,T,D): mean over T -> (B,D), then mean over B -> (D,)
- If tensor is (B,N,D) vision tokens: same pooling
- If tensor is (B,D): mean over B -> (D,)
- Otherwise: flatten (careful) and compare by truncating to min length.

Usage:
  python tools/analyze_feature_probes_v2.py --root probes_cdpr_v2
"""

import argparse
import glob
import numpy as np

def cosine(a, b, eps=1e-12):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float((a @ b) / (na * nb + eps))

def to_vector(x: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary tensor to a 1D vector that is comparable across probes.
    """
    x = x.astype(np.float32)

    # Common cases
    if x.ndim == 3:
        # (B,T,D) or (B,N,D): mean over tokens, then over batch
        v = x.mean(axis=1).mean(axis=0)  # -> (D,)
        return v
    if x.ndim == 2:
        # (B,D): mean over batch
        v = x.mean(axis=0)  # -> (D,)
        return v
    if x.ndim == 1:
        return x

    # Fallback: flatten
    return x.reshape(-1)

def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compare vectors even if lengths differ: truncate to min length.
    """
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    n = min(a.size, b.size)
    return float(np.max(np.abs(a[:n] - b[:n])))

def summarize(x: np.ndarray):
    x = x.astype(np.float32)
    return {
        "shape": tuple(x.shape),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="probes_cdpr_v2")
    ap.add_argument("--ref", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.root}/probe_*/tensors.npz"))
    if not files:
        raise SystemExit(f"No probe_*/tensors.npz found under {args.root}")

    arrs = [np.load(f, allow_pickle=True) for f in files]
    print("Loaded", len(arrs), "probes from", args.root)

    tags = ["vision_pre", "vision_post", "llm_block0_in", "llm_block0_out", "action_head_in", "action_head_out", "acts"]

    ref_npz = arrs[args.ref]

    for tag in tags:
        if tag not in ref_npz:
            print(f"\n== {tag}: MISSING ==")
            continue

        print(f"\n== {tag} ==")
        print("ref summary:", summarize(ref_npz[tag]))

        # Build vectors
        v_ref = to_vector(ref_npz[tag])

        cos_vals = []
        mad_vals = []
        std_vals = []

        for i in range(len(arrs)):
            if tag not in arrs[i]:
                continue
            v_i = to_vector(arrs[i][tag])

            # If pooled vectors still differ in length (rare), truncate
            n = min(v_ref.size, v_i.size)
            c = cosine(v_ref[:n], v_i[:n])
            mad = max_abs_diff(v_ref, v_i)

            cos_vals.append(c)
            mad_vals.append(mad)
            std_vals.append(float(arrs[i][tag].astype(np.float32).std()))

        cos_vals = np.array(cos_vals, dtype=np.float64)
        mad_vals = np.array(mad_vals, dtype=np.float64)
        std_vals = np.array(std_vals, dtype=np.float64)

        print(f"cos(ref, all): min={cos_vals.min():.8f} mean={cos_vals.mean():.8f} max={cos_vals.max():.8f}")
        print(f"max_abs_diff(ref, all): min={mad_vals.min():.8f} mean={mad_vals.mean():.8f} max={mad_vals.max():.8f}")
        print(f"std(tensor) across probes: min={std_vals.min():.6f} mean={std_vals.mean():.6f} max={std_vals.max():.6f}")

if __name__ == "__main__":
    main()
