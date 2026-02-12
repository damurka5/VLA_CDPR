#!/usr/bin/env python3
"""
tools/analyze_feature_probes.py

Reads the probe folders produced by feature_probe_cdpr.py and reports:
- cosine similarities between probes for each feature tensor
- per-tensor variance summary
- quick detection of "collapsed" features

Usage:
  python tools/analyze_feature_probes.py --root probes_cdpr
"""

import argparse
import json
from pathlib import Path
import numpy as np

def cosine(a, b, eps=1e-9):
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(a @ b / (na * nb + eps))

def summarize_tensor(x):
    x = x.astype(np.float32)
    return {
        "shape": list(x.shape),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="probes_cdpr")
    ap.add_argument("--max-pairs", type=int, default=50, help="limit printed pairwise comparisons")
    args = ap.parse_args()

    root = Path(args.root)
    probe_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("probe_")])

    if not probe_dirs:
        raise SystemExit(f"No probe_* dirs found under {root}")

    metas = []
    tensors = []
    for pd in probe_dirs:
        meta_path = pd / "meta.json"
        npz_path = pd / "tensors.npz"
        if not meta_path.exists() or not npz_path.exists():
            continue
        metas.append(json.loads(meta_path.read_text()))
        tensors.append(np.load(npz_path, allow_pickle=True))

    print(f"✅ Loaded {len(tensors)} probes from: {root.resolve()}")

    tags = ["vision_pre", "vision_post", "action_head_in", "acts"]

    # Summaries
    print("\n================= TENSOR SUMMARIES =================")
    for tag in tags:
        if tag not in tensors[0]:
            print(f"\n[{tag}] missing in files (not captured)")
            continue
        s0 = summarize_tensor(tensors[0][tag])
        print(f"\n[{tag}] example summary (probe_000): {s0}")

    # Pairwise cosine comparisons
    print("\n================= PAIRWISE COSINES =================")
    N = len(tensors)
    for tag in ["vision_pre", "vision_post", "action_head_in"]:
        if tag not in tensors[0]:
            print(f"\n[{tag}] missing")
            continue
        print(f"\n[{tag}] pairwise cosine similarity (higher ~ more similar / more collapsed):")
        printed = 0
        for i in range(N):
            for j in range(i + 1, N):
                ci = cosine(tensors[i][tag], tensors[j][tag])
                print(f"  cos({i:03d},{j:03d}) = {ci:.4f}   instr_i='{metas[i]['instr']}'  instr_j='{metas[j]['instr']}'")
                printed += 1
                if printed >= args.max_pairs:
                    break
            if printed >= args.max_pairs:
                break

    # Collapse heuristic
    print("\n================= COLLAPSE HEURISTIC =================")
    # If avg cosine is very high and std is tiny, it’s basically constant.
    for tag in ["vision_pre", "vision_post", "action_head_in"]:
        if tag not in tensors[0]:
            continue

        # compute mean cosine across a subset
        cos_vals = []
        for i in range(min(N, 10)):
            for j in range(i + 1, min(N, 10)):
                cos_vals.append(cosine(tensors[i][tag], tensors[j][tag]))
        avg_cos = float(np.mean(cos_vals)) if cos_vals else float("nan")
        avg_std = float(np.mean([tensors[k][tag].std().astype(np.float32) for k in range(min(N, 10))]))

        print(f"\n[{tag}] avg_cos(first10)={avg_cos:.4f}   avg_std(first10)={avg_std:.6f}")
        if avg_cos > 0.98:
            print("  ⚠️ Very high cosine similarity -> features may be collapsed / not responding to changes.")
        if avg_std < 1e-4:
            print("  ⚠️ Very low std -> near-constant tensor values.")

    print("\n✅ Analysis complete.")

if __name__ == "__main__":
    main()
