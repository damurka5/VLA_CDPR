#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="probes_cdpr")
    args = ap.parse_args()

    root = Path(args.root)
    dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("probe_")])
    arrs = []
    for d in dirs:
        npz = np.load(d / "tensors.npz", allow_pickle=True)
        if "action_head_in" not in npz:
            raise SystemExit(f"missing action_head_in in {d}")
        arrs.append(npz["action_head_in"].astype(np.float32))

    print("loaded", len(arrs), "probes")
    ref = arrs[0]
    for i, a in enumerate(arrs[1:], start=1):
        max_abs = float(np.max(np.abs(ref - a)))
        mean_abs = float(np.mean(np.abs(ref - a)))
        same = bool(np.array_equal(ref, a))
        print(f"probe_000 vs probe_{i:03d}: array_equal={same}  max_abs={max_abs:.8f}  mean_abs={mean_abs:.8f}")

if __name__ == "__main__":
    main()
