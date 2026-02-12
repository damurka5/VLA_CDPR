#!/usr/bin/env python3
"""
Replay saved action_head_in tensors through your action head checkpoint,
WITHOUT importing OpenVLA / LIBERO.

This version matches the discovered checkpoint structure:

Top-level:
  layer_norm1 (in_dim)
  fc1 (in_dim -> hidden_dim)
  mlp_resnet_blocks: N blocks
     block.ffn.0 = LayerNorm(hidden_dim)   <-- weight shape [hidden_dim]
     block.ffn.1 = Linear(hidden_dim->hidden_dim)  <-- weight shape [hidden_dim, hidden_dim]
  layer_norm2 (hidden_dim)
  fc2 (hidden_dim -> out_dim)

Residual assumed: x = x + ffn(x)

Usage:
  python tools/replay_head_from_probes_minimal_v3.py \
    --root probes_cdpr_v2 \
    --action-head-path /root/repo/VLA_CDPR/.../action_head_cdpr.pt
"""

import argparse, glob, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

def strip_prefix(sd: dict):
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    return sd

def infer_arch(sd: dict, in_dim_probe: int):
    sd = strip_prefix(sd)

    if "fc1.weight" not in sd or "fc2.weight" not in sd:
        raise SystemExit("Checkpoint missing fc1/fc2 weights; paste first 50 keys.")

    fc1_w = sd["fc1.weight"]   # (hidden, in)
    fc2_w = sd["fc2.weight"]   # (out, hidden)

    hidden_dim = int(fc1_w.shape[0])
    in_dim_ckpt = int(fc1_w.shape[1])
    out_dim = int(fc2_w.shape[0])

    # count blocks
    block_ids = set()
    pat = re.compile(r"mlp_resnet_blocks\.(\d+)\.ffn\.0\.weight")
    for k in sd.keys():
        m = pat.match(k)
        if m:
            block_ids.add(int(m.group(1)))
    n_blocks = (max(block_ids) + 1) if block_ids else 0

    # confirm ffn.0 is LayerNorm-like
    if n_blocks > 0:
        w_ln = sd["mlp_resnet_blocks.0.ffn.0.weight"]
        if w_ln.ndim != 1 or int(w_ln.shape[0]) != hidden_dim:
            raise SystemExit(f"Expected ffn.0.weight to be 1D [hidden_dim], got {tuple(w_ln.shape)}")

        w_lin = sd["mlp_resnet_blocks.0.ffn.1.weight"]
        if w_lin.ndim != 2:
            raise SystemExit(f"Expected ffn.1.weight to be 2D, got {tuple(w_lin.shape)}")
        if int(w_lin.shape[0]) != hidden_dim or int(w_lin.shape[1]) != hidden_dim:
            raise SystemExit(f"Expected ffn.1.weight to be [hidden,hidden], got {tuple(w_lin.shape)}")

    print(f"[infer] in_dim_ckpt={in_dim_ckpt} (probe={in_dim_probe}) hidden_dim={hidden_dim} out_dim={out_dim} n_blocks={n_blocks}")
    return sd, in_dim_ckpt, hidden_dim, out_dim, n_blocks

class ResBlockLNLinear(nn.Module):
    """
    Matches keys:
      ffn.0.weight/bias : LayerNorm(hidden_dim)
      ffn.1.weight/bias : Linear(hidden_dim -> hidden_dim)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ffn = nn.ModuleList([
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ])
        self.act = nn.GELU()

    def forward(self, x):
        y = self.ffn[0](x)
        y = self.ffn[1](self.act(y))
        return x + y

class ActionHeadResNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_blocks: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()

        self.mlp_resnet_blocks = nn.ModuleList([ResBlockLNLinear(hidden_dim) for _ in range(n_blocks)])

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        for blk in self.mlp_resnet_blocks:
            x = blk(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--action-head-path", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    npz_files = sorted(glob.glob(str(root / "probe_*" / "tensors.npz")))
    if not npz_files:
        raise SystemExit(f"No probes found under {root}")

    d0 = np.load(npz_files[0], allow_pickle=True)
    x0 = d0["action_head_in"].astype(np.float32)
    in_dim_probe = int(x0.shape[1])

    sd_raw = torch.load(args.action_head_path, map_location="cpu")
    if not isinstance(sd_raw, dict):
        raise SystemExit(f"ckpt not dict: {type(sd_raw)}")

    sd, in_dim, hidden_dim, out_dim, n_blocks = infer_arch(sd_raw, in_dim_probe)
    if n_blocks == 0:
        raise SystemExit("No mlp_resnet_blocks found; this script expects them.")

    head = ActionHeadResNet(in_dim, hidden_dim, out_dim, n_blocks)

    # strict load SHOULD work now
    missing, unexpected = head.load_state_dict(sd, strict=True)
    print("✅ load_state_dict(strict=True) ok")
    print("   missing:", missing)
    print("   unexpected:", unexpected)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = head.to(device=device, dtype=torch.float32).eval()

    outs = []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        x = d["action_head_in"].astype(np.float32)

        if x.shape[1] != in_dim:
            if x.shape[1] > in_dim:
                x = x[:, :in_dim]
            else:
                pad = np.zeros((x.shape[0], in_dim - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)

        xt = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            y = head(xt).cpu().numpy().astype(np.float32)
        outs.append(y)

    ref = outs[0]
    print("\nComparing float32 replay outputs vs probe_000")
    for i in range(1, len(outs)):
        max_abs = float(np.max(np.abs(ref - outs[i])))
        mean_abs = float(np.mean(np.abs(ref - outs[i])))
        same = bool(np.array_equal(ref, outs[i]))
        print(f"probe_000 vs probe_{i:03d}: array_equal={same}  max_abs={max_abs:.10f}  mean_abs={mean_abs:.10f}")

if __name__ == "__main__":
    main()
