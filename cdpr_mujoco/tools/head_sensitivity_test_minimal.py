#!/usr/bin/env python3
"""
Minimal sensitivity test for your action head checkpoint.
No OpenVLA/LIBERO imports.

It builds the exact head architecture inferred from the checkpoint:
  layer_norm1 -> fc1 -> GELU -> (2x ResBlock(LN+Linear)) -> layer_norm2 -> fc2

Usage:
  python tools/head_sensitivity_test_minimal.py \
    --action-head-path /root/repo/VLA_CDPR/.../action_head_cdpr.pt
"""

import argparse, re
import numpy as np
import torch
import torch.nn as nn

def strip_prefix(sd: dict):
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    return sd

def infer_arch(sd: dict):
    sd = strip_prefix(sd)

    fc1_w = sd["fc1.weight"]
    fc2_w = sd["fc2.weight"]
    hidden_dim = int(fc1_w.shape[0])
    in_dim = int(fc1_w.shape[1])
    out_dim = int(fc2_w.shape[0])

    # count blocks
    block_ids = set()
    pat = re.compile(r"mlp_resnet_blocks\.(\d+)\.ffn\.0\.weight")
    for k in sd.keys():
        m = pat.match(k)
        if m:
            block_ids.add(int(m.group(1)))
    n_blocks = (max(block_ids) + 1) if block_ids else 0
    if n_blocks == 0:
        raise SystemExit("No mlp_resnet_blocks found; this script expects them.")

    # sanity: ffn.0 is LN, ffn.1 is Linear(hidden,hidden)
    w_ln = sd["mlp_resnet_blocks.0.ffn.0.weight"]
    w_lin = sd["mlp_resnet_blocks.0.ffn.1.weight"]
    assert w_ln.ndim == 1 and w_ln.shape[0] == hidden_dim
    assert w_lin.ndim == 2 and w_lin.shape[0] == hidden_dim and w_lin.shape[1] == hidden_dim

    return sd, in_dim, hidden_dim, out_dim, n_blocks

class ResBlockLNLinear(nn.Module):
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
    ap.add_argument("--action-head-path", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = ap.parse_args()

    sd_raw = torch.load(args.action_head_path, map_location="cpu")
    if not isinstance(sd_raw, dict):
        raise SystemExit(f"ckpt not dict: {type(sd_raw)}")

    sd, in_dim, hidden_dim, out_dim, n_blocks = infer_arch(sd_raw)
    print(f"✅ inferred: in_dim={in_dim} hidden_dim={hidden_dim} out_dim={out_dim} n_blocks={n_blocks}")

    head = ActionHeadResNet(in_dim, hidden_dim, out_dim, n_blocks)
    head.load_state_dict(sd, strict=True)
    print("✅ load_state_dict(strict=True) ok")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    head = head.to(device=device, dtype=dtype).eval()

    # Random sensitivity
    x1 = torch.randn(args.batch, in_dim, device=device, dtype=dtype)
    x2 = torch.randn(args.batch, in_dim, device=device, dtype=dtype)

    with torch.no_grad():
        y1 = head(x1).detach().to("cpu", torch.float32).numpy()
        y2 = head(x2).detach().to("cpu", torch.float32).numpy()

    diff = y1 - y2
    print("\nRandom input sensitivity:")
    print("  max_abs(y1-y2) =", float(np.max(np.abs(diff))))
    print("  mean_abs(y1-y2) =", float(np.mean(np.abs(diff))))
    print("  y1[0] =", y1[0])
    print("  y2[0] =", y2[0])

    # Local sensitivity: small epsilon perturbation
    x = torch.randn(args.batch, in_dim, device=device, dtype=dtype)
    eps = (1e-3 if args.dtype == "fp32" else 1e-2)  # bf16 needs larger eps
    delta = torch.zeros_like(x)
    delta[:, 0] = eps  # perturb one dimension

    with torch.no_grad():
        ya = head(x).detach().to("cpu", torch.float32).numpy()
        yb = head(x + delta).detach().to("cpu", torch.float32).numpy()

    d2 = yb - ya
    print("\nSmall-perturbation sensitivity:")
    print("  eps =", eps)
    print("  max_abs(dy) =", float(np.max(np.abs(d2))))
    print("  mean_abs(dy) =", float(np.mean(np.abs(d2))))

if __name__ == "__main__":
    main()
