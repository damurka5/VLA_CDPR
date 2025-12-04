#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openvla_oft_root", type=str, required=True,
                    help="Path to your openvla-oft repo (the one that contains vla-scripts/finetune.py)")
    ap.add_argument("--cdpr_dataroot", type=str, required=True,
                    help="Path to cdpr_dataset/datasets/cdpr_synth (folder that contains tfrecords/ and action_stats json)")
    ap.add_argument("--pretrained", type=str, default="moojink/openvla-7b-oft-finetuned-libero-spatial")
    ap.add_argument("--outdir", type=str, default="./oft_cdpr_ckpts")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--center_crop", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    oft = Path(args.openvla_oft_root).resolve()
    script = oft / "vla-scripts" / "finetune.py"
    if not script.exists():
        sys.exit(f"Could not find finetune.py at {script}")

    dataset_glob = str(Path(args.cdpr_dataroot).resolve() / "tfrecords" / "*.tfrecord")

    cmd = [
    sys.executable, str(script),
    "--pretrained_checkpoint", args.pretrained,
    "--dataset_glob", dataset_glob,
    "--unnorm_key", "cdpr_synth",
    "--outdir", args.outdir,
    "--num_images_in_input", "2",
    "--use_proprio", "True",     # <— add value
    "--batch_size", str(args.batch_size),
    "--epochs", str(args.epochs),
    "--learning_rate", str(args.lr),
    ]
    if args.center_crop:
        cmd += ["--center_crop", "True"]      # <— add value


    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
