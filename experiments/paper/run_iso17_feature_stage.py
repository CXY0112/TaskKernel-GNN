"""Resume three seeds of one feature-message variant, without full-test tuning."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from train_iso17_feature_kernels import PROJECT, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A","B","C"), required=True)
    args = parser.parse_args()
    script = Path(__file__).with_name("train_iso17_feature_kernels.py")
    for seed in (0,42,37):
        folder = ROOT / args.variant / f"budget_128_seed_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "metrics.json").exists():
            previous = json.loads((folder / "metrics.json").read_text())
            if previous["steps"] != 6000:
                raise ValueError("refusing to treat a short run as a completed formal run")
            continue
        print(f"START {args.variant} seed={seed}",flush=True)
        with (folder / "train.log").open("w",encoding="utf-8") as output:
            subprocess.run([sys.executable,str(script),"--variant",args.variant,"--seed",str(seed)],
                           cwd=PROJECT,stdout=output,stderr=subprocess.STDOUT,check=True)
        print(f"DONE {args.variant} seed={seed}",flush=True)


if __name__ == "__main__":
    main()
