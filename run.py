import torch
from pathlib import Path
import argparse

from benchmark import sweep
from plotter import plot
from utils import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batches")
    parser.add_argument("--depths")
    parser.add_argument("--in_dim")
    parser.add_argument("--out_dim")
    parser.add_argument("--plots_dir", default="results")
    args = parser.parse_args()

    configs = load_config(
        args.config,
        batches=args.batches,
        depths=args.depths,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        plots_dir=args.plots_dir,
    )

    df = sweep(**configs)
    plots_dir = Path(configs["plots_dir"])
    plots_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(plots_dir / "benchmarks.csv", index=False)
    plot(df, plots_dir)


if __name__ == "__main__":
    main()