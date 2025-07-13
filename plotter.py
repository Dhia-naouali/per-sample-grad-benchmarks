import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot(df, plots_dir="results"):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)

    sns.lineplot(
        df, 
        x="batch_size",
        y="time",
        hue="method",
    )
    plt.tight_layout()
    plt.save(plots_dir / "time_vs_batch_size.png")


    df["params"] = df["hidden"] * df["depth"] # considering hidden == in + out for simplicity
    sns.scatterplot(
        df,
        x="params",
        y="memory",
        hue="method",
        size="batch_size"
    )
    plt.tight_layout()
    plt.save(plots_dir / "params_vs_memory.png")

