#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training/validation loss from a CSV log.")
    parser.add_argument("csv_file", help="Path to CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file, na_values=["None"])

    csv_path = Path(args.csv_file)
    stem = csv_path.stem

    for col, ylabel, suffix in [
    ("train_loss", "Training Loss", "train_loss"),
    ("val_loss", "Validation Loss", "val_loss"),
    #("lr", "Learning Rate", "lr"),
    ("lrm", "Learning Rate Multiplier", "lrm"),
    ]:
        plot_df = df.dropna(subset=[col]) if col == "val_loss" else df

        plt.figure()
        plt.plot(plot_df["step"], plot_df[col])
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Step")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(csv_path.parent / f"{stem}_{suffix}.png")
        plt.close()
