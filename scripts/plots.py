"""
Plot the fit of the regression to the Mauna Loa data nicely.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import build_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True,
                        help="Folder containing the `train.csv` & `test.csv`")
    parser.add_argument("-o", "--output", required=False, default="plots",
                        help="Directory for plots and visualizations")
    args = parser.parse_args()
    
    # load the data
    with open(f"{args.input}/train.csv", 'r') as train_file:
        train_df = pd.read_csv(train_file, index_col=0).rename({
            "average": "before 2000"
        }, axis="columns")
    with open(f"{args.input}/test.csv", 'r') as test_file:
        test_df = pd.read_csv(test_file, index_col=0).rename({
            "average": "after 2000"
        }, axis="columns")
    
    year = np.array([*train_df["year"].values, *test_df["year"].values])
    
    features = build_features(year)
    num_features = features.shape[0]
    
    # load the learned weights
    post_mean = np.load(f"{args.input}/post_mean.npy")
    post_covmat = np.load(f"{args.input}/post_covmat.npy")
    
    # compute posterior predictive
    post_predict_mean = features.T @ post_mean
    
    fig, ax = plt.subplots(figsize=(10,8))
    train_df.plot(
        title="Monthly mean CO2 at Mauna Loa",
        x="year",
        y="before 2000",
        xlabel="Year",
        ylabel="parts per million [ppm]",
        ax=ax,
    )
    test_df.plot(
        x="year",
        y="after 2000",
        ax=ax
    )
    ax.plot(year, post_predict_mean, label="posterior predictive")
    ax.grid("on")
    ax.legend()
    fig.savefig(f"{args.output}/predictive.png", dpi=300)