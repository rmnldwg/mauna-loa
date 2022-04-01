"""
Do a Gaussian parametric regression with the training data (CO2 averages from 
the beginning to a certain year `K`) and evaluate the prediction using the 
test data (CO2 averages from the year `K` to today).
"""

import argparse
import yaml

import pandas as pd
import numpy as np
import scipy as sp
from scipy.linalg import cho_factor, cho_solve


def build_features(year, cycle_freq, scale):
    """Compute feature vector for a given year."""
    rescaled = scale * year
    return np.array([
        [1.] * rescaled,
        rescaled**4,
        np.sin(rescaled * cycle_freq),
        np.exp(rescaled),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True,
                        help="Training data CSV file")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to store the posterior weights & covariance")
    parser.add_argument("-p", "--params", required=True,
                        help="Path to parameter YAML file")
    args = parser.parse_args()
    
    with open(args.params, 'r') as params_file:
        params = yaml.safe_load(params_file)
    
    scale = params["scale"]
    noise_precision = params["train"]["noise_precision"]
    prior_precision = params["train"]["prior_precision"]
    cycle_freq = params["train"]["cycle_freq"]
    
    with open(args.input, 'r') as train_file:
        train_df = pd.read_csv(train_file, index_col=0)
    
    year = train_df["year"].values
    target = train_df["average"].values
    
    features = build_features(
        year=year,
        scale=scale,
        cycle_freq=cycle_freq,
    )
    num_features = features.shape[0]
    
    # prior on the weights
    prior_mean = np.zeros(num_features)
    prior_covmat = np.eye(num_features) / prior_precision
    
    # implied prior on function
    mean_fn = features.T @ prior_mean
    kernel_fn = features.T @ prior_covmat @ features
    
    # the 'error' of the data
    llh_covmat = np.eye(features.shape[1]) / noise_precision
    
    # invert a part efficiently and stably
    cho_decomp = cho_factor(kernel_fn + llh_covmat * (1e-6 + 1.))
    sol = cho_solve(cho_decomp, (prior_covmat @ features).T).T
    
    # posterior mean and covariance
    post_mean = prior_mean + sol @ (target - features.T @ prior_mean)
    post_covmat = prior_covmat - sol @ features.T @ prior_covmat
    
    # save arrays to file
    np.save(f"{args.output}/post_mean", post_mean)
    np.save(f"{args.output}/post_covmat", post_covmat)