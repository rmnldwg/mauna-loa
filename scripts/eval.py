"""
Evaluate the Gaussian parametric regression using the `test.csv` data and store 
the result in `metrics.json` file.
"""

import argparse
import json
import yaml

import pandas as pd
import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import cho_factor, cho_solve

from features import build_features, noise_precision



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", 
                        required=True,
                        help=("Folder where the test data `test.csv`, as well "
                              "as the `post_mean.npy` & `post_covmat.npy` "
                              "arrays can be found"))
    parser.add_argument("-m", "--metrics", 
                        required=False, default="metrics.json",
                        help="Path to the metrics JSON file")
    args = parser.parse_args()
    
    # load the test data
    with open(f"{args.input}/test.csv", 'r') as test_file:
        test_df = pd.read_csv(test_file, index_col=0)
    
    year = test_df["year"].values
    targets = test_df["average"].values
    
    # features for test data
    features = build_features(year)
    num_features = features.shape[0]
    
    # load the learned weights
    post_mean = np.load(f"{args.input}/post_mean.npy")
    post_covmat = np.load(f"{args.input}/post_covmat.npy")
    
    # compute posterior predictive
    post_predict_mean = features.T @ post_mean
    post_predict_covmat = (
        features.T @ post_covmat @ features 
        + np.eye(len(targets)) * noise_precision
    )
    
    # compute log-evidence of data given posterior predictive
    diff = targets - post_predict_mean
    cho_decomp = cho_factor(post_predict_covmat)
    sol = cho_solve(cho_decomp, diff)
    _, occam_factor = slogdet(post_predict_covmat)
    log_evi = - 0.5 * diff @ sol - 0.5 * occam_factor
    
    # store metrics
    metrics = {
        "log-evidence": log_evi,
        "RMSE": np.sqrt(np.mean(diff @ diff))
    }
    with open(f"{args.metrics}", 'w') as metrics_file:
        json.dump(metrics, metrics_file)
    