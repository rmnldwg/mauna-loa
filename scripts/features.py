import numpy as np
import yaml


# load the (hyper-params)
with open("params.yaml", 'r') as params_file:
    params = yaml.safe_load(params_file)

scale = params["scale"]
loc = params["loc"]
noise_precision = params["noise_precision"]
prior_precision = params["prior_precision"]
season_shift = params["season_shift"]
cycle_len = params["cycle_len"]
cycle_shift = params["cycle_shift"]

def build_features(year):
    """Compute feature vector for a given year."""
    rescaled = scale * (year - loc)
    return np.array([
        [1.] * len(rescaled),
        rescaled**5,
        np.sin(2 * np.pi * (year + season_shift)),
        np.sin(2 * np.pi * (year + cycle_shift) / cycle_len),
        np.exp(rescaled),
    ])