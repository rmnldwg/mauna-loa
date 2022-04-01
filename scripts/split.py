"""
Split the cleaned data into a training part and a testing part. Do that split 
at a `split_year` parameter.
"""

import argparse
import yaml

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True,
                        help="The cleaned CSV data file")
    parser.add_argument("-o", "--output", required=True,
                        help="Folder to store the train.csv & test.csv files")
    parser.add_argument("-p", "--params", required=True,
                        help="Parameter YAML file containing a `split_year`")
    args = parser.parse_args()
    
    # get parameter file
    with open(args.params, 'r') as param_file:
        params = yaml.safe_load(param_file)

    # open cleaned CSV data
    with open(args.input, 'r') as input_file:
        clean_df = pd.read_csv(input_file, index_col=0)
    
    # split data
    split_year = params["split_year"]
    before_idx = clean_df["year"] < split_year
    after_idx = clean_df["year"] >= split_year
    
    train_df = clean_df.loc[before_idx]
    test_df = clean_df.loc[after_idx]
    
    with open(f"{args.output}/train.csv", 'w') as train_file:
        train_df.to_csv(train_file)
    
    with open(f"{args.output}/test.csv", 'w') as test_file:
        test_df.to_csv(test_file)