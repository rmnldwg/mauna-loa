"""
Clean the raw Mauna Loa data by removing the explanation at the beginning and 
dropping not needed columns.
"""

import argparse
import pandas as pd


if __name__ == "__main__":
    # get command line arguments (only input file)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, 
                        help="Path to the input file")
    parser.add_argument("-o", "--output", required=True,
                        help="Path where the cleaned file should be stored")
    args = parser.parse_args()
    
    # open the file and get the columns we need
    with open(args.input, 'r') as raw_file:
        df = pd.read_csv(
            raw_file, 
            header=51, 
            usecols=["decimal date", "average", "interpolated"],
            index_col=False, 
        ).rename(columns={
            "decimal date": "year",
        })
    
    # save the data
    with open(args.output, 'w') as ouput_file:
        df.to_csv(ouput_file)
    
    # make a siple plot to look at the data
    ax = df.plot(
        title="Monthly mean CO2 at Mauna Loa",
        x="year", 
        y=["average", "interpolated"], 
        ylabel="parts per million [ppm]",
        figsize=(8,6)
    )
    
    # save the plot
    fig = ax.get_figure()
    fig.savefig("plots/mauna_loa.png", dpi=300)
    