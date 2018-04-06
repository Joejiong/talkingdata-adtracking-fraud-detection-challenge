#####
# Imports
#####
import argparse
import numpy as np
import pandas as pd


#####
# Defaults
#####
np.set_printoptions(formatter={"float": "{:0.5f}".format})


#####
# Execute the training process
#####
def execute(infile, pct, outfile, seed):

    np.random.seed(seed)

    print("Using infile:  ", infile)
    print("Using pct:     ", pct)
    print("Using outfile: ", outfile)
    print("Using seed:    ", seed)

    df = pd.read_hdf(infile, "table")
    print("Input rows, columns: ", df.shape)
    df = df.sample(frac=pct, random_state=seed)
    print("Sampled rows, columns: ", df.shape)

    if "is_attributed" in df.columns.values:
        zeros = df.is_attributed.value_counts()[0]
        ones = df.is_attributed.value_counts()[1]
        total = zeros + ones
        dist = ones/total
        print("STATISTICS: ZEROS: {:10} ONES: {:10} TOTAL: {:10} DIST: {:0.5f}".format(zeros, ones, total, dist))

    compressor = "blosc"
    df.to_hdf(outfile, "table", mode="w", complevel=9, complib=compressor)


#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument("-i", "--infile", help="is the CSV file containing training data (required)")
    parser.add_argument("-t", "--pct", help="is the percentage of data to be retrieved (required)")
    parser.add_argument("-l", "--outfile", help="is the fully qualified filename for the retrieved data (required)")
    parser.add_argument("-s", "--seed", help="is the random seed (optional, default: 0)")
    args = parser.parse_args()
    return args


#####
# Mainline program
#####
def main():
    """
    Mainline
    """

    # Get the command line parameters
    args = cli()
    if not args.infile:
        raise Exception("Missing argument: --infile")
    if not args.pct:
        raise Exception("Missing argument: --pct")
    if not args.outfile:
        raise Exception("Missing argument: --outfile")
    if not args.seed:
        args.seed = 42

    # Execute the command
    execute(args.infile, float(args.pct), args.outfile, int(args.seed))


main()
