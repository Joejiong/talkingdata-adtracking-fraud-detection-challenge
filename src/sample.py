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
def execute(csvfile, pct, outfile, seed):

    np.random.seed(seed)

    print("Using csvfile:  ", csvfile)
    print("Using pct:      ", pct)
    print("Using outfile:  ", outfile)
    print("Using seed:     ", seed)

    # Large files sizes require chunking (otherwise memory usage errors occur)
    chunksize = 1e6
    print("Using chunksize:", chunksize)
    chunknum = 0
    for chunk in pd.read_csv(csvfile, header=0, chunksize=chunksize):
        chunk = chunk.sample(frac=pct, random_state=seed)

        if "is_attributed" in chunk.columns.values:
            zeros = chunk.is_attributed.value_counts()[0]
            ones = chunk.is_attributed.value_counts()[1]
            total = zeros + ones
            dist = ones/total
            print("CHUNK: {:7} ZEROS: {:7} ONES: {:7} TOTAL: {:8} DIST: {:0.5f}".format(chunknum, zeros, ones, total, dist))

        with_header = False
        if chunknum == 0:
            with_header = True

        with open(outfile, "a") as f:
            chunk.to_csv(f, encoding="utf-8", header=with_header, index=False)

        chunknum = chunknum + 1


#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument("-c", "--csvfile", help="is the CSV file containing training data (required)")
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
    if not args.csvfile:
        raise Exception("Missing argument: --csvfile")
    if not args.pct:
        raise Exception("Missing argument: --pct")
    if not args.outfile:
        raise Exception("Missing argument: --outfile")
    if not args.seed:
        args.seed = 42

    # Execute the command
    execute(args.csvfile, float(args.pct), args.outfile, int(args.seed))


main()
