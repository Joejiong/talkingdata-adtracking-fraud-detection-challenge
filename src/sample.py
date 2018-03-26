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

    print("Using csvfile: ", csvfile)
    print("Using pct:     ", pct)
    print("Using outfile: ", outfile)
    print("Using seed:    ", seed)

    np.random.seed(seed)

    # Large files sizes require chunking (otherwise memory usage errors occur)
    chunksize = 10000000
    chunknum = 0
    for chunk in pd.read_csv(csvfile, header=0, chunksize=chunksize):
        print("\n--- Chunk: ", chunknum, ", size: ", chunksize)
        chunk = chunk.sample(frac=pct, random_state=seed)
        count = chunk.is_attributed.sum()
        dist = count/chunk.shape[0]
        print("Count: {:5} Distribution: {:0.5f}".format(count, dist))

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
