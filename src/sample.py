#####
# Imports
#####
import numpy as np
import pandas as pd
import argparse

#####
# Load the CSV training file
#####
def load(csv_file):

    #Import data
    df = pd.read_csv(csv_file, header=0)

    return df


#####
# Sample the data
#####
def sample(df, pct, seed):
    df = df.sample(frac=pct, random_state=seed)
    dist = df.is_attributed.sum()/df.shape[0]
    return df, dist


#####
# Save the data
#####
def save(df, outfile):
    df.to_csv(outfile, encoding='utf-8', index=False)


#####
# Execute the training process
#####
def execute(csvfile, pct, outfile, seed ):

    print("Using csvfile: ", csvfile)
    print("Using pct:     ", pct)
    print("Using outfile: ", outfile)
    print("Using seed:    ", seed)

    np.random.seed(seed)

    print("Loading data, csvfile: ", csvfile)
    df = load(csvfile)

    print("Sampling data, pct: ", pct)
    df, dist = sample(df, pct, seed)
    print("Sampling distribution (pct is_attributed): ", dist)

    print("Saving data, outfile: ", outfile)
    save(df, outfile)

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
        args.seed = RANDOM_SEED

    # Execute the command
    execute(args.csvfile, float(args.pct), args.outfile, int(args.seed))


main()
