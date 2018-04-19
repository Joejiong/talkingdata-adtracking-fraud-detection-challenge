#####
# Imports
#####
import argparse
import os
import gc
import numpy as np
import pandas as pd


#####
# Defaults
#####
os.environ["OMP_NUM_THREADS"] = "4"
np.set_printoptions(formatter={"float": "{:0.5f}".format})


#####
# Execute the training process
#####
def execute(infile, pcts, seed):

    np.random.seed(seed)
    pcts = pcts.split(",")

    print("Using infile:  ", infile)
    print("Using pcts:    ", pcts)
    print("Using seed:    ", seed)

    df = pd.read_hdf(infile, "table")
    print("Input rows, columns: ", df.shape)
    print("df memory:\n", df.info(memory_usage='deep'))

    for pct in pcts:
        pct = float(pct)
        sampledf = df.sample(frac=pct, random_state=seed)
        print("Sampled rows, columns: ", df.shape)
        print("Sampled column names: ", df.columns.values)

        spct = "{:0.3f}".format(pct)

        if "is_attributed" in sampledf.columns.values:
            print(sampledf.is_attributed.value_counts())
            zeros = sampledf.is_attributed.value_counts()[0]
            ones = sampledf.is_attributed.value_counts()[1]
            total = zeros + ones
            dist = ones/total
            print("STATISTICS PCT: {:5}\n: ZEROS: {:10} ONES: {:10} TOTAL: {:10} DIST: {:0.5f}".format(spct, zeros, ones, total, dist))

        outdir, outfilename = os.path.split(infile)
        outfilepart, outfileext = os.path.splitext(outfilename)
        outfilepath = outdir + "/" + outfilepart + "-" + spct + outfileext

        print("Saving to file: ", outfilepath)
        compressor = "blosc"
        sampledf.to_hdf(outfilepath, "table", mode="w", complevel=9, complib=compressor)
        del sampledf; gc.collect()

        print("Saved to file:  ", outfilepath)


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
    parser.add_argument("-p", "--pcts", help="is the comma delimited list of percentage to be sampled (required)")
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
    if not args.pcts:
        raise Exception("Missing argument: --pcts")
    if not args.seed:
        args.seed = 0

    # Execute the command
    execute(args.infile, args.pcts, int(args.seed))


main()
