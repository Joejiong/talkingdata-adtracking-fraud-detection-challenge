#####
# Imports
#####

import numpy as np
import pandas as pd
import argparse

from xfeature import Feature

#####
# Constants
#####


#####
# Defaults
#####


#####
# Execute the training process
#####
def execute(infile):
    print("Using infile: ", infile)

    feature = Feature.Feature()

    ranks = {}

    X, y = feature.assemble(infile)

    f = feature.ridge(X, y)
    ranks["RIDGE"] = dict(sorted(f.items(), key=lambda x: x[1], reverse=True))

    f = feature.lasso(X, y)
    ranks["LASSO"] = dict(sorted(f.items(), key=lambda x: x[1], reverse=True))

    f = feature.stability(X, y)
    ranks["STABL"] = dict(sorted(f.items(), key=lambda x: x[1], reverse=True))

    # Takes a very long time
    # f = feature.mine(X, y)
    # ranks["MINEx"] = dict(sorted(f.items(), key=lambda x: x[1], reverse=True))

    # Takes a very long time
    # f = feature.recursive(X, y)
    # ranks["RECUR"] = dict(sorted(f.items(), key=lambda x: x[1], reverse=True))

    f = feature.importance(X, y)
    ranks["IMPOR"] = dict(sorted(f.items(), key=lambda x: x[1], reverse=True))

    f = feature.univariate(X, y)
    ranks["UNIVA"] = dict(sorted(f.items(), key=lambda x: x[1], reverse=True))

    f = feature.pca(X, y)
    ranks["PCAxx"] = dict(sorted(f.items(), key=lambda x: x[1], reverse=True))

    f = feature.regression(X, y)
    ranks["REGRE"] = dict(sorted(f.items(), key=lambda x: x[1], reverse=True))

    r = {}
    names = X.columns.values
    for name in names:
        r[name] = round(np.nanmean([ ranks[method][name] for method in ranks.keys() ]), 6)

    methods = sorted(ranks.keys())
    ranks["MEANx"] = r
    methods.append("MEANx")

    print("{:<25}\t\t{}".format("  ", "\t".join(methods)))
    for name in names:
        # stats = "\t".join(map(str, [ranks[method][name] for method in methods]))
        stats = " ".join(map(str, [ "\t{:6.3f}".format(ranks[method][name]) for method in methods ]))
        print("{:<25}\t{}".format(name, stats))


#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="feature description and analysis script")
    parser.add_argument("-i", "--infile", help="is the input CSV file containing test/submission data (required)")
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

    # Execute the command
    execute(args.infile)


main()
