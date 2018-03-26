#####
# Imports
#####

import numpy as np
import pandas as pd
import argparse
import tensorflow as tf

from keras import backend as K
from keras.models import load_model


#####
# Constants
#####


#####
# Defaults
#####


#####
# Execute the training process
#####
def execute(csvfile, feature, value, output):

    print("Using csvfile: ", csvfile)
    print("Using feature: ", feature)
    print("Using value:   ", value)
    print("Using output:  ", output)

    chunksize = 100000
    chunknum = 0
    print("Loading chunks of csvfile: ", csvfile, ", chuncksize: ", chunksize)
    for chunk in pd.read_csv(csvfile, header=0, chunksize=chunksize):
        print("\n--- Chunk: ", chunknum)

        # Note DROPPING columns means keeping opposites
        # is_not_value = chunk.drop(chunk[chunk.is_attributed == value].index)
        is_value = chunk.drop(chunk[chunk[feature] != value].index)

        with_header = False
        if chunknum == 0:
            with_header = True

        print("Feature: ", feature, ", value: ", str(value), ", saved to: ", output)
        with open(output, "a") as f:
            is_value.to_csv(f, header=with_header, index=False)

        # is_not_value_file = "is_not_value.csv"
        # print("Feature: ", feature, ", NOT value: ", str(value), ", saved to: ", is_not_value_file)
        # with open(is_not_value_file, "a") as f:
        #     is_not_value.to_csv(f, header=with_header, index=False)

        chunknum = chunknum + 1


#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="filter CSV for specific feature/value with output to stdout")
    parser.add_argument("-c", "--csvfile", help="is the CSV file containing test/submission data (required)")
    parser.add_argument("-f", "--feature", help="is the feature name for which to filter (required)")
    parser.add_argument("-x", "--value", help="is the value of the filter (required)")
    parser.add_argument("-o", "--output", help="is the CSV file containing filtered values (required)")
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
    if not args.feature:
        raise Exception("Missing argument: --feature")
    if not args.value:
        raise Exception("Missing argument: --value")
    if not args.output:
        raise Exception("Missing argument: --output")

    # Execute the command
    execute(args.csvfile, args.feature, int(args.value), args.output)


main()
