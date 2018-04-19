import argparse
import os
import gc

import numpy as np

# import sys
# sys.path.append('..')

from xdata import Data

#####
# Programatic default values
#####
np.set_printoptions(formatter={"float": "{: 0.5f}".format})


#####
# Execute the program
#####
def execute(trainfile, testfile):

    print("--- Executing")
    print("Using trainfile (in):  ", trainfile)
    print("Using testfile (in):   ", testfile)

    print("--- Transforming data")
    traindir, trainfilename = os.path.split(trainfile)
    trainfilename, _ = os.path.splitext(trainfilename)
    trainfilename = trainfilename + ".h5"
    trainout = traindir + "/" + "transform-" + trainfilename
    testdir, testfilename = os.path.split(testfile)
    testfilename, _ = os.path.splitext(testfilename)
    testfilename = testfilename + ".h5"
    testout = testdir + "/" + "transform-" + testfilename
    data = Data.Data()
    X_train, X_test = data.transform(trainfile=trainfile, testfile=testfile)

    print("X_train shape: ", X_train.shape)
    print("X_train columns: ", X_train.columns.values)
    print("X_train data: \n", X_train.head())
    print("X_train memory:\n", X_train.info(memory_usage='deep'))

    print("X_test shape:  ", X_test.shape)
    print("X_test columns: ", X_test.columns.values)
    print("X_test data: \n", X_test.head())
    print("X_test memory:\n", X_test.info(memory_usage='deep'))

    print("--- Saving data")
    print("Using trainfile (out): ", trainout)
    print("Using testfile (out):  ", testout)

    data.save(X_train, trainout)
    del X_train; gc.collect()

    data.save(X_test, testout)
    del X_test; gc.collect()


#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument("-t", "--trainfile", help="is the CSV file containing training data (required)")
    parser.add_argument("-T", "--testfile", help="is the CSV file containing test data (required)")
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
    if not args.trainfile:
        raise Exception("Missing argument: --trainfile")
    if not args.testfile:
        raise Exception("Missing argument: --testfile")

    # Execute the command
    execute(args.trainfile, args.testfile)


main()
