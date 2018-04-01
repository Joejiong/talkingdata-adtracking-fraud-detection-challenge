import argparse
import numpy as np

import sys
sys.path.append('..')

from xdata import Data
from xmodel import DenseModel

#####
# Programatic default values
#####
np.set_printoptions(formatter={"float": "{: 0.5f}".format})


#####
# Execute the program
#####
def execute(trainfile, testfile, seed):

    print("--- Executing")
    print("Using trainfile:  ", trainfile)
    print("Using testfile:   ", testfile)
    print("Using seed:       ", seed)

    print("--- Transforming data")
    data = Data.Data()
    X_train, X_test, Y_train = data.transform(trainfile=trainfile, testfile=testfile)
    print("X_train shape: ", X_train.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_test shape:  ", X_test.shape)

    print("--- Saving data")
    modelfile = modeldir + "/" + "dense-model-final-" + roc_auc + ".h5"
    model.save(modelfile)
    print("Data saved to: ", modelfile)


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
    if not args.trainfile:
        raise Exception("Missing argument: --trainfile")
    if not args.testfile:
        raise Exception("Missing argument: --testfile")
    if not args.seed:
        args.seed = RANDOM_SEED

    # Execute the command
    execute(args.trainfile, args.testfile, int(args.seed))


main()
