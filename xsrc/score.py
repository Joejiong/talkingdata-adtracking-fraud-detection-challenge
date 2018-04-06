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
def execute(infile, modelfile, fraction, seed):

    print("--- Executing")
    print("Using infile:   ", infile)
    print("Using modelfile: ", modelfile)
    print("Using fraction:  ", fraction)
    print("Using seed:      ", seed)

    print("--- Loading data file")
    data = Data.Data()
    X_train = data.load(infile)

    print("--- Loading model")
    model = DenseModel.DenseModel()
    model.load(modelfile)
    print("Model loaded from: ", modelfile)

    print("--- Scoring model")
    X_fraction = X_train.sample(frac=fraction, random_state=seed)
    Y_fraction = X_fraction["is_attributed"].values
    print("X_fraction shape: ", X_fraction.shape)
    print("Y_fraction shape: ", Y_fraction.shape)

    roc_auc, probabilities = model.score(X_fraction, Y_fraction)
    print("Score probabilities shape: ", probabilities.shape)
    roc_auc = "{:0.6f}".format(roc_auc)
    print("Score: ROC-AUC: ",roc_auc)


#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument("-t", "--infile", help="is the HDF5 file containing transformed training data (required)")
    parser.add_argument("-m", "--modelfile", help="is the fully qualified file to the model (required)")
    parser.add_argument("-f", "--fraction", help="is the fraction of data from the infile to be sampled (required)")
    parser.add_argument("-s", "--seed", help="is the random number seed (required)")
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
    if not args.modelfile:
        raise Exception("Missing argument: --modelfile")
    if not args.fraction:
        raise Exception("Missing argument: --fraction")
    if not args.seed:
        args.seed = 42

    # Execute the command
    execute(args.infile, args.modelfile, float(args.fraction), int(args.seed))


main()
