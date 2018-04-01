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
def execute(csvfile, modelfile):

    print("--- Executing")
    print("Using csvfile:  ", csvfile)
    print("Using modelfile:   ", modelfile)

    print("--- Loading model")
    model = DenseModel.DenseModel(modelfile=modelfile, logdir=logdir)
    model.load(modelfile)
    print("Model loaded from: ", modelfile)

    print("--- Scoring model")
    X_sample = X_train.sample(frac=0.10)
    Y_sample = X_sample["is_attributed"].values
    X_sample.drop(['click_id', 'click_time', 'ip', 'is_attributed'], 1, inplace=True)
    X_sample = model.convert(X_sample)

    roc_auc, probabilities = model.score(X_sample, Y_sample)
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
    parser.add_argument("-t", "--csvfile", help="is the CSV file containing training data (required)")
    parser.add_argument("-m", "--modelfile", help="is the fully qualified directory to store checkpoint models (required)")
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
    if not args.modelfile:
        raise Exception("Missing argument: --modelfile")

    # Execute the command
    execute(args.csvfile, args.modelfile)


main()
