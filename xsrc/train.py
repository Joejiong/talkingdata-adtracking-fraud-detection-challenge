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
def execute(trainfile, testfile, modeldir, logdir, epochs, batch_size, seed ):

    print("--- Executing")
    print("Using trainfile:  ", trainfile)
    print("Using testfile:   ", testfile)
    print("Using modeldir:   ", modeldir)
    print("Using logdir:     ", logdir)
    print("Using epochs:     ", epochs)
    print("Using batch_size: ", batch_size)
    print("Using seed:       ", seed)

    print("--- Loading (transformed) data")
    data = Data.Data()
    X_train = data.load(trainfile)
    Y_train = X_train["is_attributed"].values

    # For scoring (uses sample of training data)
    fraction = 0.5
    X_fraction = X_train.sample(frac=fraction, random_state=seed)
    Y_fraction = X_fraction["is_attributed"].values

    X_train.drop(["is_attributed"], 1, inplace=True)
    X_test = data.load(testfile)

    print("X_train shape: ", X_train.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_test shape:  ", X_test.shape)

    print("--- Creating model")
    model = DenseModel.DenseModel()

    print("--- Training model")
    model.fit(
        X_train, X_test, Y_train,
        modeldir=modeldir, logdir=logdir,
        epochs=epochs, batch_size=batch_size)

    print("--- Scoring model")
    print("Fraction used for scoring: ", fraction)
    print("X_fraction shape: ", X_fraction.shape)
    print("Y_fraction shape: ", Y_fraction.shape)

    roc_auc, probabilities = model.score(X_fraction, Y_fraction)
    print("Score probabilities shape: ", probabilities.shape)
    roc_auc = "{:0.6f}".format(roc_auc)
    print("Score: ROC-AUC: ",roc_auc)

    print("--- Saving model")
    modelfile = modeldir + "/" + "dense-model-final-" + roc_auc + ".h5"
    model.save(modelfile)
    print("Model saved to: ", modelfile)

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
    parser.add_argument("-m", "--modeldir", help="is the fully qualified directory to store checkpoint models (required)")
    parser.add_argument("-l", "--logdir", help="is the fully qualified directory where the tensorboard logs will be saved (required)")
    parser.add_argument("-e", "--epochs", help="is the number of epochs (optional, default: 100)")
    parser.add_argument("-b", "--batch", help="is the batch size (optional, default: 1000)")
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
    if not args.modeldir:
        raise Exception("Missing argument: --modeldir")
    if not args.logdir:
        raise Exception("Missing argument: --logdir")
    if not args.epochs:
        args.epochs = EPOCHS
    if not args.batch:
        args.batch = BATCH_SIZE
    if not args.seed:
        args.seed = RANDOM_SEED

    # Execute the command
    execute(args.trainfile, args.testfile, args.modeldir, args.logdir, int(args.epochs), int(args.batch), int(args.seed))


main()
