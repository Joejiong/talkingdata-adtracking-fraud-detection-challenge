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

    print("--- Transforming data")
    data = Data.Data()
    X_train, X_test, Y_train = data.transform(trainfile=trainfile, testfile=testfile)
    print("X_train shape: ", X_train.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_test shape:  ", X_test.shape)

    print("--- Creating model")
    model = DenseModel.DenseModel(modeldir=modeldir, logdir=logdir)

    print("--- Training model")
    model.fit(X_train, X_test, Y_train, epochs=epochs, batch_size=batch_size)

    print("--- Scoring model")
    X_sample = X_train.sample(frac=0.10)
    Y_sample = X_sample["is_attributed"].values
    X_sample.drop(['click_id', 'click_time', 'ip', 'is_attributed'], 1, inplace=True)
    X_sample = model.convert(X_sample)

    roc_auc, probabilities = model.score(X_sample, Y_sample)
    print("Score probabilities shape: ", probabilities.shape)
    roc_auc = "{:0.6f}".format(roc_auc)
    print("Score: ROC-AUC: ",roc_auc)

    print("--- Saving model")
    modelfile = modeldir + "/" + "dense-model-final-" + roc_auc + ".h5"
    model.save(modelfile)
    print("Model saved to: ", modelfile)

    print("--- Loading model")
    model.load(modelfile)
    print("Model loaded from: ", modelfile)

    print("--- Predicting using model")
    X_test = model.convert(X_test)
    probabilities = model.predict(X_test)
    print("Predict probabilities shape: ", probabilities.shape)


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
