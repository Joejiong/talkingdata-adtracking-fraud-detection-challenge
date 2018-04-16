import argparse
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = "4"
import gc

# import sys
# sys.path.append('..')

from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE

from xdata import Data
# from xmodel import DenseModelOne
# from xmodel import DenseModelTwo
# from xmodel import DenseModelThree
from xmodel import DenseModelFour

#####
# Programatic default values
#####
np.set_printoptions(formatter={"float": "{: 0.5f}".format})


#####
# Execute the program
#####
def execute(trainfile, testfile, modeldir, logdir, epochs, batch_size, seed):

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
    df = data.load(trainfile)
    # print("df: ", df.shape)

    # For scoring (uses sample of training data)
    fraction = 0.1
    X_validation = df.sample(frac=fraction)
    Y_validation = X_validation["is_attributed"].values
    X_validation.drop(["is_attributed"], 1, inplace=True)
    # print("X_validation: ", X_validation.shape)

    # For training, use the part of data NOT in the validation fraction
    X_train = df.loc[~df.index.isin(X_validation.index)]
    Y_train = X_train["is_attributed"].values
    X_train.drop(["is_attributed"], 1, inplace=True)
    del df; gc.collect()

    is_oversampled = False
    if is_oversampled:
        print("Loaded X_train: ", X_train.shape)
        print("---- BEFORE:\n", X_train.head())
        print(Y_train)
        print("unique values: ", set(Y_train))

        columns = X_train.columns.values
        total = len(Y_train)
        ones = np.sum(Y_train)
        zeros = total - ones

        nration = 1.0
        nzeros = int(zeros)
        nones = int(nzeros * nration)
        ratio = {0:nzeros, 1:nones}
        print("ratio: ", ratio)

        oversampler = ADASYN(random_state=seed, ratio=ratio)
        oversampler.fit(X_train, Y_train)
        X_resampled, y_resampled = oversampler.sample(X_train, Y_train)
        X_resampled = X_resampled.astype(int)
        y_resampled = y_resampled.astype(int)
        X_train = pd.DataFrame(data=X_resampled, columns=columns)
        Y_train = y_resampled
        del X_resampled; del y_resampled; gc.collect()

        print("Oversampled (Random), ratio: ", ratio, " X_train: ", X_train.shape)
        print("---- AFTER:\n", X_train.head())
        print(Y_train)
        print("unique values: ", set(Y_train))

    X_test = data.load(testfile)

    print("X_train shape:      ", X_train.shape)
    print("Y_train shape:      ", Y_train.shape)
    print("X_test shape:       ", X_test.shape)
    print("X_validation shape: ", X_validation.shape)
    print("Y_validation shape: ", Y_validation.shape)

    print("--- Creating model")
    model = DenseModelFour.DenseModelFour()

    print("--- Configuring model")
    model.configure(X_train, X_test, X_validation)
    model.set_validation(X_validation, Y_validation)

    print("--- Training model")
    model.fit(
        X_train, Y_train, X_test,
        modeldir=modeldir, logdir=logdir,
        epochs=epochs, batch_size=batch_size)

    del X_train; gc.collect()
    del Y_train; gc.collect()
    del X_test; gc.collect()

    print("--- Scoring model")
    print("Fraction used for scoring: ", fraction)
    print("X_validation shape: ", X_validation.shape)
    print("Y_validation shape: ", Y_validation.shape)

    roc_auc, probabilities = model.score(X_validation, Y_validation)
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
