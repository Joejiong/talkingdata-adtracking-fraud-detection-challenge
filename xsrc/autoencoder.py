import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = "4"
import gc

import sys
sys.path.append('..')

from sklearn.model_selection import train_test_split

from xdata import Data
from xmodel import AutoEncoderModel

#####
# Programatic default values
#####
np.set_printoptions(formatter={"float": "{: 0.5f}".format})


#####
# Execute the program
#####
def execute(trainfile, modeldir, logdir, epochs, batch_size, seed ):

    print("--- Executing")
    print("Using trainfile:  ", trainfile)
    print("Using modeldir:   ", modeldir)
    print("Using logdir:     ", logdir)
    print("Using epochs:     ", epochs)
    print("Using batch_size: ", batch_size)
    print("Using seed:       ", seed)

    print("--- Loading (transformed) data")
    data = Data.Data()
    df = data.load(trainfile)
    # print("df: ", df.shape)

    nohits = df[df.is_attributed == 0]  # 99.8% of data
    hits = df[df.is_attributed == 1]    #  0.2% of data

    # NOTE: Train on the "nohits" as they are the vast majority of data,
    # and then we will try to identify the "hits" as anomalies
    test_size = 0.3
    X_train, X_test = train_test_split(df, test_size=test_size, random_state=seed)
    Y_train = X_train["is_attributed"]
    X_train = X_train[X_train.is_attributed == 0]
    X_train = X_train.drop(["is_attributed"], axis=1)

    Y_test = X_test["is_attributed"]
    X_test = X_test.drop(["is_attributed"], axis=1)

    X_train = X_train.values
    X_test = X_test.values

    X_train, Y_train, X_test, Y_test
    print("X_train shape: ", X_train.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_test shape:  ", X_test.shape)
    print("Y_test shape:  ", Y_test.shape)

    print("--- Creating model")
    model = AutoEncoderModel.AutoEncoderModel()

    print("--- Configuring model")
    model.set_validation(X_test, Y_test)

    print("--- Training model")
    model.fit(
        X_train,
        modeldir=modeldir, logdir=logdir,
        epochs=epochs, batch_size=batch_size)

    del X_train; gc.collect()
    del Y_train; gc.collect()

    print("--- Scoring model")
    print("X_test shape: ", X_test.shape)
    print("Y_test shape: ", Y_test.shape)

    roc_auc, probabilities = model.score(X_test, Y_test)
    print("Score probabilities shape: ", probabilities.shape)
    roc_auc = "{:0.6f}".format(roc_auc)
    print("Score: ROC-AUC: ",roc_auc)

    del X_test; gc.collect()
    del Y_test; gc.collect()

    # print("--- Saving model")
    # modelfile = modeldir + "/" + "autoencoder-model-final-" + roc_auc + ".h5"
    # model.save(modelfile)
    # print("Model saved to: ", modelfile)

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
    execute(args.trainfile, args.modeldir, args.logdir, int(args.epochs), int(args.batch), int(args.seed))


main()
