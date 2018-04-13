#####
#
# Adversarial Validation
#
# Modified from original blog/source:
#       http://fastml.com/adversarial-validation-part-one/
#       http://fastml.com/adversarial-validation-part-two/
#
#####

import argparse
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = "4"
import gc

# import sys
# sys.path.append("..")
from sklearn.utils import compute_class_weight

from xdata import Data

#####
# Programatic default values
#####
np.set_printoptions(formatter={"float": "{: 0.5f}".format})


#####
# Execute the program
#####
def execute(trainfile, testfile, seed ):

    print("--- Executing")
    print("Using trainfile:  ", trainfile)
    print("Using testfile:   ", testfile)
    print("Using seed:       ", seed)

    print("--- Loading (transformed) data")
    data = Data.Data()
    train_df = data.load(trainfile)
    train_df = train_df.drop( ["is_attributed"], axis=1 )
    test_df = data.load(testfile)
    test_df = test_df.drop( ["click_id"], axis=1 )

    print("--- Configuring data")
    train_df["TARGET"] = 1
    test_df["TARGET"] = 0

    # Concatenate both frames and shuffle the examples
    data = pd.concat(( train_df, test_df ))
    y = data.TARGET
    x = data.drop( ["TARGET"], axis=1 )

    # Create train/test split
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split( x, y )
    # x_test["app"] = 1
    # x_test["channel"] = 1
    # x_test["device"] = 1
    # x_test["os"] = 1
    # x_test["hour"] = 1
    # x_test["day"] = 1
    # x_test["wday"] = 1
    # x_test["qhour"] = 1
    # x_test["dqhour"] = 1
    # x_test["qty"] = 1
    # x_test["ip_app_count"] = 1
    # x_test["ip_app_os_count"] = 1
    # x_test["new_column_1"] = 1
    # x_test["new_column_2"] = 1
    # x_test["new_column_3"] = 1
    print("x_train:\n", x_train.head())
    print("x_test:\n", x_test.head())
    print("y_train:\n", y_train.head())
    print("y_test:\n", y_test.head())

    train_class_weight = dict(zip([0, 1], compute_class_weight('balanced', [0, 1], y_train)))
    print("train_class_weight: ", train_class_weight)
    test_class_weight = dict(zip([0, 1], compute_class_weight('balanced', [0, 1], y_test)))
    print("test_class_weight: ", test_class_weight)

    from sklearn.metrics import roc_auc_score as AUC
    from sklearn.metrics import accuracy_score as accuracy

    print("--- Validating: Logistic Regression")
    from sklearn.linear_model import LogisticRegression as LR
    clf = LR()
    clf.fit( x_train, y_train )
    p = clf.predict_proba( x_test )[:,1]
    print("p:\n", p[0:50])
    print("y_test:\n", y_test)
    auc = AUC( y_test, p )
    print("AUC: {:0.6%}".format( auc ))

    print("--- Validating: RandomForest Classifier")
    from sklearn.ensemble import RandomForestClassifier as RF
    clf = RF( n_estimators = 500, verbose = True, n_jobs = -1 )
    clf.fit( x_train, y_train )
    p = clf.predict_proba( x_test )[:,1]
    auc = AUC( y_test, p )
    print("AUC: {:0.6%}".format( auc ))

    print("--- Validating: Cross Validation")
    from sklearn import cross_validation as CV
    scores = CV.cross_val_score( LR(), x, y, scoring = "roc_auc", cv = 2, verbose = 1 )
    print("mean AUC: {:0.6%}, std: {:0.6%} \n".format( scores.mean(), scores.std()))



#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument("-t", "--trainfile", help="is the HDF5 file containing training data (required)")
    parser.add_argument("-m", "--testfile", help="is the fHDF5 file containing test data (required)")
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
