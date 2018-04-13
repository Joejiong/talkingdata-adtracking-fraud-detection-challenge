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
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE

from xdata import Data

#####
# Programatic default values
#####
np.set_printoptions(formatter={"float": "{: 0.5f}".format})


#####
# Execute the program
#####
def execute(trainfile, sampler):

    print("--- Executing")
    print("Using trainfile:  ", trainfile)

    print("--- Loading (transformed) data")
    data = Data.Data()
    train_df = data.load(trainfile)
    y = train_df["is_attributed"]
    X = train_df.drop( ["is_attributed"], axis=1 )
    columns = X.columns.values

    before_class_weight = dict(zip([0, 1], compute_class_weight('balanced', [0, 1], y)))
    print("Original weights: ", before_class_weight)

    X_resampled = None
    y_resampled = None
    if sampler == "RANDOM":
        oversampler = RandomOverSampler(random_state=0)
        oversampler.fit(X, y)
        X_resampled, y_resampled = oversampler.sample(X, y)

    elif sampler == "ADASYN":
        oversampler = ADASYN(random_state=0)
        oversampler.fit(X, y)
        X_resampled, y_resampled = oversampler.sample(X, y)

    elif sampler == "SMOTE":
        oversampler = SMOTE(random_state=0)
        oversampler.fit(X, y)
        X_resampled, y_resampled = oversampler.sample(X, y)

    else:
        print("Invalid sampler: ", sampler)

    after_class_weight = dict(zip([0, 1], compute_class_weight('balanced', [0, 1], y_resampled)))
    print("Sampler: ", sampler, ", weights: ", after_class_weight)

    X_resampled = X_resampled.astype(int)
    y_resampled = y_resampled.astype(int)

    # print("X_resampled: ", X_resampled)
    # print("y_resampled: ", y_resampled)

    df = pd.DataFrame(data=X_resampled, columns=columns)
    df["is_attributed"] = y_resampled
    # df["is_attributed"] = df["is_attributed"].astype(int)

    compressor = "blosc"
    outfilename = trainfile + "." + sampler
    print("Output file (over-sampled): ", outfilename)
    df.to_hdf(outfilename, "table", mode="w", append=True, complevel=9, complib=compressor)


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
    parser.add_argument("-x", "--sampler", help="is the type of over-sampler to use (RANDOM|ADASYN|SMOTE, required)")
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
    if not args.sampler:
        raise Exception("Missing argument: --sampler")

    # Execute the command
    execute(args.trainfile, args.sampler)


main()
