#####
# Imports
#####

import numpy as np
import pandas as pd
import argparse

from xdata import Data
from xmodel import DenseModel

#####
# Constants
#####


#####
# Defaults
#####



#####
# Create submission
#####
def save_submission(click_ids, predictions, submission_file, with_header):
    print("Using click_ids: ", click_ids.shape)
    print("Using predictions: ", predictions.shape)

    submission = pd.DataFrame()
    submission["click_id"] = click_ids
    submission["is_attributed"] = predictions

    with open(submission_file, "a") as f:
        submission.to_csv(f, header=with_header, index=False)

    return submission


#####
# Execute the training process
#####
def execute(infile, submissionfile, modelfile):

    print("Using infile:        ", infile)
    print("Using submissionfile: ", submissionfile)
    print("Using modelfile:      ", modelfile)

    print("--- Loading model")
    model = DenseModel.DenseModel()
    model.load(modelfile)
    print("Model loaded from: ", modelfile)

    chunksize = 1000000
    chunknum = 0
    print("Loading chunks of test/submission data, infile: ", infile, ", chunksize: ", chunksize)
    for chunk in pd.read_hdf(infile, "table", chunksize=chunksize):
        print("\n--- Chunk: ", chunknum)
        print("columns: ", chunk.columns.values)

        click_ids = None
        if "click_id" in chunk.columns.values:
            print("Column click_id present")
            click_ids = chunk["click_id"].astype('int')
            chunk = chunk.drop(["click_id"], axis=1)

        if "is_attributed" in chunk.columns.values:
            print("Column is_attributed present")
            # Temporary -- transform.py bug kept this column in test data
            # rows, _ = chunk.shape
            # click_ids = np.arange(rows)
            chunk = chunk.drop(["is_attributed"], axis=1)

        print("Making prediction")
        predictions = model.predict(chunk)

        with_header = False
        if chunknum == 0:
            with_header = True

        print("Saving submission, submission: ", submissionfile)
        save_submission(click_ids, predictions, submissionfile, with_header)

        chunknum = chunknum + 1


#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument("-i", "--infile", help="is the input HDF5 file containing test/submission data (required)")
    parser.add_argument("-s", "--submissionfile", help="is the fully qualified CSV filename to store the submission output (required)")
    parser.add_argument("-m", "--modelfile", help="is the fully qualified filename for the model HDF5 file (required)")
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
    if not args.submissionfile:
        raise Exception("Missing argument: --submissionfile")
    if not args.modelfile:
        raise Exception("Missing argument: --modelfile")

    # Execute the command
    execute(args.infile, args.submissionfile, args.modelfile)


main()
