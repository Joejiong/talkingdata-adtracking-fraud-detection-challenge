#####
# Imports
#####

import numpy as np
import pandas as pd
import argparse
import tensorflow as tf

from keras import backend as K
from keras.models import load_model


#####
# Constants
#####


#####
# Defaults
#####


#####
# Model artifact - Keras AUC for a binary classifier
#####
def kauc(y_true, y_pred):
   score, up_opt = tf.metrics.auc(y_true, y_pred)
   #score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score


#####
# Load the CSV test file
#####
def loadCSV(csv_file):
    df = pd.read_csv(csv_file, header=0)
    click_ids = df["click_id"]
    test_x = df.drop(["click_id"], axis=1)
    return click_ids, test_x


#####
# Load the model
#####
def loadMODEL(model_file):
    model = load_model(model_file, custom_objects={"kauc": kauc})
    return model


#####
# Make a prediction
#####
def predict(model, test_x):

    # Predict on test set
    probabilities = model.predict(test_x)[:,0]

    # Turn probability to 0-1 binary output
    # predictions = np.where(probabilities > 0.5, 1, 0)
    predictions = probabilities

    return predictions


#####
# Create submission
#####
def save_submission(click_ids, predictions, submission_file, with_header):
    submission = pd.DataFrame()
    submission["click_id"] = click_ids
    submission["is_attributed"] = predictions

    with open(submission_file, "a") as f:
        submission.to_csv(f, header=with_header, index=False)

    return submission


#####
# Execute the training process
#####
def execute(csvfile, submissionfile, modelfile):

    print("Using csvfile:        ", csvfile)
    print("Using submissionfile: ", submissionfile)
    print("Using modelfile:      ", modelfile)

    print("Loading model: ", modelfile)
    model = loadMODEL(modelfile)

    # print("Loading test/submission data, csvfile: ", csvfile)
    # df = pd.read_csv(csv_file, header=0)
    # return click_ids, test_x

    print("Loading chunks of test/submission data, csvfile: ", csvfile)
    chunksize = 100000
    chunknum = 0
    for chunk in pd.read_csv(csvfile, header=0, chunksize=chunksize):
        print("\n--- Chunk: ", chunknum)

        test_x = chunk
        click_ids = None
        if "click_id" in chunk.columns.values:
            click_ids = chunk["click_id"]
            test_x = chunk.drop(["click_id"], axis=1)

        if "is_attributed" in test_x.columns.values:
            click_ids = np.arange(chunksize)
            test_x = test_x.drop(["is_attributed"], axis=1)

        print("Making prediction")
        predictions = predict(model, test_x)

        print("Saving submission, submission: ", submissionfile)

        with_header = False
        if chunknum == 0:
            with_header = True

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
    parser.add_argument("-c", "--csvfile", help="is the CSV file containing test/submission data (required)")
    parser.add_argument("-t", "--submissionfile", help="is the fully qualified filename to store the submission data (required)")
    parser.add_argument("-l", "--modelfile", help="is the fully qualified filename for the model HDF5 file (required)")
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
    if not args.submissionfile:
        raise Exception("Missing argument: --submissionfile")
    if not args.modelfile:
        raise Exception("Missing argument: --modelfile")

    # Execute the command
    execute(args.csvfile, args.submissionfile, args.modelfile)


main()
