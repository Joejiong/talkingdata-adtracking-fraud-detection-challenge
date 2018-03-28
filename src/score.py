#####
# Imports
#####

import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


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
    Y_val = df.is_attributed
    X_val = df.drop(['is_attributed'], axis=1)
    return X_val, Y_val


#####
# Load the model
#####
def loadMODEL(model_file):
    model = load_model(model_file, custom_objects={'kauc': kauc})
    return model


#####
# Score the model
#####
def score(model, X_val, Y_val, threshold):

    #Predict on test set
    raw_predictions = model.predict(X_val)
    probabilities = raw_predictions[:,0]
    false_positive_rate, recall, thresholds = roc_curve(Y_val, probabilities)
    roc_auc = auc(false_positive_rate, recall)
    print("ROC-AUC: {:0.6f}".format(roc_auc))
    show_score(Y_val, probabilities)
    return roc_auc, probabilities



def show_score(Y_val, probabilities):

    #Turn probability to 0-1 binary output
    threshold = 0.5
    binary_score = np.where(probabilities > threshold, 1, 0)
    # predictions_NN_01 = predictions_NN_prob

    print('Accuracy:  ', accuracy_score(Y_val, binary_score))
    print('Precision: ', precision_score(Y_val, binary_score))
    print('Recall:    ', recall_score(Y_val, binary_score))
    print('F1:        ', f1_score(Y_val, binary_score))
    print('\nClassification Report:\n', classification_report(Y_val, binary_score))

    # Show Area Under Curve (key evaluation metric)
    roc_auc = roc_auc_score(Y_val, probabilities)
    print("ROC-AUC: ", roc_auc)

    # Since this is a classifier, show the confusion matrix
    show_confusion_matrix(Y_val, binary_score, ["FRAUD", "OK"], with_graphics=False)


#####
# Show confusion matrix in text and graphical format
#####
def show_confusion_matrix(Y_actual, Y_prediction, class_names, with_graphics=False):
    """
    Show the confusion matrix, for text and graphical output.
    """

    cm = confusion_matrix(Y_actual, Y_prediction)
    print("Raw Confusion Matrix: \n", cm)
    ncm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized Confusion Matrix: \n", ncm)

    if with_graphics:
        # Plot raw confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix, without normalization')
        plt.show()

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(ncm, classes=class_names, normalize=True, title='Normalized confusion matrix')
        plt.show()


#####
# Plot confusion matrix in text and graphical format
#####
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Print and plot the confusion matrix.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#####
# Execute the training process
#####
def execute(csvfile, modelfile, threshold):

    print("Using csvfile:        ", csvfile)
    print("Using modelfile:      ", modelfile)
    print("Using threshold:      ", threshold)

    print("Loading training validation data, csvfile: ", csvfile)
    X_val, Y_val = loadCSV(csvfile)

    print("Loading model: ", modelfile)
    model = loadMODEL(modelfile)

    print("Making prediction")
    score(model, X_val, Y_val, threshold)


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
    parser.add_argument("-l", "--modelfile", help="is the fully qualified filename for the model HDF5 file (required)")
    parser.add_argument("-t", "--threshold", help="is the threshold value (optional)")
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
    if not args.threshold:
        raise Exception("Missing argument: --threshold")

    # Execute the command
    execute(args.csvfile, args.modelfile, float(args.threshold))


main()
