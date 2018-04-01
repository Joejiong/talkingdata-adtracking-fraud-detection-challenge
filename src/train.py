#####
# Imports
#####
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers
# from keras import regularizers
from keras.callbacks import Callback
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, BaseLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import compute_class_weight

#####
# Constants
#####
ES_PATIENCE = 7
ES_MIN_DELTA = 1e-4

RLRP_PATIENCE = 7
RLRP_MIN_LR = 1e-6
RLRP_FACTOR = 0.25

OPT_LEARNING_RATE = 1.0*1e-5
OPT_DECAY = 1e-7

#####
# Defaults
#####
EPOCHS = 100
BATCH_SIZE = 1000
RANDOM_SEED = 0

LAYER1_SIZE = 1000

#####
# Some programatic default values
#####
np.set_printoptions(formatter={"float": "{: 0.5f}".format})
sns.set_style("whitegrid")


#####
# Create the model
#####
def create_model(input_dim, output_dim, seed):

    l1 = LAYER1_SIZE
    l2 = int(l1*1.0)
    l3 = int(l2*0.5)
    l4 = int(l3*0.5)
    l5 = int(l4*0.5)
    l6 = int(l5*0.5)
    d1 = 0.5
    d2 = 0.5
    d3 = 0.5
    d4 = 0.5
    d5 = 0.5

    # kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)
    model = Sequential([
        Dense(units=l1, input_dim=input_dim, kernel_initializer="normal", activation="tanh"),
        Dropout(d1, seed=seed),
        Dense(units=l2, activation="tanh"),
        Dropout(d2, seed=seed),
        # Dense(l3, kernel_initializer="normal", activation="tanh"),
        # Dropout(d3, seed=seed),
        # Dense(l4, kernel_initializer="normal", activation="tanh"),
        # Dropout(d4, seed=seed),
        # Dense(l5, kernel_initializer="normal", activation="tanh"),
        # Dropout(d5, seed=seed),
        # Dense(l6, kernel_initializer="normal", activation="tanh"),
        # Dropout(l6, seed=seed),
        Dense(output_dim, kernel_initializer="uniform", activation="sigmoid")
    ])
    model.summary()

    optimizer = optimizers.adam(lr=OPT_LEARNING_RATE, decay=OPT_DECAY)
    # optimizer = optimizers.adamax(lr=OPT_LEARNING_RATE, decay=OPT_DECAY)

    # Compile the model
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # model.compile(optimizer="adam", loss="binary_crossentropy")
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[kauc])

    print("Available metrics: ", model.metrics_names)

    return model


#####
# Load the CSV training file
#####
def load(csv_file, seed):

    # Import data
    df = pd.read_csv(csv_file, header=0)
    # xdf = pd.read_csv("./data/transform-xones.csv", header=0)
    # df = df.append(xdf)


    # Add the autoencoder hints
    # hint_df = pd.read_csv("autoencoder-predictions.csv", header=0)
    # df["hint-50"] = hint_df.prediction.values

    # hint_df = pd.read_csv("autoencoder-predictions-50.csv", header=0)
    # df["hint-50"] = hint_df.prediction.values
    #
    # hint_df = pd.read_csv("autoencoder-predictions-100.csv", header=0)
    # df["hint-100"] = hint_df.prediction.values
    #
    # hint_df = pd.read_csv("autoencoder-predictions-200.csv", header=0)
    # df["hint-200"] = hint_df.prediction.values
    #
    # hint_df = pd.read_csv("autoencoder-predictions-300.csv", header=0)
    # df["hint-300"] = hint_df.prediction.values

    # Check num of cases in label
    print(df.is_attributed.value_counts()) #very imbalanced data set

    # #Create new variables
    # df['ip_cut'] = pd.cut(df.ip,15)
    # df['time_interval'] = df.click_time.str[11:13]
    #
    # #Drop unneeded variables
    # df = df.drop(['ip', 'attributed_time', 'click_time'], axis = 1)
    #
    # #Encode categorical variables to ONE-HOT
    # categorical_columns = ['app', 'device', 'os', 'channel', 'ip_cut', 'time_interval']
    # df = pd.get_dummies(df, columns = categorical_columns)

    #Split train test set
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=seed)

    #Make sure labels are equally distributed in train and test set
    train_dist = train_df.is_attributed.sum()/train_df.shape[0] #0.2233
    test_dist = test_df.is_attributed.sum()/test_df.shape[0] #0.2148
    print("Training set distribution (is_attributed/total): ", train_dist)
    print("Test set distribution (is_attributed/total):     ", test_dist)

    #Get the data ready for the Neural Network
    train_y = train_df.is_attributed
    test_y = test_df.is_attributed

    train_x = train_df.drop(['is_attributed'], axis=1)
    test_x = test_df.drop(['is_attributed'], axis=1)

    train_x =np.array(train_x)
    test_x = np.array(test_x)

    train_y = np.array(train_y)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y, df


#####
# ROC callback used to show AUC-ROC during model fit (training)
#####
class ROC_Callback(Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0])
        score = roc_auc_score(self.validation_data[1], y_pred)
        self.aucs.append(score)
        print("ROC-AUC Score: ", score)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


#####
# Plot training/validation history
#####
def show_history(history):
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


#####
# Show AUC in text and graphical format
#####
def show_AUC(Y_actual, Y_prediction, with_graphics=False):
    false_positive_rate, recall, thresholds = roc_curve(Y_actual, Y_prediction)
    roc_auc = auc(false_positive_rate, recall)
    print("ROC-AUC: ", roc_auc)

    if with_graphics:
        plt.figure()
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], 'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out (1-Specificity)')
        plt.show()

    return roc_auc


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
# Keras AUC for a binary classifier
#####
def kauc(y_true, y_pred):
   score, up_opt = tf.metrics.auc(y_true, y_pred)
   #score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score


#####
# Tensorflow native AUC for a binary classifier -- sometimes gies NAN, use kauc (above)
#####
def xauc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    score = K.sum(s, axis=0)
    return score


#####
# PFA, prob false alert for binary classifier
#####
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N


#####
# P_TA prob true alerts for binary classifier
#####
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


#####
# Train the model
#####
def train(model, train_x, train_y, epochs, batch_size, modeldir, logdir):

    class_weight = dict(zip([0, 1], compute_class_weight('balanced', [0, 1], train_y)))
    print("Unbalanced data, actual class_weight: ", class_weight)
    # class_weight = {0: 1, 1: 1000}
    print("Unbalanced data, using class_weight: ", class_weight)

    modelpath = modeldir + '/' + 'model_epoch-{epoch:02d}_loss-{loss:.4f}_valloss-{val_loss:.4f}_val_kauc-{val_kauc:.4f}.h5'
    # CHK = ModelCheckpoint(modelpath, monitor="val_kauc", verbose=1, save_best_only=True, save_weights_only=False, mode="max", period=1)
    ES = EarlyStopping(monitor="val_kauc", min_delta=ES_MIN_DELTA, patience=ES_PATIENCE, verbose=2, mode="max")
    TB = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)
    RLRP = ReduceLROnPlateau(monitor="val_loss", factor=RLRP_FACTOR, patience=RLRP_PATIENCE, min_lr=RLRP_MIN_LR)
    BL = BaseLogger()

    roc_callback = ROC_Callback()

    history = model.fit(
                train_x, train_y,
                validation_split=0.2,
                epochs=epochs, batch_size=batch_size,
                class_weight=class_weight,
                callbacks=[ES, TB, BL, RLRP, roc_callback])

    return history


#####
# Score the model
#####
def score(model, test_x, test_y):

    #Predict on test set
    raw_predictions = model.predict(test_x)
    probabilities = raw_predictions[:,0]
    false_positive_rate, recall, thresholds = roc_curve(test_y, probabilities)
    roc_auc = auc(false_positive_rate, recall)
    print("INITIAL ROC-AUC: {:0.6f}".format(roc_auc))
    return roc_auc, probabilities

    #####
    # The rest of the code below is experimentation to get a better ROC by thresholding
    # probabilities -- it did not help
    #####

    roc_auc = 0.0
    best_roc_auc = 0.0
    best_predictions = None
    best_threshold = 0

    thresholds = np.linspace(0.0, 1.0, 99)
    for threshold in thresholds:

        # print("\n\n-------------- Threshold: ", threshold)
        #Turn probability to 0-1 binary output
        predictions = np.where(probabilities > threshold, 1, 0)
        # predictions = probabilities

        # print('Accuracy:  ', accuracy_score(test_y, predictions))
        # print('Precision: ', precision_score(test_y, predictions))
        # print('Recall:    ', recall_score(test_y, predictions))
        # print('F1:        ', f1_score(test_y, predictions))
        # print('\nClassification Report:\n', classification_report(test_y, predictions))

        # Show Area Under Curve (key evaluation metric)
        false_positive_rate, recall, thresholds = roc_curve(test_y, predictions)
        roc_auc = auc(false_positive_rate, recall)
        # roc_auc = show_AUC(test_y, predictions, with_graphics=False)

        # Since this is a classifier, show the confusion matrix
        # show_confusion_matrix(test_y, predictions, ["FRAUD", "OK"], with_graphics=False)

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_predictions = predictions
            best_threshold = threshold
            print("MAX -- threshold: {:0.4f} ROC-AUC: {:0.6f}".format(threshold, roc_auc))

    return best_roc_auc, best_predictions


#####
# Save the model
#####
def save_model(model, modelpath):
    model.save(modelpath)


#####
# Execute the training process
#####
def execute(csvfile, modeldir, logdir, epochs, batch_size, seed ):

    print("Using LAYER1_SIZE:       ", LAYER1_SIZE)
    print("Using ES_PATIENCE:       ", ES_PATIENCE)
    print("Using ES_MIN_DELTA:      ", ES_MIN_DELTA)
    print("Using RLRP_PATIENCE:     ", RLRP_PATIENCE)
    print("Using RLRP_MIN_LR:       ", RLRP_MIN_LR)
    print("Using RLRP_FACTOR:       ", RLRP_FACTOR)
    print("Using OPT_LEARNING_RATE: ", OPT_LEARNING_RATE)
    print("Using OPT_DECAY:         ", OPT_DECAY)
    print("Using random seed:       ", seed)

    np.random.seed(seed)

    print("Loading data, csvfile: ", csvfile)
    # csv_file = '../data/train_sample.csv'
    train_x, train_y, test_x, test_y, df = load(csvfile, seed)

    print("Creating model")
    input_dim = train_x.shape[1]
    output_dim = 1
    print("Using input_dim:  ", input_dim)
    print("Using output_dim: ", output_dim)

    model = create_model(input_dim, output_dim, seed)

    print("Training model")
    history = train(model, train_x, train_y, epochs, batch_size, modeldir, logdir)

    print("Showing history")
    # show_history(history)

    print("Scoring model")
    roc_auc, predictions = score(model, test_x, test_y)
    x = "%0.6f" % roc_auc

    print("Final ROC-AUC: ", roc_auc)

    # click_ids = np.arange(len(predictions))
    # predictionspath = "predictions-" + x + ".csv"
    # print("Saving predictions: ", predictionspath)
    # save_submission(click_ids, predictions, predictionspath)

    modelpath = modeldir + "/" + "model-final-auc-" + x + ".h5"
    print("Saving final model: ", modelpath)
    save_model(model, modelpath)


#####
# Create submission
#####
def save_submission(click_ids, predictions, submission_file):
    submission = pd.DataFrame()
    submission["click_id"] = click_ids
    submission["is_attributed"] = predictions

    with open(submission_file, "a") as f:
        submission.to_csv(f, header=True, index=False)

    return submission


#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument("-c", "--csvfile", help="is the CSV file containing training data (required)")
    parser.add_argument("-t", "--modeldir", help="is the fully qualified directory to store checkpoint models (required)")
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
    if not args.csvfile:
        raise Exception("Missing argument: --csvfile")
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
    execute(args.csvfile, args.modeldir, args.logdir, int(args.epochs), int(args.batch), int(args.seed))


main()
