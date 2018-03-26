#####
# Imports
#####
import pandas as pd
import numpy as np
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
import itertools
# from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_auc_score,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout
from keras import regularizers
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, BaseLogger
from scipy.spatial.distance import euclidean

#####
# Constants
#####
LABELS = ["Normal", "Fraud"]

LAYER1_SIZE = 200
TEST_PCT = 0.3

ES_PATIENCE = 3
ES_MIN_DELTA = 1e-5

RLRP_PATIENCE = 3
RLRP_MIN_LR = 1e-5
RLRP_FACTOR = 0.5

OPT_LEARNING_RATE = 1.0*1e-6
OPT_DECAY = 1e-7

#####
# Defaults
#####
sns.set(style="whitegrid", palette="muted", font_scale=1.5)
# rcParams["figure.figsize"] = 14, 8
np.set_printoptions(formatter={"float": "{: 0.5f}".format})


#####
# Plot training/validation history
#####
def show_history(history):
    # summarize history for loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
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
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.plot(false_positive_rate, recall, "b", label = "AUC = %0.3f" %roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0,1], [0,1], "r--")
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel("Recall")
        plt.xlabel("Fall-out (1-Specificity)")
        plt.show()


#####
# Show confusion matrix in text and graphical format
#####
def show_confusion_matrix(Y_actual, Y_prediction, class_names, with_graphics=False):
    """
    Show the confusion matrix, for text and graphical output.
    """

    cm = confusion_matrix(Y_actual, Y_prediction)
    print("Raw Confusion Matrix: \n", cm)
    ncm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized Confusion Matrix: \n", ncm)

    if with_graphics:
        # Plot raw confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names, title="Confusion matrix, without normalization")
        plt.show()

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(ncm, classes=class_names, normalize=True, title="Normalized confusion matrix")
        plt.show()


#####
# Plot confusion matrix in text and graphical format
#####
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    """
    Print and plot the confusion matrix.
    """

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".4f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


#####
# Load the CSV test/validation file
#####
def load_validation(csv_file):
    #Import data
    df = pd.read_csv(csv_file, header=0)

    num_click_ids = len(df.values)
    # print("num_click_ids: ", num_click_ids)
    click_ids = np.arange(num_click_ids)
    if "click_id" in df.columns.values:
        print("Capturing click_ids")
        click_ids = df.click_id
        df = df.drop(["click_id"], axis=1)

    # print("click_ids: ", click_ids)

    print("columns: ", df.columns.values)
    if "is_attributed" in df.columns.values:
        print("Dropping is_attributed when loading validation set")
        df = df.drop(["is_attributed"], axis=1)

    values = df.values
    # print("df: ", df.shape)
    return values, click_ids


#####
# Load the CSV training file
#####
def load_train(csv_file, seed):

    #Import data
    df = pd.read_csv(csv_file, header=0)
    print(df.is_attributed.value_counts())

    nohits = df[df.is_attributed == 0]  # 99.8% of data
    hits = df[df.is_attributed == 1]    #  0.2% of data

    # NOTE: Train on the "nohits" as they are the vast majority of data,
    # and then we will try to identify the "hits" as anomalies
    X_train, X_test = train_test_split(df, test_size=TEST_PCT, random_state=seed)
    Y_train = X_train["is_attributed"]
    X_train = X_train[X_train.is_attributed == 0]
    X_train = X_train.drop(["is_attributed"], axis=1)

    Y_test = X_test["is_attributed"]
    X_test = X_test.drop(["is_attributed"], axis=1)

    X_train = X_train.values
    X_test = X_test.values

    X_train, Y_train, X_test, Y_test
    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)
    print("Y_train: ", Y_train.shape)
    print("Y_test: ", Y_test.shape)

    return X_train, Y_train, X_test, Y_test, df


#####
# Create the model
#####
def create_model(input_dim, encoding_dim):

    output_dim = input_dim

    # autoencoder = Sequential([
    #     Dense(units=input_dim, input_dim=input_dim, activation="tanh"),
    #     Dense(encoding_dim, activation="relu"),
    #     Dense(int(encoding_dim / 2), activation="relu"),
    #     Dense(int(encoding_dim / 2), activation="tanh"),
    #     Dense(output_dim, activation="relu")
    #     ])
    # autoencoder.summary()

    l1 = LAYER1_SIZE
    l2 = int(l1*1.25)
    l3 = int(l2*0.5)
    l4 = int(l3*0.5)
    l5 = int(l4*0.5)
    l6 = int(l5*0.5)
    d1 = 0.25
    d2 = 0.25
    d3 = 0.25
    d4 = 0.5
    d5 = 0.5
    seed = 0

    autoencoder = Sequential([
        Dense(units=l1, input_dim=input_dim, kernel_initializer="normal", activation="tanh"),
        Dropout(d1, seed=seed),
        Dense(units=l2, activation="tanh"),
        Dropout(d2, seed=seed),
        # Dense(l3, kernel_initializer="normal", activation="relu"),
        # Dropout(d3, seed=seed),
        # Dense(l4, kernel_initializer="normal", activation="tanh"),
        # Dropout(d4, seed=seed),
        # Dense(l5, kernel_initializer="normal", activation="tanh"),
        # Dropout(d5, seed=seed),
        # Dense(l6, kernel_initializer="normal", activation="tanh"),
        # Dropout(l6, seed=seed),
        Dense(output_dim, kernel_initializer="uniform", activity_regularizer=regularizers.l1(1e-5), activation="sigmoid")
    ])
    # autoencoder.summary()

    # input_layer = Input(shape=(input_dim, ))
    # # encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
    # encoder = Dense(encoding_dim, activation="tanh")(input_layer)
    # encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
    # decoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
    # decoder = Dense(input_dim, activation="relu")(decoder)
    # autoencoder = Model(inputs=input_layer, outputs=decoder)

    optimizer = optimizers.adam(lr=OPT_LEARNING_RATE, decay=OPT_DECAY)

    # autoencoder.compile(optimizer=optimizer, loss="mean_squared_logarithmic_error", metrics=["accuracy"])
    autoencoder.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"])

    autoencoder.summary()

    return autoencoder


#####
# Train the model
#####
def train(model, X_train, X_test, epochs, batch_size, modeldir, logdir):
    modelpath = modeldir + "/" + "autoencoder_epoch-{epoch:02d}_loss-{loss:.4f}_valloss-{val_loss:.4f}.h5"
    CHK = ModelCheckpoint(filepath=modelpath, verbose=0, save_best_only=True)
    # CHK = ModelCheckpoint(modelpath, monitor="val_kauc", verbose=1, save_best_only=True, save_weights_only=False, mode="max", period=1)
    TB = TensorBoard(log_dir=logdir,
                     histogram_freq=0,
                     write_graph=True,
                     write_images=False)
    ES = EarlyStopping(monitor="val_loss", min_delta=ES_MIN_DELTA, patience=ES_PATIENCE, verbose=2, mode="auto")
    TB = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)
    RLRP = ReduceLROnPlateau(monitor="val_loss", factor=RLRP_FACTOR, patience=RLRP_PATIENCE, min_lr=RLRP_MIN_LR)
    BL = BaseLogger()

    # NOTE: training on X_train/X_train (not Y_train) since autoencoder
    # tries to train only on good data (errors in prediction will be bad values).
    # Similarly, the validation set is X_test/X_test.
    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)
    history = model.fit(X_train, X_train,
                        epochs=epochs, batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=[CHK, ES, RLRP, TB])

    # history = model.fit(
    #             train_x, train_y,
    #             epochs=epochs, batch_size=batch_size,
    #             class_weight=class_weight,
    #             callbacks=[ES, TB, BL, RLRP, roc_callback])

    return history


#####
# Score the model
#####
def score(model, X_test, Y_test):
    print("X_test: ", X_test.shape)
    print("Y_test: ", Y_test.shape)

    predictions = model.predict(X_test)
    print("predictions: ", predictions.shape)

    xerr = np.linalg.norm(X_test - predictions, axis=1)
    # print("err (1): ", err)
    err = np.mean(np.power(X_test - predictions, 3), axis=1)
    # print("err (2): ", err)

    print("err: ", err)
    error_df = pd.DataFrame({"reconstruction_error": err, "true_class": Y_test})

    print("error_df:\n", error_df.describe())
    print("error_df:\n", error_df)
    filename = "reconstruction_error.csv"
    error_df.to_csv(filename)
    print("saved file to: ", filename)

    # threshold = 2
    # predictions = np.where(error_df["reconstruction_error"] > threshold, 1, 0)
    # print("predictions:\n", predictions)

    average = np.average(err)
    stddev = np.std(err)
    tstart = int( (average - 1*stddev)*0.1 )
    tend = int( (average + 2.0*stddev)*0.1 )

    print("--- average:  ", average)
    print("--- stddev:   ", stddev)
    print("--- tstart:   ", tstart)
    print("--- tend:     ", tend)

    roc_auc = 0.0
    best_roc_auc = 0.0
    best_predictions = None
    best_threshold = 0
    items = (tend-tstart)/0.1

    thresholds = np.linspace(tstart, tend, items)
    for threshold in thresholds:

        # 0 == nohit
        # 1 == hit
        nohit_error_df = error_df[error_df["true_class"] == 0]
        # print("nohit_error_df:\n", nohit_error_df)
        hit_error_df = error_df[(error_df["true_class"]== 1) & (error_df["reconstruction_error"] < threshold)]
        # print("hit_error_df:\n", hit_error_df)

        predictions = np.where(error_df["reconstruction_error"] < threshold, 1, 0)
        # print("reconstruction_error: ", error_df["reconstruction_error"])
        # print("predictions: ", predictions)

        # false_positive_rate, true_positive_rate, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(error_df.true_class, predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        # print("threshold: {:0.4f} ROC-AUC: {:0.6f}".format(threshold, roc_auc))

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_predictions = predictions
            best_threshold = threshold
            avg_mult = threshold / average
            stddev_mult = threshold / stddev
            print("MAX -- threshold: {:0.4f} ROC-AUC: {:0.6f}".format(threshold, roc_auc))

    # Show Area Under Curve (key evaluation metric)
    show_AUC(Y_test, best_predictions, with_graphics=False)

    # Since this is a classifier, show the confusion matrix
    show_confusion_matrix(Y_test, best_predictions, ["FRAUD", "OK"], with_graphics=False)

    print("BEST -- stddev_mult: {:4.4f} avg_mult: {:4.4f} threshold: {:0.4f} ROC-AUC: {:0.6f}".format(stddev_mult, avg_mult, best_threshold, best_roc_auc))

    return best_roc_auc, best_threshold, predictions


#####
# Predict using the model
#####
def predict(model, X_validation, threshold):
    predictions = model.predict(X_validation)
    print("predictions: ", predictions.shape)

    err = np.linalg.norm(X_validation - predictions, axis=1)

    average = np.average(err)
    stddev = np.std(err)
    rstart = int(average - 1*stddev)
    rend = int(average + 1.5*stddev)
    interval = 0.1

    print("--- average:  ", average)
    print("--- stddev:   ", stddev)
    print("--- rstart:   ", rstart)
    print("--- rend:     ", rend)
    print("--- interval: ", interval)

    avg_mult = threshold / average
    stddev_mult = threshold / stddev

    print("ASSUME -- stddev_mult: {:4.4f} avg_mult: {:4.4f} threshold: {:0.4f}".format(stddev_mult, avg_mult, threshold))

    error_df = pd.DataFrame({"reconstruction_error": err})
    print("error_df:\n", error_df)
    filename = "reconstruction_error.csv"
    error_df.to_csv(filename)
    print("saved file to: ", filename)

    predictions = np.where(error_df["reconstruction_error"] < threshold, 1, 0)

    return predictions


#####
# Save the model
#####
def save_model(model, modelpath):
    model.save(modelpath)


#####
# Execute the training process
#####
def execute(csvfile, modeldir, logdir, epochs, batch_size, seed, operation, threshold):

    print("Using LAYER1_SIZE:       ", LAYER1_SIZE)
    print("Using ES_PATIENCE:       ", ES_PATIENCE)
    print("Using ES_MIN_DELTA:      ", ES_MIN_DELTA)
    print("Using RLRP_PATIENCE:     ", RLRP_PATIENCE)
    print("Using RLRP_MIN_LR:       ", RLRP_MIN_LR)
    print("Using RLRP_FACTOR:       ", RLRP_FACTOR)
    print("Using OPT_LEARNING_RATE: ", OPT_LEARNING_RATE)
    print("Using OPT_DECAY:         ", OPT_DECAY)
    print("Using csvfile:           ", csvfile)
    print("Using modeldir:          ", modeldir)
    print("Using logdir:            ", logdir)
    print("Using epochs:            ", epochs)
    print("Using batch_size:        ", batch_size)
    print("Using seed:              ", seed)
    print("Using operation:         ", operation)
    print("Using threshold:         ", threshold)

    np.random.seed(seed)

    if "train" in operation:

        print("Loading data, csvfile: ", csvfile)
        # csv_file = "../data/train_sample.csv"
        X_train, Y_train, X_test, Y_test, df = load_train(csvfile, seed)

        print("Creating model")
        input_dim = X_train.shape[1]
        encoding_dim = int(input_dim/25)
        print("Using input_dim:  ", input_dim)
        print("Using encoding_dim: ", encoding_dim)
        model = create_model(input_dim, encoding_dim)

        print("Training model")
        history = train(model, X_train, X_test, epochs, batch_size, modeldir, logdir)

        # print("Showing history")
        # show_history(history)

        print("Scoring model")
        best_roc_auc, best_threshold, best_predictions = score(model, X_test, Y_test)

        modelpath = modeldir + "/" + "autoencoder-final.h5"
        print("Saving final model: ", modelpath)
        save_model(model, modelpath)

        return

    if "predict" in operation:
        print("Loading data, csvfile: ", csvfile)
        # csv_file = "../data/train_sample.csv"
        X_validation, click_ids = load_validation(csvfile)

        modelpath = modeldir + "/" + "autoencoder-final.h5"
        print("Loading model: ", modelpath)
        model = load_model(modelpath)

        print("Making predictions...")
        predictions = predict(model, X_validation, threshold)
        print("Predictions: ", predictions)

        # pcsv = "autoencoder-predictions-" + str(int(threshold)) + ".csv"
        pcsv = "autoencoder-predictions.csv"
        print("Saving predictions: ", pcsv)
        pdf = pd.DataFrame({"prediction": predictions})
        pdf.to_csv(pcsv, encoding='utf-8')

        return


#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument("-C", "--csvfile", help="is the CSV file containing training data (required)")
    parser.add_argument("-M", "--modeldir", help="is the fully qualified directory to store checkpoint models (required)")
    parser.add_argument("-L", "--logdir", help="is the fully qualified directory where the tensorboard logs will be saved (required)")
    parser.add_argument("-E", "--epochs", help="is the number of epochs (optional, default: 100)")
    parser.add_argument("-B", "--batch", help="is the batch size (optional, default: 1000)")
    parser.add_argument("-S", "--seed", help="is the random seed (optional, default: 0)")
    parser.add_argument("-O", "--operation", help="is the set of operations to perform (comma separated, one of: train|predict)")
    parser.add_argument("-T", "--threshold", help="is the threshold for errors used in predictions")
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
    EPOCHS = 10
    BATCH_SIZE = 100
    RANDOM_SEED = 42
    OPERATION = "train"
    THRESHOLD = 25
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
    if not args.operation:
        args.operation = OPERATION
    if not args.threshold:
        args.threshold = THRESHOLD

    # Execute the command
    execute(args.csvfile, args.modeldir, args.logdir, int(args.epochs), int(args.batch), int(args.seed), args.operation, float(args.threshold))


main()
