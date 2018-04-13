#####
#
# AutoEncoderModel
#
# NOTE: this does not work well in current state -- should find a way
# (future work) to create embedded layers similar to high performing
# DenseModel
#
#####

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'
import gc

np.random.seed(42)

import keras
from keras import optimizers
from keras import regularizers
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, BaseLogger

from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils import compute_class_weight

from xmodel import ROCCallback
from xmodel import AUCCallback

class AutoEncoderModel:

    #####
    # Constants
    #####
    ES_PATIENCE = 3
    ES_MIN_DELTA = 1e-6

    RLRP_PATIENCE = 1
    RLRP_MIN_LR = 1e-8
    RLRP_FACTOR = 0.25

    OPT_LEARNING_RATE = 1.0*1e-3
    OPT_DECAY = 1e-4


    def __init__(self, seed=0):
        # Initialize model

        self._seed = seed
        self._model = None
        self._X_validation = None
        self._Y_validation = None
        self._config = {}


    def set_validation(self, X_validation, Y_validation):
        self._X_validation = X_validation
        self._Y_validation = Y_validation


    def fit(self, X_train, modeldir=".", logdir=".", epochs=2, batch_size=20000):

        input_dim = X_train.shape[1]
        encoding_dim = int(input_dim/2)
        output_dim = input_dim
        print("Using input_dim:  ", input_dim)
        print("Using encoding_dim: ", encoding_dim)
        print("Using output_dim:  ", output_dim)

        L1 = 1000
        L2 = int(L1*2.25)
        L3 = int(L2*0.5)
        L4 = int(L3*0.5)
        L5 = int(L4*0.5)
        L6 = int(L5*0.5)
        D1 = 0.25
        D2 = 0.25
        D3 = 0.25
        D4 = 0.5
        D5 = 0.5
        seed = 0

        model = Sequential([
            Dense(units=L1, input_dim=input_dim, kernel_initializer="normal", activation="tanh"),
            Dropout(D1, seed=seed),
            Dense(units=L2, activation="tanh"),
            Dropout(D2, seed=seed),
            # Dense(L3, kernel_initializer="normal", activation="tanh"),
            # Dropout(D3, seed=seed),
            # Dense(L4, kernel_initializer="normal", activation="tanh"),
            # Dropout(D4, seed=seed),
            # Dense(l\L5, kernel_initializer="normal", activation="tanh"),
            # Dropout(D5, seed=seed),
            # Dense(L6, kernel_initializer="normal", activation="tanh"),
            # Dropout(L6, seed=seed),
            # Dense(output_dim, kernel_initializer="uniform", activity_regularizer=regularizers.l1(1e-5), activation="sigmoid")
            Dense(output_dim, kernel_initializer="uniform", activation="sigmoid")
        ])

        # model = Sequential([
        #     Dense(units=l1, input_dim=input_dim, activation="tanh"),
        #     Dense(encoding_dim, activation="relu"),
        #     Dense(int(encoding_dim / 2), activation="relu"),
        #     Dense(int(encoding_dim / 2), activation="tanh"),
        #     Dense(output_dim, activation="relu")
        #     ])

        optimizer = optimizers.adam(lr=AutoEncoderModel.OPT_LEARNING_RATE)
        # model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"])
        model.compile(optimizer=optimizer, loss="mean_squared_logarithmic_error", metrics=["accuracy"])
        model.summary()
        self._model = model

        # Establish callbacks for training
        modelpath = modeldir + '/' + 'autoencoder-model-checkpoint.h5'
        CHK = ModelCheckpoint(
            modelpath, monitor="val_loss", verbose=1,
            save_best_only=True, save_weights_only=False,
            mode="auto", period=1)
        ES = EarlyStopping(
            monitor="val_loss", min_delta=AutoEncoderModel.ES_MIN_DELTA,
            patience=AutoEncoderModel.ES_PATIENCE, verbose=2, mode="auto")
        TB = TensorBoard(
            log_dir=logdir, histogram_freq=0,
            write_graph=True, write_images=False)
        RLRP = ReduceLROnPlateau(
            monitor="val_loss", factor=AutoEncoderModel.RLRP_FACTOR,
            patience=AutoEncoderModel.RLRP_PATIENCE, min_lr=AutoEncoderModel.RLRP_MIN_LR,
            verbose=1)
        BL = BaseLogger()
        # ROC = ROCCallback.ROCCallback()

        # callbacks=[ES, TB, BL, RLRP, CHK]
        callbacks=[ES, TB, BL, RLRP]
        if self._X_validation is not None:
            print("Using AUCCallback...")
            AUC = AUCCallback.AUCCallback(self, self._X_validation, self._Y_validation, convert=False)
            callbacks.append(AUC)

        print("X_train: ", X_train.shape)

        validation_data = None
        # if self._X_validation is not None:
        #     print("Using validation_data (overrides validation_split)...")
        #     X_validation = self._X_validation
        #     Y_validation = self._Y_validation
        #     validation_data = (X_validation, Y_validation)
        #     print("X_validation: ", X_validation.shape)
        #     print("Y_validation: ", Y_validation.shape)

        # Train the model
        # (NOTE: if validation_data exists then it will OVERRIDE the validation split)
        history = self._model.fit(
            X_train, X_train, batch_size=batch_size, epochs=epochs,
            validation_split=0.2,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=True,
            verbose=1)

        return history


    def summary(self):
        # Save model
        self._model.summary()


    def score(self, X_test, Y_test):
        # Score model
        raw_predictions = self.predict(X_test)
        probabilities = raw_predictions[:, 0]
        false_positive_rate, recall, thresholds = roc_curve(Y_test, probabilities)
        roc_auc = auc(false_positive_rate, recall)
        return roc_auc, probabilities


    def predict(self, X_test):
        # Predict model
        predictions = self._model.predict(X_test)
        return predictions


    def save(self, modelfile):
        # Save model
        self._model.save(modelfile)


    def load(self, modelfile):
        # Load model
        self._model = load_model(modelfile)
