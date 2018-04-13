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

class DenseModel:

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

    LAYER1_SIZE = 1000

    def __init__(self, seed=0):
        # Initialize model

        self._seed = seed
        self._model = None
        self._X_validation = None
        self._Y_validation = None
        self._config = {}


    def convert(self, df):
        X = {
            'app': np.array(df.app),
            'ch': np.array(df.channel),
            'dev': np.array(df.device),
            'os': np.array(df.os),
            'h': np.array(df.hour),
            'dqh': np.array(df.dqhour),
            'qh': np.array(df.qhour),
            'd': np.array(df.day),
            'wd': np.array(df.wday),
            'qty': np.array(df.qty),
            'c1': np.array(df.ip_app_count),
            'c2': np.array(df.ip_app_os_count),
            'c3': np.array(df.new_column_1),
            'c4': np.array(df.new_column_2),
            'c5': np.array(df.new_column_3)
        }
        return X


    def set_validation(self, X_validation, Y_validation):
        self._X_validation = X_validation
        self._Y_validation = Y_validation


    def configure(self, X_train, X_test, X_validation):
        max_app = np.max([X_train['app'].max(), X_test['app'].max(), X_validation['app'].max()])+1
        max_ch = np.max([X_train['channel'].max(), X_test['channel'].max(), X_validation['channel'].max()])+1
        max_dev = np.max([X_train['device'].max(), X_test['device'].max(), X_validation['device'].max()])+1
        max_os = np.max([X_train['os'].max(), X_test['os'].max(), X_validation['os'].max()])+1
        max_h = np.max([X_train['hour'].max(), X_test['hour'].max(), X_validation['hour'].max()])+1
        max_dqh = np.max([X_train['dqhour'].max(), X_test['dqhour'].max(), X_validation['dqhour'].max()])+1
        max_qh = np.max([X_train['qhour'].max(), X_test['qhour'].max(), X_validation['qhour'].max()])+1
        max_d = np.max([X_train['day'].max(), X_test['day'].max(), X_validation['day'].max()])+1
        max_wd = np.max([X_train['wday'].max(), X_test['wday'].max(), X_validation['wday'].max()])+1
        max_qty = np.max([X_train['qty'].max(), X_test['qty'].max(), X_validation['qty'].max()])+1
        max_c1 = np.max([X_train['ip_app_count'].max(), X_test['ip_app_count'].max(), X_validation['ip_app_count'].max()])+1
        max_c2 = np.max([X_train['ip_app_os_count'].max(), X_test['ip_app_os_count'].max(), X_validation['ip_app_os_count'].max()])+1
        max_c3 = np.max([X_train['new_column_1'].max(), X_test['new_column_1'].max(), X_validation['new_column_1'].max()])+1
        max_c4 = np.max([X_train['new_column_2'].max(), X_test['new_column_2'].max(), X_validation['new_column_2'].max()])+1
        max_c5 = np.max([X_train['new_column_3'].max(), X_test['new_column_3'].max(), X_validation['new_column_3'].max()])+1

        self._config["max_app"] = max_app
        self._config["max_ch"] = max_ch
        self._config["max_dev"] = max_dev
        self._config["max_os"] = max_os
        self._config["max_h"] = max_h
        self._config["max_qh"] = max_qh
        self._config["max_dqh"] = max_dqh
        self._config["max_d"] = max_d
        self._config["max_wd"] = max_wd
        self._config["max_qty"] = max_qty
        self._config["max_c1"] = max_c1
        self._config["max_c2"] = max_c2
        self._config["max_c3"] = max_c3
        self._config["max_c4"] = max_c4
        self._config["max_c5"] = max_c5

        return


    def fit(self, X_train, Y_train, X_test, modeldir=".", logdir=".", epochs=2, batch_size=20000):

        max_app = self._config["max_app"]
        max_ch = self._config["max_ch"]
        max_dev = self._config["max_dev"]
        max_os = self._config["max_os"]
        max_h = self._config["max_h"]
        max_dqh = self._config["max_dqh"]
        max_qh = self._config["max_qh"]
        max_d = self._config["max_d"]
        max_wd = self._config["max_wd"]
        max_qty = self._config["max_qty"]
        max_c1 = self._config["max_c1"]
        max_c2 = self._config["max_c2"]
        max_c3 = self._config["max_c3"]
        max_c4 = self._config["max_c4"]
        max_c5 = self._config["max_c5"]

        X_train = self.convert(X_train)

        # Establish the network
        emb_n = 100
        dense_n = 1000
        in_app = Input(shape=[1], name='app')
        emb_app = Embedding(max_app, emb_n)(in_app)
        in_ch = Input(shape=[1], name='ch')
        emb_ch = Embedding(max_ch, emb_n)(in_ch)
        in_dev = Input(shape=[1], name='dev')
        emb_dev = Embedding(max_dev, emb_n)(in_dev)
        in_os = Input(shape=[1], name='os')
        emb_os = Embedding(max_os, emb_n)(in_os)
        in_h = Input(shape=[1], name='h')
        emb_h = Embedding(max_h, emb_n)(in_h)
        in_dqh = Input(shape=[1], name='dqh')
        emb_dqh = Embedding(max_dqh, emb_n)(in_dqh)
        in_qh = Input(shape=[1], name='qh')
        emb_qh = Embedding(max_qh, emb_n)(in_qh)
        in_d = Input(shape=[1], name='d')
        emb_d = Embedding(max_d, emb_n)(in_d)
        in_wd = Input(shape=[1], name='wd')
        emb_wd = Embedding(max_wd, emb_n)(in_wd)
        in_qty = Input(shape=[1], name='qty')
        emb_qty = Embedding(max_qty, emb_n)(in_qty)
        in_c1 = Input(shape=[1], name='c1')
        emb_c1 = Embedding(max_c1, emb_n)(in_c1)
        in_c2 = Input(shape=[1], name='c2')
        emb_c2 = Embedding(max_c2, emb_n)(in_c2)
        in_c3 = Input(shape=[1], name='c3')
        emb_c3 = Embedding(max_c3, emb_n)(in_c3)
        in_c4 = Input(shape=[1], name='c4')
        emb_c4 = Embedding(max_c4, emb_n)(in_c4)
        in_c5 = Input(shape=[1], name='c5')
        emb_c5 = Embedding(max_c5, emb_n)(in_c5)
        fe = concatenate([
            (emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h), (emb_qh), (emb_dqh),
            (emb_d), (emb_wd), (emb_qty), (emb_c1), (emb_c2), (emb_c3), (emb_c4), (emb_c5)])
        s_dout = SpatialDropout1D(0.2)(fe)
        fl = Flatten()(s_dout)
        x = Dropout(0.2)(Dense(dense_n,activation='relu')(fl))
        x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
        gl = MaxPooling1D(pool_size=1, strides=1)(s_dout)
        fl = Flatten()(gl)
        x = concatenate([(x), (fl)])
        outp = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h,in_dqh,in_qh,in_d,in_wd,in_qty,in_c1,in_c2,in_c3,in_c4,in_c5], outputs=outp)

        # Compile the model
        exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
        steps = int(len(X_train) / batch_size) * epochs
        # lr_init, lr_fin = 0.001, 0.0001
        # lr_init, lr_fin = 0.00005, 0.000005
        # lr_decay = exp_decay(lr_init, lr_fin, steps)
        # lr_decay = 0.0000001
        # lr = lr_init
        # optimizer_adam = optimizers.Adam(lr=lr, decay=lr_decay)
        # lr = DenseModel.OPT_LEARNING_RATE
        # lr_decay = DenseModel.OPT_DECAY
        # optimizer_adam = optimizers.adam(lr=lr, decay=lr_decay)
        # print("Using learning init rate: ", lr, ", decay: ", lr_decay)
        lr = DenseModel.OPT_LEARNING_RATE
        optimizer_adam = optimizers.Adam(lr=lr)
        print("Using learning init rate: ", lr)
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer_adam,
            metrics=["accuracy"])

        model.summary()

        self._model = model

        # Establish class weights since this is a very unbalanced dataset
        class_weight = dict(zip([0, 1], compute_class_weight('balanced', [0, 1], Y_train)))
        print("Unbalanced data, actual class_weight: ", class_weight)
        class_weight = {0:.01, 1:.99}
        print("Unbalanced data, using  class_weight: ", class_weight)

        # Establish callbacks for training
        modelpath = modeldir + '/' + 'dense-model-checkpoint.h5'
        CHK = ModelCheckpoint(
            modelpath, monitor="val_loss", verbose=1,
            save_best_only=True, save_weights_only=False,
            mode="auto", period=1)
        ES = EarlyStopping(
            monitor="val_loss", min_delta=DenseModel.ES_MIN_DELTA,
            patience=DenseModel.ES_PATIENCE, verbose=2, mode="auto")
        TB = TensorBoard(
            log_dir=logdir, histogram_freq=0,
            write_graph=True, write_images=False)
        RLRP = ReduceLROnPlateau(
            monitor="val_loss", factor=DenseModel.RLRP_FACTOR,
            patience=DenseModel.RLRP_PATIENCE, min_lr=DenseModel.RLRP_MIN_LR,
            verbose=1)
        BL = BaseLogger()
        # ROC = ROCCallback.ROCCallback()

        callbacks=[ES, TB, BL, RLRP, CHK]
        if self._X_validation is not None:
            print("Using AUCCallback...")
            AUC = AUCCallback.AUCCallback(self, self._X_validation, self._Y_validation, convert=True)
            callbacks.append(AUC)

        validation_data = None
        if self._X_validation is not None:
            print("Using validation_data (overrides validation_split)...")
            X_validation = self.convert(self._X_validation)
            Y_validation = self._Y_validation
            validation_data = (X_validation, Y_validation)

        # Train the model
        # (NOTE: if validation_data exists then it will OVERRIDE the validation split)
        history = self._model.fit(
            X_train, Y_train, batch_size=batch_size, epochs=epochs,
            validation_split=0.2,
            validation_data=validation_data,
            class_weight=class_weight,
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
        X_test = self.convert(X_test)
        predictions = self._model.predict(X_test)
        return predictions


    def save(self, modelfile):
        # Save model
        self._model.save(modelfile)


    def load(self, modelfile):
        # Load model
        self._model = load_model(modelfile)
