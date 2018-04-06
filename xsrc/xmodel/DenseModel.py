import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'
import gc

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

class DenseModel:

    #####
    # Constants
    #####
    ES_PATIENCE = 3
    ES_MIN_DELTA = 1e-6

    RLRP_PATIENCE = 1
    RLRP_MIN_LR = 1e-8
    RLRP_FACTOR = 0.25

    OPT_LEARNING_RATE = 1.0*1e-5
    OPT_DECAY = 1e-6

    LAYER1_SIZE = 1000

    def __init__(self, seed=0):
        # Initialize model

        self._seed = seed
        self._model = None


    def convert(self, df):
        X = {
            'app': np.array(df.app),
            'ch': np.array(df.channel),
            'dev': np.array(df.device),
            'os': np.array(df.os),
            'h': np.array(df.hour),
            'd': np.array(df.day),
            'wd': np.array(df.wday),
            'qty': np.array(df.qty),
            'c1': np.array(df.ip_app_count),
            'c2': np.array(df.ip_app_os_count)
        }
        return X


    def fit(self, X_train, X_test, Y_train, modeldir=".", logdir=".", epochs=2, batch_size=20000):

        # Establish the network
        max_app = np.max([X_train['app'].max(), X_test['app'].max()])+1
        max_ch = np.max([X_train['channel'].max(), X_test['channel'].max()])+1
        max_dev = np.max([X_train['device'].max(), X_test['device'].max()])+1
        max_os = np.max([X_train['os'].max(), X_test['os'].max()])+1
        max_h = np.max([X_train['hour'].max(), X_test['hour'].max()])+1
        max_d = np.max([X_train['day'].max(), X_test['day'].max()])+1
        max_wd = np.max([X_train['wday'].max(), X_test['wday'].max()])+1
        max_qty = np.max([X_train['qty'].max(), X_test['qty'].max()])+1
        max_c1 = np.max([X_train['ip_app_count'].max(), X_test['ip_app_count'].max()])+1
        max_c2 = np.max([X_train['ip_app_os_count'].max(), X_test['ip_app_os_count'].max()])+1

        X_train = self.convert(X_train)

        emb_n = 50
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
        fe = concatenate([
            (emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h),
            (emb_d), (emb_wd), (emb_qty), (emb_c1), (emb_c2)])
        s_dout = SpatialDropout1D(0.2)(fe)
        fl = Flatten()(s_dout)
        x = Dropout(0.2)(Dense(dense_n,activation='relu')(fl))
        x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
        gl = MaxPooling1D(pool_size=1, strides=1)(s_dout)
        fl = Flatten()(gl)
        x = concatenate([(x), (fl)])
        outp = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h,in_d,in_wd,in_qty,in_c1,in_c2], outputs=outp)

        # Compile the model
        exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
        steps = int(len(X_train) / batch_size) * epochs
        # lr_init, lr_fin = 0.001, 0.0001
        lr_init, lr_fin = 0.00005, 0.000005
        # lr_decay = exp_decay(lr_init, lr_fin, steps)
        lr_decay = 0.0000001
        lr = lr_init
        optimizer_adam = optimizers.Adam(lr=lr, decay=lr_decay)
        # lr = DenseModel.OPT_LEARNING_RATE
        # lr_decay = DenseModel.OPT_DECAY
        # optimizer_adam = optimizers.adam(lr=lr, decay=lr_decay)
        print("Using learning init rate: ", lr, ", decay: ", lr_decay)
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer_adam,
            metrics=["accuracy"])

        model.summary()

        self._model = model

        # Establish class weights since this is a very unbalanced dataset
        class_weight = dict(zip([0, 1], compute_class_weight('balanced', [0, 1], Y_train)))
        print("Unbalanced data, actual class_weight: ", class_weight)
        # class_weight = {0:.01, 1:.99}
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

        # Train the model
        history = self._model.fit(
            X_train, Y_train, batch_size=batch_size, epochs=epochs,
            validation_split=0.2, class_weight=class_weight, shuffle=True,
            callbacks=[ES, TB, BL, RLRP, CHK], verbose=1)

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
