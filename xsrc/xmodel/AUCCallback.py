import numpy as np

import keras
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

#####
# ROC callback used to show AUC-ROC during model fit (training)
#####
class AUCCallback(Callback):
    def __init__(self, mymodel, X_validation, Y_validation, convert=False):
        # Initialize
        print("AUCCallback initializing")
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self._convert = convert
        self.mymodel = mymodel

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        X_test = self.X_validation
        if self._convert:
            X_test = self.mymodel.convert(self.X_validation)
            
        y_pred = self.model.predict(X_test)
        probabilities = y_pred[:, 0]
        score = roc_auc_score(self.Y_validation, probabilities)
        print("ROC-AUC Score: ", score)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
