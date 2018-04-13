import numpy as np

import keras
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#####
# ROC callback used to show AUC-ROC during model fit (training)
#####
class ROCCallback(Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        # Need to convert data here (not implemented... will break)
        y_pred = self.model.predict(self.validation_data[0])
        score = roc_auc_score(self.validation_data[1], y_pred)
        self.aucs.append(score)
        print("ROC-AUC Score: ", score)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
