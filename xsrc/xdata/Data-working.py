import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = "4"
import gc

class Data:

    #####
    # Constants
    #####

    def __init__(self):
        # Initialize
        print("Initializing")

    def group(self, df, name, columns, key, fields):

        print("Grouping: name: {}, columns: {}, key: {}, fields: {}".format(name, columns, key, fields))
        gp = df[columns].groupby(by=fields)[[key]].count().reset_index().rename(index=str, columns={key: name})
        df = df.merge(gp, on=fields, how="left")
        del gp; gc.collect()
        return df


    def transform(self, trainfile="train.csv", testfile="test.csv"):
        print("Using trainfile: ", trainfile)
        print("Using testfile:  ", testfile)

        # Transform data
        dtypes = {
            "ip"            : "uint32",
            "app"           : "uint16",
            "device"        : "uint16",
            "os"            : "uint16",
            "channel"       : "uint16",
            "is_attributed" : "uint8",
            "click_id"      : "uint32"
            }

        print("Loading training set: ", trainfile)
        train_df = pd.read_csv(
            trainfile, dtype=dtypes,
            usecols=["ip", "app", "device", "os", "channel", "click_time", "is_attributed"])
        # print("DATA: train_df:\n", train_df.head())

        print("Loading test set: ", testfile)
        test_df = pd.read_csv(
            testfile, dtype=dtypes,
            usecols=["ip","app","device","os", "channel", "click_time", "click_id"])
        # print("DATA: test_df:\n", test_df.head())
        len_train = len(train_df)
        train_df=train_df.append(test_df)
        del test_df; gc.collect()

        print("Identifying quarter hour, hour, day, day of week")
        train_df["hour"] = pd.to_datetime(train_df.click_time).dt.hour.astype("uint8")
        train_df["day"] = pd.to_datetime(train_df.click_time).dt.day.astype("uint8")
        train_df["wday"]  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype("uint8")

        series = pd.to_datetime(train_df.click_time).dt.minute.astype("uint8")
        train_df["qhour"] = pd.cut(series, [0, 15, 30, 45, 60], labels=[1, 2, 3, 4]).astype("uint16")
        del series; gc.collect()
        #
        series = train_df["hour"]*4 + train_df["qhour"]
        dqrange = np.linspace(0, 4*24, num=4*24+1, dtype="uint16")
        dqlabels = np.linspace(0, 4*24, num=4*24, dtype="uint16")
        train_df["dqhour"] = pd.cut(series, dqrange, labels=dqlabels).astype("uint16")
        del series; gc.collect()
        print(train_df.head())

        name = "qty"
        columns = ["ip", "day", "dqhour", "channel"]
        key = "channel"
        fields = ["ip", "day", "dqhour"]
        train_df = self.group(train_df, name, columns, key, fields)
        train_df[name] = train_df[name].astype("uint16")

        name = "ip_app_count"
        columns = ["ip", "app", "channel"]
        key = "channel"
        fields = ["ip", "app"]
        train_df = self.group(train_df, name, columns, key, fields)
        train_df[name] = train_df[name].astype("uint16")

        name = "ip_app_os_count"
        columns = ["ip", "app", "os", "channel"]
        key = "channel"
        fields = ["ip", "app", "os"]
        train_df = self.group(train_df, name, columns, key, fields)
        train_df[name] = train_df[name].astype("uint16")

        name = "new_column_1"
        columns = ["app", "wday", "dqhour", "channel"]
        key = "channel"
        fields = ["app", "wday", "dqhour"]
        train_df = self.group(train_df, name, columns, key, fields)
        train_df[name] = train_df[name].astype("uint16")

        name = "new_column_2"
        columns = ["os", "wday", "dqhour", "channel"]
        key = "channel"
        fields = ["os", "wday", "dqhour"]
        train_df = self.group(train_df, name, columns, key, fields)
        train_df[name] = train_df[name].astype("uint16")

        name = "new_column_3"
        columns = ["channel", "wday", "dqhour", "device"]
        key = "device"
        fields = ["channel", "wday", "dqhour"]
        train_df = self.group(train_df, name, columns, key, fields)
        train_df[name] = train_df[name].astype("uint16")

        ########################## NEW (start)

        # most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
        # least_freq_hours_in_test_data = [6, 11, 15]
        # train_df['in_test_hh'] = (3
        #     - 2*train_df['hour'].isin(  most_freq_hours_in_test_data )
        #     - 1*train_df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')

        ########################## NEW (end)

        print("Encoding labels")
        from sklearn.preprocessing import LabelEncoder
        train_df[["app", "device", "os", "channel", "hour", "qhour", "dqhour", "day", "wday"]].apply(LabelEncoder().fit_transform)

        print ("Cleaning up")
        X_test = train_df[len_train:]
        X_test.drop(["click_time", "ip", "is_attributed"], 1, inplace=True)
        X_train = train_df[:len_train]
        X_train.drop(["click_id", "click_time", "ip"], 1, inplace=True)


        return X_train, X_test


    def save(self, df, outfilename):
        # Save data in appropriate format
        compressor = "blosc"
        df.to_hdf(outfilename, "table", mode="w", append=True, complevel=9, complib=compressor)
        return


    def load(self, infilename):
        df = pd.read_hdf(infilename, "table", mode="r")
        return df
