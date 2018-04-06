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
        gp = train_df[columns].groupby(by=fields)[key].count().reset_index().rename(index=str, columns={key: name})
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
            usecols=["ip","app","device","os", "channel", "click_time", "is_attributed"])
        # print("DATA: train_df:\n", train_df.head())

        print("Loading test set: ", testfile)
        test_df = pd.read_csv(
            testfile, dtype=dtypes,
            usecols=["ip","app","device","os", "channel", "click_time", "click_id"])
        # print("DATA: test_df:\n", test_df.head())
        len_train = len(train_df)
        train_df=train_df.append(test_df)
        del test_df; gc.collect()

        print("Identifying hour, day, day of week")
        train_df["hour"] = pd.to_datetime(train_df.click_time).dt.hour.astype("uint8")
        train_df["day"] = pd.to_datetime(train_df.click_time).dt.day.astype("uint8")
        train_df["wday"]  = pd.to_datetime(train_df.click_time).dt.dayofweek.astype("uint8")

        print("Grouping by ip, day, hour")
        gp = train_df[["ip","day","hour","channel"]].groupby(by=["ip","day","hour"])[["channel"]].count().reset_index().rename(index=str, columns={"channel": "qty"})
        train_df = train_df.merge(gp, on=["ip","day","hour"], how="left")
        del gp; gc.collect()

        print("Grouping by ip, app")
        gp = train_df[["ip","app", "channel"]].groupby(by=["ip", "app"])[["channel"]].count().reset_index().rename(index=str, columns={"channel": "ip_app_count"})
        train_df = train_df.merge(gp, on=["ip","app"], how="left")
        del gp; gc.collect()

        print("Grouping by ip, app, os")
        gp = train_df[["ip","app", "os", "channel"]].groupby(by=["ip", "app", "os"])[["channel"]].count().reset_index().rename(index=str, columns={"channel": "ip_app_os_count"})
        train_df = train_df.merge(gp, on=["ip","app", "os"], how="left")
        del gp; gc.collect()

        train_df["qty"] = train_df["qty"].astype("uint16")
        train_df["ip_app_count"] = train_df["ip_app_count"].astype("uint16")
        train_df["ip_app_os_count"] = train_df["ip_app_os_count"].astype("uint16")

        print("Encoding labels")
        from sklearn.preprocessing import LabelEncoder
        train_df[["app","device","os", "channel", "hour", "day", "wday"]].apply(LabelEncoder().fit_transform)

        # train_df.drop(["click_id", "click_time", "ip"], 1, inplace=True)

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
