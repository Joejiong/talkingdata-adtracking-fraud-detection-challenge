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

        startrange = 1
        endrange = 1
        # endrange = 131886954
        print("Loading training set: ", trainfile, "skipping: ", startrange, "to", endrange)
        train_df = pd.read_csv(
            trainfile, dtype=dtypes,
            skiprows=range(startrange, endrange),
            usecols=["ip","app","device","os", "channel", "click_time", "is_attributed"])

        print("Loading test set: ", testfile)
        test_df = pd.read_csv(
            testfile, dtype=dtypes,
            usecols=["ip","app","device","os", "channel", "click_time", "click_id"])
        len_train = len(train_df)
        train_df=train_df.append(test_df)
        del test_df; gc.collect()

        print("Grouping by hour, day, day of week")
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

        print ("Cleaning up")
        X_test = train_df[len_train:]
        X_train = train_df[:len_train]
        Y_train = X_train["is_attributed"].values
        train_df.drop(["click_id", "click_time", "ip", "is_attributed"], 1, inplace=True)

        return X_train, X_test, Y_train


    def save(self, df, outfilename, filetype="h5"):
        # Save data in appropriate format
        if filetype == "h5":
            compressor = "blosc"
            df.to_hdf(outfilename, "table", mode="w", append=True, complevel=9, complib=compressor)
            return

        if filetype == "csv":
            df.to_csv(outfilename, header=True, index=False)
            return

    def load(self, infilename, filetype="h5"):
        if filetype == "h5":
            df = pd.read_hdf(infilename, mode="r")
            return df

        if filetype == "csv":
            df = pd.read_csv(infilename)
            return df
