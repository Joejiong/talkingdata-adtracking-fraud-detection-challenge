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


    def calc_confidences(self, df):
        ATTRIBUTION_CATEGORIES = [
            # V1 Features #
            ###############
            ['ip'], ['app'], ['device'], ['os'], ['channel'],

            # V2 Features #
            ###############
            ['app', 'channel'],
            ['app', 'os'],
            ['app', 'device'],
        ]

        # Find frequency of is_attributed for each unique value in column
        freqs = {}
        new_columns = []
        for cols in ATTRIBUTION_CATEGORIES:

            # New feature name
            new_feature = '_'.join(cols)+'_confRate'

            # Perform the groupby
            group_object = df.groupby(cols)

            # Group sizes
            group_sizes = group_object.size()
            log_group = 100000 # 1000 views -> 60% confidence, 100 views -> 40% confidence
            print("Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
                cols, new_feature,
                group_sizes.max(),
                np.round(group_sizes.mean(), 2),
                np.round(group_sizes.median(), 2),
                group_sizes.min()
            ))

            # Aggregation function: Calculate the attributed rate and scale by confidence
            def rate_calculation(x):
                rate = x.sum() / float(x.count())
                conf = np.min([1, np.log(x.count()) / log_group])
                return rate * conf

            # Perform the merge
            df = df.merge(
                group_object['is_attributed']. \
                    apply(rate_calculation). \
                    reset_index(). \
                    rename(
                        index=str,
                        columns={'is_attributed': new_feature}
                    )[cols + [new_feature]],
                on=cols, how='left'
            )
            del group_object; gc.collect()

            # Remove NaN data in the dataframe
            df = df.fillna(0)

            # Scale the new feature to 0-1
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            print("Scaling feature: ", new_feature)
            df[[new_feature]] = scaler.fit_transform(df[[new_feature]].as_matrix())

            new_columns.append(new_feature)

        return df, new_columns

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
        # print(train_df.head())

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

        train_df, new_columns = self.calc_confidences(train_df)
        print("New columns added: ", new_columns)
        print(train_df.head())

        ########################## NEW (end)

        print("Encoding labels")
        from sklearn.preprocessing import LabelEncoder

        # train_df[[
        #     "app", "device", "os", "channel", "hour", "qhour", "dqhour", "day", "wday"
        #     ]] = train_df[[
        #     "app", "device", "os", "channel", "hour", "qhour", "dqhour", "day", "wday"
        #     ]].apply(LabelEncoder().fit_transform)
        # train_df[[
        #     "app", "device", "os", "channel", "hour", "qhour", "dqhour", "day", "wday"
        #     ]].apply(LabelEncoder().fit_transform)

        for colname in ["app", "device", "os", "channel", "hour", "qhour", "dqhour", "day", "wday"]:
            print("Encoding column: ", colname)
            train_df[colname] = LabelEncoder().fit_transform(train_df[colname])

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
