#####
# Imports
#####
import numpy as np
import pandas as pd
import argparse
from time import gmtime, strftime

#####
# Load the CSV training file
#####
def load(csv_file):
    df = pd.read_csv(csv_file, header=0)
    return df


#####
# Describe the data
#####
def transform(df, ipbuckets, appbuckets, osbuckets, channelbuckets, devicebuckets):

    # df["click_time"] = pd.to_datetime(df["click_time"], errors='coerce')
    # print(df[df.click_time.isnull()])
    # return

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Processing click time column...")
    df["click_time"] = pd.to_datetime(df["click_time"], errors='coerce')
    df["click_time"] = df["click_time"].dt.round("H")
    df["click_hour"] = df["click_time"].dt.hour.astype("uint8")
    df["click_day"] = pd.to_datetime(df.click_time).dt.day.astype("uint8")
    df["click_dow"] = df["click_time"].dt.dayofweek
    # df["click_doy"] = df["click_time"].dt.dayofyear
    # df["click_dom"] = df["click_time"].dt.daysinmonth - df["click_time"].dt.day


    # Count ip/click_day/click_hour/channel
    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Group and merge...")

    gp = df[["ip","click_day","click_hour","channel"]].groupby(by=[
        "ip","click_day","click_hour"])[["channel"]
        ].count().reset_index().rename(index=str, columns={"channel": "ip_qty"})
    df = df.merge(gp, on=["ip","click_day","click_hour"], how="left")

    gp = df[["ip","click_day","channel"]].groupby(by=[
        "ip","click_day"])[["channel"]
        ].count().reset_index().rename(index=str, columns={"channel": "ip_day"})
    df = df.merge(gp, on=["ip","click_day"], how="left")

    gp = df[["ip","click_hour","channel"]].groupby(by=[
        "ip","click_hour"])[["channel"]
        ].count().reset_index().rename(index=str, columns={"channel": "ip_hour"})
    df = df.merge(gp, on=["ip","click_hour"], how="left")

    gp = df[["app","click_day","click_hour","channel"]].groupby(by=[
        "app","click_day","click_hour"])[["channel"]
        ].count().reset_index().rename(index=str, columns={"channel": "app_qty"})
    df = df.merge(gp, on=["app","click_day","click_hour"], how="left")

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Bucketing...")
    df["ip_bucket"] = pd.cut(df.ip,ipbuckets, labels=range(ipbuckets))
    df["app_bucket"] = pd.cut(df.app,appbuckets, labels=range(appbuckets))
    df["os_bucket"] = pd.cut(df.os,osbuckets, labels=range(osbuckets))
    df["channel_bucket"] = pd.cut(df.channel,channelbuckets, labels=range(channelbuckets))
    df["device_bucket"] = pd.cut(df.device,devicebuckets, labels=range(devicebuckets))

    # print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Creating categorical columns...")
    categorical_columns = ["ip_bucket", "app_bucket", "os_bucket", "channel_bucket", "device_bucket"]
    for item in categorical_columns:
        print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        print("Processing categorical column: ", item)
        df = pd.get_dummies(df, columns=[item])

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Dropping redundant columns...")
    df = df.drop(["ip", "app", "os", "channel", "device", "click_time"], axis = 1)

    if "attributed_time" in df.columns:
        df = df.drop(["attributed_time"], axis = 1)

    # print(df.head())
    # return

    df = df.fillna(0)

    return df


#####
# Save the data
#####
def save(df, outfile, ischunked):
    if not ischunked:
        df.to_csv(outfile, encoding="utf-8", index=False, chunksize=10000)
        return

    chunksize = 100000
    print(df.shape)
    r,c = df.shape
    numchunks = int(r/chunksize)+1
    # print("numchunks: ", numchunks)
    print("Chunking files, chunksize: ", chunksize, ", columns: ", c)
    start = 0
    end = 0
    for i in range(numchunks):
        start = i * chunksize
        end = (i+1) * chunksize
        if end > r:
            end = r

        print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        print("Chunking rows: start: ", start, ", end: ", end)
        chunk = df.iloc[start:end,0:c]

        with_header = False
        if i==0:
            with_header = True

        with open(outfile, "a") as f:
            chunk.to_csv(f, header=with_header, index=False)

    # start = end+1
    # end = r
    # print("cleanup...start: ", start, ", end: ", end, ", c: ", c)
    # xdf = df.iloc[start:end,0:c]
    # print(xdf.shape)


#####
# Execute the training process
#####
def execute(trainfile, testfile, outtrainfile, outtestfile, ipbuckets, appbuckets, osbuckets, channelbuckets, devicebuckets, ischunked):

    print("Using trainfile: ", trainfile)
    print("Using testfile: ", testfile)
    print("Using outtrainfile: ", outtrainfile)
    print("Using outtestfile: ", outtestfile)
    print("Using ipbuckets: ", ipbuckets)
    print("Using appbuckets: ", appbuckets)
    print("Using osbuckets: ", osbuckets)
    print("Using channelbuckets: ", channelbuckets)
    print("Using devicebuckets: ", devicebuckets)
    print("Using ischunked: ", ischunked)

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Loading data, trainfile: ", trainfile)
    traindf = load(trainfile)
    print(traindf.head())

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Loading data, testfile: ", testfile)
    testdf = load(testfile)
    print(testdf.head())

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Merging train and test data")
    df, trainidx, testidx, is_attributeds, click_ids = merge(traindf, testdf)
    print("Merged dataframes, trainidx: ", trainidx, ", testidx: ", testidx)

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Transforming data")
    df = transform(df, ipbuckets, appbuckets, osbuckets, channelbuckets, devicebuckets)
    print(df.head(n=10))
    print(list(df.columns.values))

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Separating train and test data")
    traindf, testdf = separate(df, trainidx, testidx, is_attributeds, click_ids)
    print("Separated dataframes, traindf: ", traindf.shape, ", testdf: ", testdf.shape)

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Saving data, outtrainfile: ", outtrainfile)
    save(traindf, outtrainfile, ischunked)
    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Saving data, outtestfile: ", outtestfile)
    save(testdf, outtestfile, ischunked)
    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

#####
# Merge the train and test file and return respective indices
#####
def merge(traindf, testdf):

    # Cleanup the training file and retain the is_attributed flag
    is_attributeds = traindf["is_attributed"]
    traindf = traindf.drop(["attributed_time"], axis = 1)
    traindf = traindf.drop(["is_attributed"], axis = 1)

    # Cleanup the test file and retain the click_id flag
    click_ids = testdf["click_id"]
    testdf = testdf.drop(["click_id"], axis = 1)
    testdf = testdf.reset_index(drop=True)

    trainrows = traindf.shape[0]
    testrows = testdf.shape[0]
    trainidx = (0, trainrows)
    testidx = (trainrows, trainrows+testrows)

    df = traindf.append(testdf, ignore_index=True)

    return df, trainidx, testidx, is_attributeds, click_ids


#####
# Separate the train and test merged dataframe
#####
def separate(df, trainidx, testidx, is_attributeds, click_ids):
    trainstart, trainend = trainidx
    teststart, testend = testidx

    # Create the training dataframe and add in is_attributed column
    traindf = df[trainstart:trainend]
    # traindf["is_attributed"] = is_attributeds.values
    traindf.insert(loc=len(traindf.columns.values), column="is_attributed", value=is_attributeds.values)

    # Create test dataframe and add back in the click_id column
    testdf = df[teststart:testend]
    testdf.insert(loc=0, column="click_id", value=click_ids.values)
    # testdf["click_id"] = click_ids
    # cols = ["click_id", "ip", "app", "device", "os", "channel", "click_time"]
    # testdf = testdf[cols]

    return traindf, testdf


#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument("-t", "--trainfile", help="is the CSV file containing training data (required)")
    parser.add_argument("-T", "--testfile", help="is the CSV file containing test data (required)")
    parser.add_argument("-o", "--outtrainfile", help="is the transformed training file (required)")
    parser.add_argument("-O", "--outtestfile", help="is the transformed test file (required)")
    parser.add_argument("-i", "--ipbuckets", help="is the number of IP buckets (optional, default: 15)")
    parser.add_argument("-a", "--appbuckets", help="is the number of APP buckets (optional, default: 15)")
    parser.add_argument("-s", "--osbuckets", help="is the number of OS buckets (optional, default: 15)")
    parser.add_argument("-c", "--channelbuckets", help="is the number of CHANNEL buckets (optional, default: 15)")
    parser.add_argument("-d", "--devicebuckets", help="is the number of DEVICE buckets (optional, default: 15)")
    parser.add_argument("-C", "--chunk", help="indicates that file is to be chucked into smaller files (default: False)")

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
    args = cli()
    if not args.trainfile:
        raise Exception("Missing argument: --trainfile")
    if not args.testfile:
        raise Exception("Missing argument: --trainfile")
    if not args.outtrainfile:
        raise Exception("Missing argument: --outtrainfile")
    if not args.outtestfile:
        raise Exception("Missing argument: --outtestfile")

    if not args.ipbuckets:
        args.ipbuckets = 15
    if not args.appbuckets:
        args.appbuckets = 15
    if not args.osbuckets:
        args.osbuckets = 15
    if not args.channelbuckets:
        args.channelbuckets = 15
    if not args.devicebuckets:
        args.devicebuckets = 15

    ischunked = False
    if args.chunk:
        ischunked = args.chunk in ["true", "True", "TRUE", "1"]

    # Execute the command
    execute(
        args.trainfile,
        args.testfile,
        args.outtrainfile,
        args.outtestfile,
        int(args.ipbuckets),
        int(args.appbuckets),
        int(args.osbuckets),
        int(args.channelbuckets),
        int(args.devicebuckets),
        ischunked)


main()
