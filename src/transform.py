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

    # os_qty: REMOVE
    # gp = df[["os","click_day","click_hour","channel"]].groupby(by=[
    #     "os","click_day","click_hour"])[["channel"]
    #     ].count().reset_index().rename(index=str, columns={"channel": "os_qty"})
    # df = df.merge(gp, on=["os","click_day","click_hour"], how="left")

    # app_hour: REMOVE
    # gp = df[["app","click_hour","channel"]].groupby(by=[
    #     "app","click_hour"])[["channel"]
    #     ].count().reset_index().rename(index=str, columns={"channel": "app_hour"})
    # df = df.merge(gp, on=["app","click_hour"], how="left")

    # app_day: REMOVE
    # gp = df[["app","click_day","channel"]].groupby(by=[
    #     "app","click_day"])[["channel"]
    #     ].count().reset_index().rename(index=str, columns={"channel": "app_day"})
    # df = df.merge(gp, on=["app","click_day"], how="left")

    # app_channel: REMOVE
    # gp = df[["app","channel"]].groupby(by=["app"])[["channel"]
    #     ].count().reset_index().rename(index=str, columns={"channel": "app_channel"})
    # df = df.merge(gp, on=["app"], how="left")

    # gp = df[["ip","device"]].groupby(by=["ip"])[["device"]
    #     ].count().reset_index().rename(index=str, columns={"device": "ip_device"})
    # df = df.merge(gp, on=["ip"], how="left")

    # gp = df[["os","channel"]].groupby(by=["os"])[["channel"]
    #     ].count().reset_index().rename(index=str, columns={"channel": "os_channel"})
    # df = df.merge(gp, on=["os"], how="left")

    ##### This did NOT help!!  (reduced score)
    # ip_count = df.groupby(["ip"])["channel"].count().reset_index()
    # ip_count.columns = ["ip", "clicks_by_ip"]
    # df = pd.merge(df, ip_count, on="ip", how="left", sort=False)
    # df["clicks_by_ip"] = df["clicks_by_ip"].astype("uint16")
    # print(df.head())
    # return

    # Not useful!
    # df["app_channel"] = df["app"].astype("str")+"_"+df["channel"].astype("str")
    # df["os_channel"] = df["os"].astype("str")+"_"+df["channel"].astype("str")

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
        xdf = df.iloc[start:end,0:c]
        # print(xdf.shape)
        # print(xdf.head(n=2))
        # print("outfile: ", outfile)
        filename = outfile.split(".csv")[0]
        # print("filename: ", filename)
        chunkname = "{}-{:010d}-{:010d}.csv".format(filename, start, end)
        print("Saving chunk file: ", chunkname)

        header = False
        if i==0:
            header = True
        xdf.to_csv(chunkname, encoding="utf-8", index=False, chunksize=10000, header=header)


    # start = end+1
    # end = r
    # print("cleanup...start: ", start, ", end: ", end, ", c: ", c)
    # xdf = df.iloc[start:end,0:c]
    # print(xdf.shape)


#####
# Execute the training process
#####
def execute(csvfile, outfile, ipbuckets, appbuckets, osbuckets, channelbuckets, devicebuckets, ischunked):

    print("Using csvfile: ", csvfile)
    print("Using outfile: ", outfile)
    print("Using ipbuckets: ", ipbuckets)
    print("Using appbuckets: ", appbuckets)
    print("Using osbuckets: ", osbuckets)
    print("Using channelbuckets: ", channelbuckets)
    print("Using devicebuckets: ", devicebuckets)
    print("Using ischunked: ", ischunked)

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Loading data, csvfile: ", csvfile)
    df = load(csvfile)
    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Input: ")
    print(df.head(n=10))

    # print("Transforming data, istest: ", istest)
    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Transforming data")
    # df = transform(df, ipbuckets, istest)
    df = transform(df, ipbuckets, appbuckets, osbuckets, channelbuckets, devicebuckets)
    print(df.head(n=10))
    print(list(df.columns.values))

    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Saving data, outfile: ", outfile)
    save(df, outfile, ischunked)
    print("Timestamp: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

#####
# Parse the command line
#####
def cli():
    """
    Command line interface
    """

    # Parse the command line and return the args/parameters
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument("-c", "--csvfile", help="is the CSV file containing training data (required)")
    parser.add_argument("-o", "--outfile", help="is the fully qualified filename for the retrieved data (required)")
    parser.add_argument("-I", "--ipbuckets", help="is the number of IP buckets (optional, default: 15)")
    parser.add_argument("-A", "--appbuckets", help="is the number of APP buckets (optional, default: 15)")
    parser.add_argument("-O", "--osbuckets", help="is the number of OS buckets (optional, default: 15)")
    parser.add_argument("-C", "--channelbuckets", help="is the number of CHANNEL buckets (optional, default: 15)")
    parser.add_argument("-D", "--devicebuckets", help="is the number of DEVICE buckets (optional, default: 15)")
    parser.add_argument("-x", "--chunk", help="indicates that file is to be chucked into smaller files (default: False)")

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
    if not args.csvfile:
        raise Exception("Missing argument: --csvfile")
    if not args.outfile:
        raise Exception("Missing argument: --outfile")

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
        args.csvfile,
        args.outfile,
        int(args.ipbuckets),
        int(args.appbuckets),
        int(args.osbuckets),
        int(args.channelbuckets),
        int(args.devicebuckets),
        ischunked)


main()
