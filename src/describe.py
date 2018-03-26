#####
# Imports
#####
import numpy as np
import pandas as pd
import argparse

#####
# Load the CSV training file
#####
def load(csv_file):

    #Import data
    df = pd.read_csv(csv_file, header=0)

    return df


#####
# Describe the data
#####
def describe(df):

    if "attributed_time" in df.columns:
        df[df["is_attributed"]==1].ip.describe()

        proportion = df[["ip", "is_attributed"]].groupby("ip", as_index=False).mean().sort_values("is_attributed", ascending=False)
        counts = df[["ip", "is_attributed"]].groupby("ip", as_index=False).count().sort_values("is_attributed", ascending=False)
        merge = counts.merge(proportion, on="ip", how="left")
        merge.columns = ["ip", "click_count", "prop_downloaded"]
        print("Conversion Rates over Counts of Most Popular IPs")
        print(merge[:20])
        print("\n")

        proportion = df[["app", "is_attributed"]].groupby("app", as_index=False).mean().sort_values("is_attributed", ascending=False)
        counts = df[["app", "is_attributed"]].groupby("app", as_index=False).count().sort_values("is_attributed", ascending=False)
        merge = counts.merge(proportion, on="app", how="left")
        merge.columns = ["app", "click_count", "prop_downloaded"]
        print("Conversion Rates over Counts of Most Popular Apps")
        print(merge[:20])
        print("\n")

        proportion = df[["os", "is_attributed"]].groupby("os", as_index=False).mean().sort_values("is_attributed", ascending=False)
        counts = df[["os", "is_attributed"]].groupby("os", as_index=False).count().sort_values("is_attributed", ascending=False)
        merge = counts.merge(proportion, on="os", how="left")
        merge.columns = ["os", "click_count", "prop_downloaded"]
        print("Conversion Rates over Counts of Most Popular Operating Systems")
        print(merge[:20])
        print("\n")

        proportion = df[["device", "is_attributed"]].groupby("device", as_index=False).mean().sort_values("is_attributed", ascending=False)
        counts = df[["device", "is_attributed"]].groupby("device", as_index=False).count().sort_values("is_attributed", ascending=False)
        merge = counts.merge(proportion, on="device", how="left")
        merge.columns = ["device", "click_count", "prop_downloaded"]
        print("Count of clicks and proportion of downloads by device:")
        print(merge[:20])
        print("\n")

        proportion = df[["channel", "is_attributed"]].groupby("channel", as_index=False).mean().sort_values("is_attributed", ascending=False)
        counts = df[["channel", "is_attributed"]].groupby("channel", as_index=False).count().sort_values("is_attributed", ascending=False)
        merge = counts.merge(proportion, on="channel", how="left")
        merge.columns = ["channel", "click_count", "prop_downloaded"]
        print("Conversion Rates over Counts of Most Popular Channels")
        print(merge[:20])
        print("\n")

    print(df.head())

    print(df.describe())

    print("\nCharacteristics:")
    print("Column: app")
    print("  unique:     ", len(df["app"].unique()))
    print("  maximum:    ", df["app"].max())
    print("  minimum:    ", df["app"].min())

    print("Column: os")
    print("  unique:     ", len(df["os"].unique()))
    print("  maximum:    ", df["os"].max())
    print("  minimum:    ", df["os"].min())

    print("Column: channel")
    print("  unique:     ", len(df["channel"].unique()))
    print("  maximum:    ", df["channel"].max())
    print("  minimum:    ", df["channel"].min())

    print("Column: ip")
    print("  unique:     ", len(df["ip"].unique()))
    print("  maximum:    ", df["ip"].max())
    print("  minimum:    ", df["ip"].min())

    print("Column: device")
    print("  unique:     ", len(df["device"].unique()))
    print("  maximum:    ", df["device"].max())
    print("  minimum:    ", df["device"].min())


#####
# Execute the training process
#####
def execute(csvfile):

    print("Using csvfile: ", csvfile)

    print("Loading data, csvfile: ", csvfile)
    df = load(csvfile)

    print("Describing data")
    describe(df)

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

    # Execute the command
    execute(args.csvfile)


main()
