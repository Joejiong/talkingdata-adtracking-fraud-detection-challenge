#!/bin/bash

#
# Purpose:  Transform input to processed data script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [csvfile] is the CSV file containing training data (required)
#    [outfile] is the fully qualified filename for the retrieved data (required)
#    [ip-buckets] is the number of IP buckets (optional, default: 15)
#
# Usage:
#
#   transform.sh [csvfile] [outfile] [ip-buckets] [appbuckets] [osbuckets] [channelbuckets] [devicebuckets]
#
#   example: transform.sh ../data/train_sample.csv ../data/tranform.csv 15 15 15 15 15
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    transform.sh [csvfile] [outfile] [ip-buckets] [app-buckets] [os-buckets] [channel-buckets] [device-buckets] [chunk]"
    echo " "
    echo "    where [csvfile] is the CSV file containing training data (required)"
    echo "          [outfile] is the fully qualified filename for the retrieved data (required)"
    echo "          [ip-buckets] is the number of IP buckets (optional, default: 15)"
    echo "          [app-buckets] is the number of IP buckets (optional, default: 15)"
    echo "          [os-buckets] is the number of IP buckets (optional, default: 15)"
    echo "          [channel-buckets] is the number of IP buckets (optional, default: 15)"
    echo "          [device-buckets] is the number of IP buckets (optional, default: 15)"
    echo "          [chunk] indicates that file is to be chucked into smaller files (default: False)"
    echo " "
    echo "    example 1:  transform.sh ../data/train_sample.csv ../data/tranform.csv 15 False"
    echo " "
}

xCSVFILE="$1"
if [ -z "$1" ]; then
  showHelp "[csvfile] parameter is missing"
  exit
fi
xCSVFILE=$(realpath $xCSVFILE)

xOUTFILE="$2"
if [ -z "$2" ]; then
  showHelp "[outfile] parameter is missing"
  exit
fi
xOUTFILE=$(realpath $xOUTFILE)

xIPBUCKETS="$3"
if [ -z "$3" ]; then
  showHelp "[ip-buckets] parameter is missing"
  exit
fi

xAPPBUCKETS="$4"
if [ -z "$4" ]; then
  showHelp "[app-buckets] parameter is missing"
  exit
fi

xOSBUCKETS="$5"
if [ -z "$5" ]; then
  showHelp "[os-buckets] parameter is missing"
  exit
fi

xCHANNELBUCKETS="$6"
if [ -z "$6" ]; then
  showHelp "[channel-buckets] parameter is missing"
  exit
fi

xDEVICEBUCKETS="$7"
if [ -z "$7" ]; then
  showHelp "[device-buckets] parameter is missing"
  exit
fi

xCHUNK="$8"
if [ -z "$8" ]; then
  xCHUNK=False
fi

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/src)

echo " "
echo "---- Transform Parameters ----"
echo "CSV File:            $xCSVFILE"
echo "Output file:         $xOUTFILE"
echo "Num IP Buckets:      $xIPBUCKETS"
echo "Num APP Buckets:     $xAPPBUCKETS"
echo "Num OS Buckets:      $xOSBUCKETS"
echo "Num CHANNEL Buckets: $xCHANNELBUCKETS"
echo "Num DEVICE Buckets:  $xDEVICEBUCKETS"
echo "Chunk:               $xCHUNK"
echo "Root Directory:      $xROOTDIR"
echo "Source Dir:          $xSRCDIR"
echo " "

# rm train*.log
echo "Start: "; date
time python3 $xSRCDIR/transform.py \
              --csvfile $xCSVFILE \
              --outfile $xOUTFILE \
              --ipbuckets $xIPBUCKETS \
              --appbuckets $xAPPBUCKETS \
              --osbuckets $xOSBUCKETS \
              --channelbuckets $xCHANNELBUCKETS \
              --devicebuckets $xDEVICEBUCKETS \
              --chunk $xCHUNK
echo "End: "; date
