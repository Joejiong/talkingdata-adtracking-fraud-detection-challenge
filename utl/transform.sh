#!/bin/bash

#
# Purpose:  Transform input to processed data script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [trainfile] is the CSV file containing training data (required)
#    [outtrainfile] is the fully qualified filename for the retrieved data (required)
#    [ip-buckets] is the number of IP buckets (optional, default: 15)
#
# Usage:
#
#   transform.sh [trainfile] [testfile] [outtrainfile] [outtestfile] [ip-buckets] [appbuckets] [osbuckets] [channelbuckets] [devicebuckets]
#
#   example: transform.sh ../data/train.csv ../data/test.csv ../data/tranform-train.csv ../data/tranform-test.csv 5 41 39 17 5 False
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    transform.sh [trainfile] [testfile] [outtrainfile] [outtestfile] [ip-buckets] [appbuckets] [osbuckets] [channelbuckets] [devicebuckets]"
    echo " "
    echo "    where [trainfile] is the CSV file containing training data (required)"
    echo "          [testfile] is the CSV file containing test data (required)"
    echo "          [outtrainfile] is the fully qualified filename for the retrieved data (required)"
    echo "          [ip-buckets] is the number of IP buckets (optional, default: 15)"
    echo "          [app-buckets] is the number of IP buckets (optional, default: 15)"
    echo "          [os-buckets] is the number of IP buckets (optional, default: 15)"
    echo "          [channel-buckets] is the number of IP buckets (optional, default: 15)"
    echo "          [device-buckets] is the number of IP buckets (optional, default: 15)"
    echo "          [chunk] indicates that file is to be chucked into smaller files (default: False)"
    echo " "
    echo "    example 1:  ./transform.sh ../data/train.csv ../data/test.csv ../data/tranform-train.csv ../data/tranform-test.csv 5 41 39 17 5 False"
    echo " "
}

xTRAINFILE="$1"
if [ -z "$1" ]; then
  showHelp "[trainfile] parameter is missing"
  exit
fi
xTRAINFILE=$(realpath $xTRAINFILE)

xTESTFILE="$2"
if [ -z "$2" ]; then
  showHelp "[testfile] parameter is missing"
  exit
fi
xTESTFILE=$(realpath $xTESTFILE)

xOUTTRAINFILE="$3"
if [ -z "$3" ]; then
  showHelp "[outtrainfile] parameter is missing"
  exit
fi
xOUTTRAINFILE=$(realpath $xOUTTRAINFILE)

xOUTTESTFILE="$4"
if [ -z "$4" ]; then
  showHelp "[outtestfile] parameter is missing"
  exit
fi
xOUTTESTFILE=$(realpath $xOUTTESTFILE)

xIPBUCKETS="$5"
if [ -z "$5" ]; then
  showHelp "[ip-buckets] parameter is missing"
  exit
fi

xAPPBUCKETS="$6"
if [ -z "$6" ]; then
  showHelp "[app-buckets] parameter is missing"
  exit
fi

xOSBUCKETS="$7"
if [ -z "$7" ]; then
  showHelp "[os-buckets] parameter is missing"
  exit
fi

xCHANNELBUCKETS="$8"
if [ -z "$8" ]; then
  showHelp "[channel-buckets] parameter is missing"
  exit
fi

xDEVICEBUCKETS="$9"
if [ -z "$9" ]; then
  showHelp "[device-buckets] parameter is missing"
  exit
fi

xCHUNK="${10}"
if [ -z "${10}" ]; then
  xCHUNK=False
fi

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/src)

echo " "
echo "---- Transform Parameters ----"
echo "Training File:       $xTRAINFILE"
echo "Test File:           $xTESTFILE"
echo "Output train file:   $xOUTTRAINFILE"
echo "Output test file:    $xOUTTESTFILE"
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
              --trainfile $xTRAINFILE \
              --testfile $xTESTFILE \
              --outtrainfile $xOUTTRAINFILE \
              --outtestfile $xOUTTESTFILE \
              --ipbuckets $xIPBUCKETS \
              --appbuckets $xAPPBUCKETS \
              --osbuckets $xOSBUCKETS \
              --channelbuckets $xCHANNELBUCKETS \
              --devicebuckets $xDEVICEBUCKETS \
              --chunk $xCHUNK
echo "End: "; date
