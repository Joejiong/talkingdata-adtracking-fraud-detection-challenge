#!/bin/bash

#
# Purpose:  Sampling script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [csvfile] is the CSV file containing training data (required)
#    [pct] is the percentage of data to be retrieved (required)
#    [outfile] is the fully qualified filename for the retrieved data (required)
#    [seed] is the random seed (optional, default: 0)
#
# Usage:
#
#   train.sh [csvfile] [model-dir] [log-dir] [epochs] [batch]
#
#   example: train.sh ../data/train_sample.csv ../models ../logs 100 1000
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    sample.sh [csvfile] [model-dir] [log-dir] [epochs] [batch] [seed]"
    echo " "
    echo "    where [csvfile] is the CSV file containing training data (required)"
    echo "          [pct] is the percentage of data to be retrieved (required)"
    echo "          [outfile] is the fully qualified filename for the retrieved data (required)"
    echo "          [seed] is the random seed (optional, default: 0)"
    echo " "
    echo "    example 1:  ./sample.sh ../data/train.csv 0.10 ../data/sampled_data.csv"
    echo " "
}

xCSVFILE="$1"
if [ -z "$1" ]; then
  showHelp "[csvfile] parameter is missing"
  exit
fi
xCSVFILE=$(realpath $xCSVFILE)

xPCT="$2"
if [ -z "$2" ]; then
  showHelp "[pct] parameter is missing"
  exit
fi

xOUTFILE="$3"
if [ -z "$3" ]; then
  showHelp "[outfile] parameter is missing"
  exit
fi
xOUTFILE=$(realpath $xOUTFILE)

xSEED="$4"
if [ -z "$4" ]; then
  xSEED=0
fi

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/src)

echo " "
echo "---- Sampling Parameters ----"
echo "CSV File:       $xCSVFILE"
echo "Pct sampled:    $xPCT"
echo "Output file:    $xOUTFILE"
echo "Seed:           $xSEED"
echo "Root Directory: $xROOTDIR"
echo "Source Dir:     $xSRCDIR"
echo " "

# rm train*.log
echo "Start: "; date
time python3 $xSRCDIR/sample.py \
              --csvfile $xCSVFILE \
              --pct $xPCT \
              --outfile $xOUTFILE \
              --seed $xSEED
echo "End: "; date
