#!/bin/bash

#
# Purpose:  Sampling script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [infile] is the CSV file containing training data (required)
#    [pct] is the percentage of data to be retrieved (required)
#    [outfile] is the fully qualified filename for the retrieved data (required)
#    [seed] is the random seed (optional, default: 0)
#
# Usage:
#
#   train.sh [infile] [model-dir] [log-dir] [epochs] [batch]
#
#   example: train.sh ../data/train_sample.csv ../models ../logs 100 1000
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    sample.sh [infile] [model-dir] [log-dir] [epochs] [batch] [seed]"
    echo " "
    echo "    where [infile] is the HDF5 file containing input data (required)"
    echo "          [pct] is the percentage of data to be retrieved (required)"
    echo "          [outfile] is the fully qualified HDF5 filename for the retrieved data (required)"
    echo "          [seed] is the random seed (optional, default: 0)"
    echo " "
    echo "    example 1:  ./sample.sh ../data/transform-train.h5 0.10 ../data/transform-train-0.1.h5"
    echo " "
}

xINFILE="$1"
if [ -z "$1" ]; then
  showHelp "[infile] parameter is missing"
  exit
fi
xINFILE=$(realpath $xINFILE)

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

xSRCDIR=$(realpath $xROOTDIR/xsrc)

echo " "
echo "---- Sample Parameters ----"
echo "CSV File:       $xINFILE"
echo "Pct sampled:    $xPCT"
echo "Output file:    $xOUTFILE"
echo "Seed:           $xSEED"
echo "Root Directory: $xROOTDIR"
echo "Source Dir:     $xSRCDIR"
echo " "

echo "Removing (previous) output file: " $xOUTFILE
rm $xOUTFILE

# rm train*.log
echo "Start: "; date
time python3 $xSRCDIR/sample.py \
              --infile $xINFILE \
              --pct $xPCT \
              --outfile $xOUTFILE \
              --seed $xSEED
echo "End: "; date
