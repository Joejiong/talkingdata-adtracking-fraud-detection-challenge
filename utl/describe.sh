#!/bin/bash

#
# Purpose:  Dataset description script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [csvfile] is the CSV file containing training data (required)
#
# Usage:
#
#   train.sh [csvfile]
#
#   example: describe.sh ../data/train_sample.csv
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    describe.sh [csvfile]"
    echo " "
    echo "    where [csvfile] is the CSV file containing training data (required)"
    echo " "
    echo "    example 1:  ./describe.sh ../data/train_sample.csv"
    echo " "
}

xCSVFILE="$1"
if [ -z "$1" ]; then
  showHelp "[csvfile] parameter is missing"
  exit
fi
xCSVFILE=$(realpath $xCSVFILE)

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/src)

echo " "
echo "---- Sampling Parameters ----"
echo "CSV File:       $xCSVFILE"
echo "Root Directory: $xROOTDIR"
echo "Source Dir:     $xSRCDIR"
echo " "

# rm train*.log
echo "Start: "; date
time python3 $xSRCDIR/describe.py \
              --csvfile $xCSVFILE
echo "End: "; date
