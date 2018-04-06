#!/bin/bash

#
# Purpose:  Feature description and importance analysis script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [infile] is the CSV file containing training data (required)
#
# Usage:
#
#   feature.sh [infile]
#
#   example: feature.sh ../data/train.csv
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    feature.sh [infile]"
    echo " "
    echo "    where [infile] is the CSV file containing training data (required)"
    echo " "
    echo "    example 1:  feature.sh ../data/train.csv"
    echo " "
}

xINFILE="$1"
if [ -z "$1" ]; then
  showHelp "[infile] parameter is missing"
  exit
fi
xINFILE=$(realpath $xINFILE)

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/xsrc)

echo " "
echo "---- Feature Parameters ----"
echo "Input (CSV) File: $xINFILE"
echo "Root Directory:   $xROOTDIR"
echo "Source Dir:       $xSRCDIR"
echo " "

echo "Start: "; date
time python3 $xSRCDIR/feature.py \
              --infile $xINFILE
echo "End: "; date
