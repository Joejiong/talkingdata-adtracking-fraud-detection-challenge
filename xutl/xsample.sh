#!/bin/bash

#
# Purpose:  Sampling script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [infile] is the CSV file containing training data (required)
#    [pcts] is the comma delimited list of percentage to be sampled (required)
#    [outfile] is the fully qualified filename for the retrieved data (required)
#    [seed] is the random seed (optional, default: 0)
#
# Usage:
#
#   train.sh [infile] [pcts] [seed]
#
#   example: ./sample.sh ../data/transform-train.h5 '0.001,0.005,0.010,0.050,0.100,0.250,0.500'
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    sample.sh [infile] [pcts] [seed]"
    echo " "
    echo "    where [infile] is the HDF5 file containing input data (required)"
    echo "          [pcts] is the comma delimited list of percentage to be sampled (required)"
    echo "          [seed] is the random seed (optional, default: 0)"
    echo " "
    echo "    example 1:  ./sample.sh ../data/transform-train.h5 '0.001,0.005,0.010,0.050,0.100,0.250,0.500'"
    echo " "
}

xINFILE="$1"
if [ -z "$1" ]; then
  showHelp "[infile] parameter is missing"
  exit
fi
xINFILE=$(realpath $xINFILE)

xPCTS="$2"
if [ -z "$2" ]; then
  showHelp "[pct] parameter is missing"
  exit
fi

xSEED="$3"
if [ -z "$3" ]; then
  xSEED=0
fi

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/xsrc)

echo " "
echo "---- Sample Parameters ----"
echo "CSV File:       $xINFILE"
echo "Pcts sampled:   $xPCTS"
echo "Seed:           $xSEED"
echo "Root Directory: $xROOTDIR"
echo "Source Dir:     $xSRCDIR"
echo " "

echo "Start: "; date
time python3 $xSRCDIR/sample.py \
              --infile $xINFILE \
              --pcts $xPCTS \
              --seed $xSEED
echo "End: "; date
