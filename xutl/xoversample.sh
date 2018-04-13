#!/bin/bash

#
# Purpose:  Adversarial Validation script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [trainfile] is the CSV file containing training data (required)
#    [sampler] is the type of over-sampler to use (RANDOM|ADASYN|SMOTE, required)
#
# Usage:
#
#   xoversample.sh [trainfile] [sampler]
#
#   example: xoversample.sh transform-train-sample.h5 SMOTE
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    xoversample.sh [trainfile] [sampler]"
    echo " "
    echo "    where [trainfile] is the HDF5 file containing training data (required)"
    echo "          [sampler] is the type of over-sampler to use (RANDOM|ADASYN|SMOTE, required)"
    echo " "
    echo "    example 1:  ./xoversample.sh ../data/transform-train-sample.h5 SMOTE"
    echo " "
}

xTRAINFILE="$1"
if [ -z "$1" ]; then
  showHelp "[trainfile] parameter is missing"
  exit
fi
xTRAINFILE=$(realpath $xTRAINFILE)

xSAMPLER="$2"
if [ -z "$2" ]; then
  showHelp "[sampler] parameter is missing"
  exit
fi

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/xsrc)

echo " "
echo "---- Sampler Parameters ----"
echo "Training File:   $xTRAINFILE"
echo "Sampler:         $xSAMPLER"
echo "Root Directory:  $xROOTDIR"
echo "Source Dir:      $xSRCDIR"
echo " "

echo "Start: "; date
time python3 $xSRCDIR/oversample.py \
              --trainfile $xTRAINFILE \
              --sampler $xSAMPLER
echo "End: "; date
