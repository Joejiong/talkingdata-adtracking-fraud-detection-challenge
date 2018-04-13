#!/bin/bash

#
# Purpose:  Data transformation script (output format: hdf5)
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [trainfile] is the CSV file containing training data (required)
#    [testfile] is the CSV file containing test data (required)
#    [seed] is the random seed (optional, default: 0)
#
# Usage:
#
#   transform.sh [trainfile] [testfile]
#
#   example: transform.sh ../data/train-sample.csv ../data/test-sample.csv
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    transform.sh [trainfile] [testfile] [seed]"
    echo " "
    echo "    where [trainfile] is the CSV file containing training data (required)"
    echo "          [testfile] is the CSV file containing test data (required)"
    echo " "
    echo "    NOTE: training and test output files will be in hdf5 format and have same name but with prefix of 'transform-' "
    echo " "
    echo "    example 1:  ./transform.sh ../data/train-sample.csv ../data/test-sample.csv"
    echo " "
    echo "          output: ../data/transform-train-sample.h5 and ../data/transform-test-sample.h5"
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

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/xsrc)

echo " "
echo "---- Tranform Parameters ----"
echo "Training File:   $xTRAINFILE"
echo "Test File:       $xTESTFILE"
echo "Root Directory:  $xROOTDIR"
echo "Source Dir:      $xSRCDIR"
echo " "

echo "Clearing old TF log files"
rm ./logs/events.out.tfevents*

echo "Start: "; date
time python3 $xSRCDIR/transform.py \
              --trainfile $xTRAINFILE \
              --testfile $xTESTFILE
echo "End: "; date
