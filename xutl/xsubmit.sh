#!/bin/bash

#
# Purpose:  Submission/prediction script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [infile] is the CSV file containing test/submission data (required)
#    [submissionfile] is the fully qualified filename to store the submission data (required)
#    [modelfile] is the fully qualified filename for the model HDF5 file (required)
#
# Usage:
#
#   submit.sh [infile] [submissionfile] [modelfile]
#
#   example: submit.sh ../data/test.csv ../data/submission.csv ../models/model-final-auc-1.0.h5
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    submit.sh [infile] [submissionfile] [modelfile]"
    echo " "
    echo "    where [infile] is the HDF5 file containing test/submission data (required)"
    echo "          [submissionfile] is the fully qualified CSV filename to store the submission data (required)"
    echo "          [modelfile] is the fully qualified filename for the model HDF5 file (required)"
    echo " "
    echo "    example 1:  submit.sh ../data/transform-test.csv ../data/submission.csv ../models/dense-model-final.h5"
    echo " "
}

xINFILE="$1"
if [ -z "$1" ]; then
  showHelp "[infile] parameter is missing"
  exit
fi
xINFILE=$(realpath $xINFILE)

xSUBMISSIONFILE="$2"
if [ -z "$2" ]; then
  showHelp "[submissionfile] parameter is missing"
  exit
fi
xSUBMISSIONFILE=$(realpath $xSUBMISSIONFILE)

xMODELFILE="$3"
if [ -z "$3" ]; then
  showHelp "[modelfile] parameter is missing"
  exit
fi
xMODELFILE=$(realpath $xMODELFILE)

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/xsrc)

echo " "
echo "---- Submit Parameters ----"
echo "Input (HDF5) File: $xINFILE"
echo "Submission file:   $xSUBMISSIONFILE"
echo "Model file:        $xMODELFILE"
echo "Root Directory:    $xROOTDIR"
echo "Source Dir:        $xSRCDIR"
echo " "

echo "Removing (previous) submission file: " $xSUBMISSIONFILE
rm $xSUBMISSIONFILE

echo "Start: "; date
time python3 $xSRCDIR/submit.py \
              --infile $xINFILE \
              --submissionfile $xSUBMISSIONFILE \
              --modelfile $xMODELFILE
echo "End: "; date
