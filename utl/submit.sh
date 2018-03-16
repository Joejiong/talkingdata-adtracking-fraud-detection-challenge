#!/bin/bash

#
# Purpose:  Submission/prediction script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [csvfile] is the CSV file containing test/submission data (required)
#    [submissionfile] is the fully qualified filename to store the submission data (required)
#    [modelfile] is the fully qualified filename for the model HDF5 file (required)
#
# Usage:
#
#   submit.sh [csvfile] [submissionfile] [modelfile]
#
#   example: submit.sh ../data/test.csv ../data/submission.csv ../models/model-final-auc-1.0.h5
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    submit.sh [csvfile] [submissionfile] [modelfile]"
    echo " "
    echo "    where [csvfile] is the CSV file containing test/submission data (required)"
    echo "          [submissionfile] is the fully qualified filename to store the submission data (required)"
    echo "          [modelfile] is the fully qualified filename for the model HDF5 file (required)"
    echo "          [epochs] is the number of epochs (optional, default: 100)"
    echo "          [batch] is the batch size (optional, default: 1000)"
    echo "          [seed] is the random seed (optional, default: 0)"
    echo " "
    echo "    example 1:  submit.sh ../data/test.csv ../data/submission.csv ../models/model-final-auc-1.0.h5"
    echo " "
}

xCSVFILE="$1"
if [ -z "$1" ]; then
  showHelp "[csvfile] parameter is missing"
  exit
fi
xCSVFILE=$(realpath $xCSVFILE)

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

xSRCDIR=$(realpath $xROOTDIR/src)

echo " "
echo "---- Training Parameters ----"
echo "CSV File:        $xCSVFILE"
echo "Submission file: $xSUBMISSIONFILE"
echo "Model file:      $xMODELFILE"
echo "Root Directory:  $xROOTDIR"
echo "Source Dir:      $xSRCDIR"
echo " "

echo "Start: "; date
time python3 $xSRCDIR/submit.py \
              --csvfile $xCSVFILE \
              --submissionfile $xSUBMISSIONFILE \
              --modelfile $xMODELFILE
echo "End: "; date
