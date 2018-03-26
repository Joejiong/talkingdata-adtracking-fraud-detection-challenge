#!/bin/bash

#
# Purpose:  Score a training validation dataset
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [csvfile] is the CSV file containing training validation data (required)
#    [modelfile] is the fully qualified filename for the model HDF5 file (required)
#    [threshold] is the threshold value (optional)
#
# Usage:
#
#   score.sh [csvfile] [modelfile]
#
#   example: score.sh ../data/test.csv ../models/model-final-auc-1.0.h5
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    score.sh [csvfile] [submissionfile] [modelfile]"
    echo " "
    echo "    where [csvfile] is the CSV file containing test/submission data (required)"
    echo "          [modelfile] is the fully qualified filename for the model HDF5 file (required)"
    echo "          [threshold] is the threshold value (optional)"
    echo " "
    echo "    example 1:  score.sh ../data/test.csv ../models/model-final-auc-1.0.h5"
    echo " "
}

xCSVFILE="$1"
if [ -z "$1" ]; then
  showHelp "[csvfile] parameter is missing"
  exit
fi
xCSVFILE=$(realpath $xCSVFILE)

xMODELFILE="$2"
if [ -z "$2" ]; then
  showHelp "[modelfile] parameter is missing"
  exit
fi
xMODELFILE=$(realpath $xMODELFILE)

xTHRESHOLD="$3"
if [ -z "$3" ]; then
  showHelp "[threshold] parameter is missing"
  exit
fi

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/src)

echo " "
echo "---- Submit Parameters ----"
echo "CSV File:        $xCSVFILE"
echo "Model file:      $xMODELFILE"
echo "Root Directory:  $xROOTDIR"
echo "Threshold:       $xTHRESHOLD"
echo "Source Dir:      $xSRCDIR"
echo " "

echo "Start: "; date
time python3 $xSRCDIR/score.py \
              --csvfile $xCSVFILE \
              --modelfile $xMODELFILE \
              --threshold $xTHRESHOLD
echo "End: "; date
