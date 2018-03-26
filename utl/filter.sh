#!/bin/bash

#
# Purpose:  Filter large training set script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [csvfile] is the CSV file containing test/submission data (required)
#    [feature] is the fully qualified filename to store the submission data (required)
#    [value] is the fully qualified filename for the model HDF5 file (required)
#    [outputfile] is the CSV file containing test/submission data (required)
#
# Usage:
#
#   filter.sh [csvfile] [feature] [value] [outputfile]
#
#   example: ./filter.sh ../data/train.csv is_attributed 1 ../data/train-xones.csv
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    filter.sh [csvfile] [feature] [value] [outputfile]"
    echo " "
    echo "    where [csvfile] is the CSV file containing test/submission data (required)"
    echo "          [feature] is the CSV file containing test/submission data (required)"
    echo "          [value] is the feature name for which to filter (required)"
    echo "          [outputfile] is the CSV file containing test/submission data (required)"
    echo " "
    echo "    example 1:  ./filter.sh ../data/train.csv is_attributed 1 ../data/train-xones.csv"
    echo " "
}

xCSVFILE="$1"
if [ -z "$1" ]; then
  showHelp "[csvfile] parameter is missing"
  exit
fi
xCSVFILE=$(realpath $xCSVFILE)

xFEATURE="$2"
if [ -z "$2" ]; then
  showHelp "[feature] parameter is missing"
  exit
fi

xVALUE="$3"
if [ -z "$3" ]; then
  showHelp "[value] parameter is missing"
  exit
fi

xOUTPUTFILE="$4"
if [ -z "$4" ]; then
  showHelp "[outputfile] parameter is missing"
  exit
fi
xOUTPUTFILE=$(realpath $xOUTPUTFILE)

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/src)

echo " "
echo "---- Filter Parameters ----"
echo "CSV File:       $xCSVFILE"
echo "Feature:        $xFEATURE"
echo "Value:          $xVALUE"
echo "Output File:    $xOUTPUTFILE"
echo "Root Directory: $xROOTDIR"
echo "Source Dir:     $xSRCDIR"
echo " "

echo "Removing output file: " $xOUTPUTFILE
rm $xOUTPUTFILE

echo "Start: "; date
time python3 $xSRCDIR/filter.py \
              --csvfile $xCSVFILE \
              --feature $xFEATURE \
              --value $xVALUE \
              --output $xOUTPUTFILE
echo "End: "; date
