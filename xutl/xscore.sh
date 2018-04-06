#!/bin/bash

#
# Purpose:  Training script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [scorefile] is the HDF5 file containing scoreing data (required)
#    [modelfile] is the fully qualified directory to store checkpoint models (required)
#    [fraction] is the fraction of data from the infile to be fractiond (required)
#    [seed] is the random number seed (optional)
#
# Usage:
#
#   score.sh [infile] [modelfile] [fraction] [seed]
#
#   example: score.sh ../data/train-sample.csv ../models/model.h5 0.25 42
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    score.sh [infile] [modelfile] [fraction]"
    echo " "
    echo "    where [infile] is the HDF5 file containing transformed training data (required)"
    echo "          [modelfile] is the fully qualified file to the model (required)"
    echo "          [fraction] is the fraction of data from the infile to be sampled nd scored (required)"
    echo "          [seed] is the random number seed (optional)"
    echo " "
    echo "    example 1:  ./score.sh ../data/train-sample.csv ../models/dense-model-checkpoint.h5 0.25 42"
    echo " "
}

xINFILE="$1"
if [ -z "$1" ]; then
  showHelp "[infile] parameter is missing"
  exit
fi
xINFILE=$(realpath $xINFILE)

xMODELFILE="$2"
if [ -z "$2" ]; then
  showHelp "[modelfile] parameter is missing"
  exit
fi
xMODELFILE=$(realpath $xMODELFILE)

xFRACTION="$3"
if [ -z "$3" ]; then
  showHelp "[fraction] parameter is missing"
  exit
fi

xSEED="$4"
if [ -z "$4" ]; then
  xSEED=42
fi

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/xsrc)

echo " "
echo "---- Train Parameters ----"
echo "Input (HDF5) File: $xINFILE"
echo "Model File:        $xMODELFILE"
echo "Sample pct:        $xFRACTION"
echo "Random seed:       $xSEED"
echo "Root Directory:    $xROOTDIR"
echo "Source Dir:        $xSRCDIR"
echo " "

echo "Clearing old TF log files"
rm ./logs/events.out.tfevents*

echo "Start: "; date
time python3 $xSRCDIR/score.py \
              --infile $xINFILE \
              --modelfile $xMODELFILE \
              --fraction $xFRACTION \
              --seed $xSEED
echo "End: "; date
