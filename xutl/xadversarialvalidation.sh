#!/bin/bash

#
# Purpose:  Adversarial Validation script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [trainfile] is the CSV file containing training data (required)
#    [testfiler] is the fully qualified directory to store checkpoint models (required)
#    [log-dir] is the fully qualified directory where the tensorboard logs will be saved (required)
#    [epochs] is the number of epochs (optional, default: 100)
#    [batch] is the batch size (optional, default: 1000)
#    [seed] is the random seed (optional, default: 0)
#
# Usage:
#
#   adversarialvalidation.sh [trainfile] [validationfile] [testfiler] [log-dir] [epochs] [batch]
#
#   example: adversarialvalidation.sh ../data/adversarialvalidation_sample.csv ../models ../logs 100 1000
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    adversarialvalidation.sh [trainfile] [testfiler] [log-dir] [epochs] [batch] [seed]"
    echo " "
    echo "    where [trainfile] is the HDF5 file containing training data (required)"
    echo "          [testfiler] is the HDF5 file containing test data (required)"
    echo "          [seed] is the random seed (optional, default: 0)"
    echo " "
    echo "    example 1:  ./adversarialvalidation.sh ../data/train_sample.csv ../data/test_sample.csv 0"
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
  showHelp "[testfiler] parameter is missing"
  exit
fi
xTESTFILE=$(realpath $xTESTFILE)

xSEED="$3"
if [ -z "$3" ]; then
  xSEED=0
fi

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/xsrc)

echo " "
echo "---- Autoencoder Parameters ----"
echo "Training File:   $xTRAINFILE"
echo "Model Directory: $xTESTFILE"
echo "Seed:            $xSEED"
echo "Root Directory:  $xROOTDIR"
echo "Source Dir:      $xSRCDIR"
echo " "

echo "Clearing old TF log files"
rm ./logs/events.out.tfevents*

echo "Start: "; date
time python3 $xSRCDIR/adversarialvalidation.py \
              --trainfile $xTRAINFILE \
              --testfile $xTESTFILE \
              --seed $xSEED
echo "End: "; date
