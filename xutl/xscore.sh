#!/bin/bash

#
# Purpose:  Training script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [trainfile] is the CSV file containing training data (required)
#    [model-dir] is the fully qualified directory to store checkpoint models (required)
#    [log-dir] is the fully qualified directory where the tensorboard logs will be saved (required)
#    [epochs] is the number of epochs (optional, default: 100)
#    [batch] is the batch size (optional, default: 1000)
#    [seed] is the random seed (optional, default: 0)
#
# Usage:
#
#   train.sh [trainfile] [testfile] [model-dir] [log-dir] [epochs] [batch]
#
#   example: train.sh ../data/train_sample.csv ../models ../logs 100 1000
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    train.sh [trainfile] [testfile] [model-dir] [log-dir] [epochs] [batch] [seed]"
    echo " "
    echo "    where [trainfile] is the CSV file containing training data (required)"
    echo "          [testfile] is the CSV file containing test data (required)"
    echo "          [model-dir] is the fully qualified directory to store checkpoint models (required)"
    echo "          [log-dir] is the fully qualified directory where the tensorboard logs will be saved (required)"
    echo "          [epochs] is the number of epochs (optional, default: 100)"
    echo "          [batch] is the batch size (optional, default: 1000)"
    echo "          [seed] is the random seed (optional, default: 0)"
    echo " "
    echo "    example 1:  ./train.sh ../data/train_sample.csv ../data/test_sample.csv ../models ../logs 100 1000 0"
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

xMODELDIR="$3"
if [ -z "$3" ]; then
  showHelp "[model-dir] parameter is missing"
  exit
fi
xMODELDIR=$(realpath $xMODELDIR)

xLOGDIR="$4"
if [ -z "$4" ]; then
  showHelp "[log-dir] parameter is missing"
  exit
fi
xLOGDIR=$(realpath $xLOGDIR)

xEPOCHS="$5"
if [ -z "$5" ]; then
  xEPOCHS=100
fi

xBATCH="$6"
if [ -z "$6" ]; then
  xBATCH=1000
fi

xSEED="$7"
if [ -z "$7" ]; then
  xSEED=0
fi

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/xsrc)

echo " "
echo "---- Train Parameters ----"
echo "Training File:   $xTRAINFILE"
echo "Test File:       $xTESTFILE"
echo "Model Directory: $xMODELDIR"
echo "Log Directory:   $xLOGDIR"
echo "Epochs:          $xEPOCHS"
echo "Batch Size:      $xBATCH"
echo "Seed:            $xSEED"
echo "Root Directory:  $xROOTDIR"
echo "Source Dir:      $xSRCDIR"
echo " "

echo "Clearing old TF log files"
rm ./logs/events.out.tfevents*

echo "Start: "; date
time python3 $xSRCDIR/train.py \
              --trainfile $xTRAINFILE \
              --testfile $xTESTFILE \
              --modeldir $xMODELDIR \
              --logdir $xLOGDIR \
              --epochs $xEPOCHS \
              --batch $xBATCH \
              --seed $xSEED
echo "End: "; date
