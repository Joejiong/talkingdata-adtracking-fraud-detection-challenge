#!/bin/bash

#
# Purpose:  Autoencoder script
#
# Author:  Eric Broda, ericbroda@rogers.com
#
# Parameters:
#    [csvfile] is the CSV file containing autoencodering data (required)
#    [model-dir] is the fully qualified directory to store checkpoint models (required)
#    [log-dir] is the fully qualified directory where the tensorboard logs will be saved (required)
#    [epochs] is the number of epochs (optional, default: 100)
#    [batch] is the batch size (optional, default: 1000)
#    [seed] is the random seed (optional, default: 0)
#
# Usage:
#
#   autoencoder.sh [csvfile] [model-dir] [log-dir] [epochs] [batch]
#
#   example: autoencoder.sh ../data/train_sample.csv ../models ../logs 100 1000
#

function showHelp {
    echo " "
    echo "Error: $1"
    echo " "
    echo "    autoencoder.sh [csvfile] [model-dir] [log-dir] [epochs] [batch] [seed]"
    echo " "
    echo "    where [csvfile] is the CSV file containing autoencodering data (required)"
    echo "          [model-dir] is the fully qualified directory to store checkpoint models (required)"
    echo "          [log-dir] is the fully qualified directory where the tensorboard logs will be saved (required)"
    echo "          [epochs] is the number of epochs (required)"
    echo "          [batch] is the batch size (required)"
    echo "          [seed] is the random seed (required)"
    echo "          [operation] is the set of operations to perform (required, comma separated, one of: train|predict)"
    echo "          [threshold] is the threshold for errors used in predictions (required)"
    echo " "
    echo "    example 1:  autoencoder.sh ../data/train_sample.csv ../models ../logs 100 1000 0"
    echo " "
}

xCSVFILE="$1"
if [ -z "$1" ]; then
  showHelp "[csvfile] parameter is missing"
  exit
fi
xCSVFILE=$(realpath $xCSVFILE)

xMODELDIR="$2"
if [ -z "$2" ]; then
  showHelp "[model-dir] parameter is missing"
  exit
fi
xMODELDIR=$(realpath $xMODELDIR)

xLOGDIR="$3"
if [ -z "$3" ]; then
  showHelp "[log-dir] parameter is missing"
  exit
fi
xLOGDIR=$(realpath $xLOGDIR)

xEPOCHS="$4"
if [ -z "$4" ]; then
  showHelp "[epochs] parameter is missing"
  exit
fi

xBATCH="$5"
if [ -z "$5" ]; then
  showHelp "[batch] parameter is missing"
  exit
fi

xSEED="$6"
if [ -z "$6" ]; then
  showHelp "[seed] parameter is missing"
  exit
fi

xOPERATION="$7"
if [ -z "$7" ]; then
  xOPERATION="train"
fi

xTHRESHOLD="$8"
if [ -z "$8" ]; then
  xTHRESHOLD=25
fi

xROOTDIR=$(realpath ../)
cd $xROOTDIR

xSRCDIR=$(realpath $xROOTDIR/src)

echo " "
echo "---- Train Parameters ----"
echo "CSV File:        $xCSVFILE"
echo "Model Directory: $xMODELDIR"
echo "Log Directory:   $xLOGDIR"
echo "Epochs:          $xEPOCHS"
echo "Batch Size:      $xBATCH"
echo "Seed:            $xSEED"
echo "Operation:       $xOPERATION"
echo "Threshold:       $xTHRESHOLD"
echo "Root Directory:  $xROOTDIR"
echo "Source Dir:      $xSRCDIR"
echo " "

echo "Clearing old TF log files"
rm ./logs/events.out.tfevents*

echo "Start: "; date
time python3 $xSRCDIR/autoencoder.py \
              --csvfile $xCSVFILE \
              --modeldir $xMODELDIR \
              --logdir $xLOGDIR \
              --epochs $xEPOCHS \
              --batch $xBATCH \
              --seed $xSEED \
              --operation $xOPERATION \
              --threshold $xTHRESHOLD
echo "End: "; date
