#!/bin/bash

xITERATIONS="$1"
if [ -z "$1" ]; then
  xITERATIONS=1
fi

xCOUNTER=0
while [ $xCOUNTER -lt $xITERATIONS ]; do

  echo "----- START ITERATION $xCOUNTER (of $xITERATIONS total) -----"

  xSEED=$xCOUNTER
  ./train.sh ../data/transform-train-0.01.csv ../models ../logs 10 100 $xSEED

  echo "----- END ITERATION $xCOUNTER (of $xITERATIONS total) -----"

  let xCOUNTER=xCOUNTER+1
done
