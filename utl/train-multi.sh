#!/bin/bash

xITERATIONS=5
xCOUNTER=0
while [ $xCOUNTER -lt $xITERATIONS ]; do

  echo "----- START ITERATION $xCOUNTER (of $xITERATIONS total) -----"

  xSEED=$xCOUNTER
  ./train.sh ../data/transform-train-sample.csv ../models ../logs 100 1000 $xSEED

  echo "----- END ITERATION $xCOUNTER (of $xITERATIONS total) -----"

  let xCOUNTER=xCOUNTER+1
done
