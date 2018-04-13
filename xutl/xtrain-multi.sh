#!/bin/bash

xITERATIONS=5
xCOUNTER=0
while [ $xCOUNTER -lt $xITERATIONS ]; do

  xSEED=$xCOUNTER
  let xCOUNTER=xCOUNTER+1

  echo "----- START ITERATION $xCOUNTER (of $xITERATIONS total) SEED: $xSEED -----"

  ./xtrain.sh ../data/transform-train-0.010.h5 ../data/transform-test.h5 ../models ../logs 5 10000 $xSEED | grep AUC

  echo "----- END ITERATION $xCOUNTER (of $xITERATIONS total) -----"

done
