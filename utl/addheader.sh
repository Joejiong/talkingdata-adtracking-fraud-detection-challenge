#!/bin/bash

xHEADERFILE=../data/header.csv
for xFILENAME in ../data/transform-test-00*.csv ; do
  xOUTPUTFILE=$xFILENAME.hdr
  echo "Concatenating: $xHEADERFILE with $xFILENAME, output: $xOUTPUTFILE"
  cat $xHEADERFILE $xFILENAME > $xOUTPUTFILE
  rm $xFILENAME
  echo "Renaming: $xOUTPUTFILE to: $xFILENAME"
  mv $xOUTPUTFILE $xFILENAME
done
