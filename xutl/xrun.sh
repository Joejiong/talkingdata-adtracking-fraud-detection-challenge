#!/bin/bash

echo "------ PCT: 0.001"
# using hour:                             0.985911
# adding label encoding for qhour/dqhour: 0.985953
# using qhour/dqhour:                     0.984657
# new model:                              0.961159
# confidence levels (new, bad model):     0.972970
# confidence levels (new, good model):    0.987029
# add ip_confRate:                        0.984339
# streamlined conf rates:                 0.963839
# batch 10,000 -> 5,000                   0.992751
# batch 10,000 -> 20,000                  0.978557
./xtrain.sh ../data/transform-train-0.001.h5  ../data/transform-test.h5 ../models ../logs 1  20000 2

echo "------ PCT: 0.005"
# using hour:                             0.963115
# adding label encoding for qhour/dqhour: 0.963170
# using qhour/dqhour:                     0.962935
# new model:                              0.961925
# confidence levels (new, bad model):     0.963681
# confidence levels (new, good model):    0.964177
# add ip_confRate:                        0.954461
# streamlined conf rates:                 0.965864
# batch 10,000 -> 5,000                   0.965435
# batch 10,000 -> 20,000                  0.960060
./xtrain.sh ../data/transform-train-0.005.h5  ../data/transform-test.h5 ../models ../logs 1  20000 2

echo "------ PCT: 0.010"
# using hour:                             0.971416
# adding label encoding for qhour/dqhour: 0.971416
# using qhour/dqhour:                     0.972254
# new model:                              0.970681
# confidence levels (new, bad model):     0.973678
# confidence levels (new, good model):    0.971271
# add ip_confRate:                        0.963885
# streamlined conf rates:                 0.971574
# batch 10,000 -> 5,000                   0.965291
# batch 10,000 -> 20,000                  0.971956
./xtrain.sh ../data/transform-train-0.010.h5  ../data/transform-test.h5 ../models ../logs 1  20000 2

echo "------ PCT: 0.050"
# using hour:                             0.968689
# adding label encoding for qhour/dqhour: 0.968687
# using qhour/dqhour:                     0.969384
# new model:                              0.966576
# confidence levels (new, bad model):     0.967011
# confidence levels (new, good model):    0.969026
# add ip_confRate:                        TBD
# streamlined conf rates:                 0.969251
# batch 10,000 -> 5,000                   0.968991
# batch 10,000 -> 20,000                  0.968938
./xtrain.sh ../data/transform-train-0.050.h5  ../data/transform-test.h5 ../models ../logs 1  20000 2

echo "------ PCT: 0.100"
# using hour:                             0.971938
# adding label encoding for qhour/dqhour: 0.971999
# using qhour/dqhour:                     0.971983
# new model:
# confidence levels (new, bad model):     0.971589
# confidence levels (new, good model):    0.971204
# add ip_confRate:                        TBD
# streamlined conf rates:                 0.971353
# batch 10,000 -> 5,000                   0.971732
# batch 10,000 -> 20,000                  0.971173
./xtrain.sh ../data/transform-train-0.100.h5  ../data/transform-test.h5 ../models ../logs 1  20000 2

# echo "------ PCT: 0.500"
# streamlined conf rates:                 0.974218
# batch 10,000 -> 5,000                   0.971732
# batch 10,000 -> 20,000                  TBD
./xtrain.sh ../data/transform-train-0.500.h5  ../data/transform-test.h5 ../models ../logs 1  20000 2
#
# echo "------ PCT: FULL (100%)"
# # using hour:
# # adding label encoding for qhour/dqhour:
# # using qhour/dqhour:
# ./xtrain.sh ../data/transform-train.h5  ../data/transform-test.h5 ../models ../logs 1  20000 2

# echo "------ PCT: 0.001"
# ./xsample.sh ../data/transform-train.h5 0.001 ../data/transform-train-0.001.h5
#
# echo "------ PCT: 0.005"
# ./xsample.sh ../data/transform-train.h5 0.005 ../data/transform-train-0.005.h5
#
# echo "------ PCT: 0.010"
# ./xsample.sh ../data/transform-train.h5 0.010 ../data/transform-train-0.010.h5
#
# echo "------ PCT: 0.050"
# ./xsample.sh ../data/transform-train.h5 0.050 ../data/transform-train-0.050.h5
#
# echo "------ PCT: 0.100"
# ./xsample.sh ../data/transform-train.h5 0.100 ../data/transform-train-0.100.h5
#
# echo "------ PCT: 0.200"
# ./xsample.sh ../data/transform-train.h5 0.200 ../data/transform-train-0.200.h5
#
# echo "------ PCT: 0.250"
# ./xsample.sh ../data/transform-train.h5 0.250 ../data/transform-train-0.250.h5
#
# echo "------ PCT: 0.500"
# ./xsample.sh ../data/transform-train.h5 0.500 ../data/transform-train-0.500.h5

# echo "------ PCT: SAMPLE"
# ./xoversample.sh ../data/transform-train-sample.h5 RANDOM
# ./xoversample.sh ../data/transform-train-sample.h5 SMOTE
# ./xoversample.sh ../data/transform-train-sample.h5 ADASYN
#
# echo "------ PCT: 0.001"
# ./xoversample.sh ../data/transform-train-0.001.h5 RANDOM
# ./xoversample.sh ../data/transform-train-0.001.h5 SMOTE
# ./xoversample.sh ../data/transform-train-0.001.h5 ADASYN
#
# echo "------ PCT: 0.005"
# ./xoversample.sh ../data/transform-train-0.005.h5 RANDOM
# ./xoversample.sh ../data/transform-train-0.005.h5 SMOTE
# ./xoversample.sh ../data/transform-train-0.005.h5 ADASYN
#
# echo "------ PCT: 0.010"
# ./xoversample.sh ../data/transform-train-0.010.h5 RANDOM
# ./xoversample.sh ../data/transform-train-0.010.h5 SMOTE
# ./xoversample.sh ../data/transform-train-0.010.h5 ADASYN
#
# echo "------ PCT: 0.050"
# ./xoversample.sh ../data/transform-train-0.050.h5 RANDOM
# ./xoversample.sh ../data/transform-train-0.050.h5 SMOTE
# ./xoversample.sh ../data/transform-train-0.050.h5 ADASYN
#
# echo "------ PCT: 0.100"
# ./xoversample.sh ../data/transform-train-0.100.h5 RANDOM
# ./xoversample.sh ../data/transform-train-0.100.h5 SMOTE
# ./xoversample.sh ../data/transform-train-0.100.h5 ADASYN
#
# echo "------ PCT: 0.200"
# ./xoversample.sh ../data/transform-train-0.200.h5 RANDOM
# ./xoversample.sh ../data/transform-train-0.200.h5 SMOTE
# ./xoversample.sh ../data/transform-train-0.200.h5 ADASYN
#
# echo "------ PCT: 0.250"
# ./xoversample.sh ../data/transform-train-0.250.h5 RANDOM
# ./xoversample.sh ../data/transform-train-0.250.h5 SMOTE
# ./xoversample.sh ../data/transform-train-0.250.h5 ADASYN
#
# echo "------ PCT: 0.500"
# ./xoversample.sh ../data/transform-train-0.500.h5 RANDOM
# ./xoversample.sh ../data/transform-train-0.500.h5 SMOTE
# ./xoversample.sh ../data/transform-train-0.500.h5 ADASYN
