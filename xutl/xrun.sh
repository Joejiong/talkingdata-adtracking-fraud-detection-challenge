#!/bin/bash

# echo "Model: dense-model-final-0.25-0.980459.h5"
# Leadership board score:  0.9632 (avg: 0.9821, delta: -0.0189)
# echo "Run: 0"
# 0.981532
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.25-0.980459.h5 0.25 0 | grep AUC
#
# echo "Run: 1"
# 0.981424
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.25-0.980459.h5 0.25 1 | grep AUC
#
# echo "Run: 2"
# 0.982704
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.25-0.980459.h5 0.25 2 | grep AUC
#
# echo "Run: 3"
# 0.983574
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.25-0.980459.h5 0.25 3 | grep AUC
#
# echo "Run: 4"
# 0.981641
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.25-0.980459.h5 0.25 4 | grep AUC

# echo " "
# echo " "
# echo "Model: dense-model-final-0.977073.h5"
# # Leadership board score:  0.9652 (avg: 0.9774, delta: -0.0122)
# echo "Run: 0"
# # 0.976601
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.50-0.977073.h5 0.25 0 | grep AUC
#
# echo "Run: 1"
# # 0.976463
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.50-0.977073.h5 0.25 1 | grep AUC
#
# echo "Run: 2"
# # 0.978075
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.50-0.977073.h5 0.25 2 | grep AUC
#
# echo "Run: 3"
# # 0.979029
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.50-0.977073.h5 0.25 3 | grep AUC
#
# echo "Run: 4"
# # 0.976927
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.50-0.977073.h5 0.25 4 | grep AUC

# echo " "
# echo " "
# echo "Model: dense-model-checkpoint-0.50-0.977073.h5"
# # Leadership board score:  0.???? (avg: 0.9768, delta: -0.????)
# echo "Run: 0"
# # 0.975865
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-checkpoint-0.50-0.977073.h5 0.25 0 | grep AUC
#
# echo "Run: 1"
# # 0.975619
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-checkpoint-0.50-0.977073.h5 0.25 1 | grep AUC
#
# echo "Run: 2"
# # 0.977381
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-checkpoint-0.50-0.977073.h5 0.25 2 | grep AUC
#
# echo "Run: 3"
# # 0.978354
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-checkpoint-0.50-0.977073.h5 0.25 3 | grep AUC
#
# echo "Run: 4"
# # 0.976927
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-checkpoint-0.50-0.977073.h5 0.25 4 | grep AUC


# echo " "
# echo " "
# echo "Model: dense-model-final-0.10-0.976596x.h5 (weights: 0.501:203.567)"
# echo "Run: 0"
# # 0.980343
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.10-0.976596x.h5 0.25 0 | grep AUC
#
# echo "Run: 1"
# # 0.980415
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.10-0.976596x.h5 0.25 1 | grep AUC
#
# echo "Run: 2"
# # 0.982020
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.10-0.976596x.h5 0.25 2 | grep AUC
#
# echo "Run: 3"
# # 0.982377
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.10-0.976596x.h5 0.25 3 | grep AUC
#
# echo "Run: 4"
# # 0.980399
# ./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.10-0.976596x.h5 0.25 4 | grep AUC

echo " "
echo " "
echo "Model: dense-model-final-0.10-0.983350.h5 (weights: 0.01:0.99)"
echo "Run: 0"
# 0.988189
./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.10-0.983350.h5 0.25 0 | grep AUC

echo "Run: 1"
# 0.987778
./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.10-0.983350.h5 0.25 1 | grep AUC

echo "Run: 2"
# 0.989003
./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.10-0.983350.h5 0.25 2 | grep AUC

echo "Run: 3"
# 0.988930
./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.10-0.983350.h5 0.25 3 | grep AUC

echo "Run: 4"
# 0.988108
./xscore.sh ../data/transform-train-0.05.h5 ../models/dense-model-final-0.10-0.983350.h5 0.25 4 | grep AUC
