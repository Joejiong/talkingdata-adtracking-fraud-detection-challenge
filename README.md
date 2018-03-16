# TalkingData-AdTracking-Fraud-Detection-Challenge

## Scores

Small (Sample) Data Set:

                    (H)     (L)     (L)     (H)
AUC-ACT  AUC        0/0     0/1     1/0     1/1   ITER Configuration
-------- -------- ------- ------- ------- ------- ---- -----------------------------------------------
0.938020 0.914246   18639    1309      12      40    8 B:100/100/100/100/100; LAY: 2000/2; DROP:0.5/0.5/0.5; act:tanh
0.926556 0.891024   19149     799      20      32    8 B:100/100/100/100/100; LAY: 2000/3; DROP:0.5/0.5/0.5; act:tanh
0.921457 0.899576   19028     920      19      33   11 B:100/100/100/100/100; LAY: 2000/4; DROP:0.5/0.5/0.5; act:tanh
0.914446 0.883641   17882    2066      15      37   14 B:100/100/100/100/100; LAY: 2000/5; DROP:0.5/0.5/0.5; act:tanh
0.932958 0.896276   18366    1582      12      40  100 B:100/100/100/100/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh
0.930129 0.903819   18176    1772      12      40  100 B:100/100/100/100/100; LAY: 2000/3; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh
0.927822 0.896819   18338    1610      12      40  100 B:100/100/100/100/100; LAY: 2000/4; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh
0.897956 0.879790   17938    2010      16      36  100 B:100/100/100/100/100; LAY:  500/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh
0.907982 0.880270   18557    1391      14      38  200 B:100/100/100/100/100; LAY:  500/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh
0.900972 0.893062   18897    1051      17      35  200 B:100/100/100/100/100; LAY:  500/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:relu
0.878358 0.901096   19605     343      25      27  500 B:100/100/100/100/100; LAY:  500/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:relu
0.925895 0.903435   18677    1271      14      38  100 B:100/100/100/100/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh
0.925287 0.931351   19150     798      16      36  100 B:1000/100/100/100/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh (ip did not improve score)
0.954196 0.949834   19302     646      16      36  100 B:100/418/100/100/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh (app improves score)
0.932685 0.899116   18917    1031      17      35  100 B:100/100/395/100/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh (os helps a bit)
0.933927 0.918936   18791    1157      14      38  100 B:100/100/100/179/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh (channel helps a bit)
0.923330 0.903572   18701    1247      15      37  100 B:100/100/100/100/1985; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh (device did not improve score)
0.955844 0.939349   18881    1067      12      40  100 B:100/418/395/179/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-5; act:tanh
0.954864 0.907145   18968     980      11      41  200 B:100/418/395/179/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-6; act:tanh (reduce LR, increase ITER)
0.967430 0.928058   18988     960       9      43  400 B:100/418/395/179/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-6; act:tanh (larger increase ITER)
0.964460 0.933468   19026     922      12      40  600 B:100/418/395/179/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-6; act:tanh (larger increase ITER)
0.000000 0.000000   00000    0000      00      00  100 B:100/418/395/179/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-6; act:tanh (use probabilities instead of binary 1/0)
0.000000 0.000000   00000    0000      00      00   00
0.000000 0.000000   00000    0000      00      00   00
0.000000 0.000000   00000    0000      00      00   00
0.000000 0.000000   00000    0000      00      00   00



## Instructions

Describe the raw data files:

```
  ./describe.sh ../data/train_sample.csv

      input (csv) file:  ../data/train_sample.csv
```

Sample the raw data files into smaller subsets for experimentation:

```
  ./sample.sh /Volumes/Seagate-4TB-USB/data/talkingdata-adtracking-fraud-detection-challenge/train.csv 0.10 ../data/full-0.10.csv

      input (csv) file:   /Volumes/Seagate-4TB-USB/data/talkingdata-adtracking-fraud-detection-challenge/train.csv
      pct:                0.10
      output (csv) file:  ../data/full-0.10.csv
```

Transform raw TRAINING files into processed TRAINING files:

```
  ./transform.sh ../data/train_sample.csv ../data/transform-train-sample.csv 15 14 13 12 10

      csv (input) file:   ../data/train_sample.csv
      csv (output) file:  ../data/transform-train-sample.csv
      ip buckets:         15
      app buckets:        15
      os buckets:         15
      channel buckets:    15
      device buckets:     15
```

Train the model:

```
  ./train.sh ../data/transform-train-sample.csv ../models ../logs 10 1000 0

      csv (input) file:   ../data/transform-sample.csv
      model (output) dir: ../models
      Tensorflow log dir: ../logs
      Epochs:             10
      Batch size:         1000
      Random seed:        0
```

Make a subset of the test file to experiment with submission file creation:

```
  head -n 10000 ../data/test.csv > ../data/test_sample.csv

      number of records:  10000
      csv (input) file:   ../data/test.csv
      csv (output) file:  ../data/test_sample.csv
```

Transform raw TEST files into processed TEST files:

```
  ./transform.sh ../data/test_sample.csv ../data/transform-test-sample.csv 15

      csv (input) file:   ../data/test_sample.csv
      csv (output) file:  ../data/transform-test-sample.csv
```

Make a prediction and create the submission file:

```
  [show submit command]
```
