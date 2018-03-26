# TalkingData-AdTracking-Fraud-Detection-Challenge

## Autoencoder

To train the autoencoder and create the autoencoder-final.h5 file:

```
  ./autoencoder.sh ../data/transform-train-100000.csv ../models ../logs 2 1000 2 train 25
```

Note that the training occurs *very* quickly: only 2-3 epochs are requires
and additional epochs make no material difference!

To make predictions from test file using autoencoder (assumes model
file of autoencoder-final.h5 is available from training):

```
  ./autoencoder.sh ../data/transform-test-0000300000-0000400000.csv ../models ../logs 10 1000 2 predict 25000

  (25000 is a threshold)
```

## Data Characteristics

```
  TEST DATA

  ---- Describe Parameters ----
  CSV File:       /Users/ericbroda/Development/python/kaggle/TalkingData-AdTracking-Fraud-Detection-Challenge/data/test.csv
  Root Directory: /Users/ericbroda/Development/python/kaggle/TalkingData-AdTracking-Fraud-Detection-Challenge
  Source Dir:     /Users/ericbroda/Development/python/kaggle/TalkingData-AdTracking-Fraud-Detection-Challenge/src

  Start:
  Sat 17 Mar 2018 10:36:45 EDT
  Using csvfile:  /Users/ericbroda/Development/python/kaggle/TalkingData-AdTracking-Fraud-Detection-Challenge/data/test.csv
  Loading data, csvfile:  /Users/ericbroda/Development/python/kaggle/TalkingData-AdTracking-Fraud-Detection-Challenge/data/test.csv
  Describing data
     click_id      ip  app  device  os  channel           click_time
  0         0    5744    9       1   3      107  2017-11-10 04:00:00
  1         1  119901    9       1   3      466  2017-11-10 04:00:00
  2         2   72287   21       1  19      128  2017-11-10 04:00:00
  3         3   78477   15       1  13      111  2017-11-10 04:00:00
  4         4  123080   12       1  13      328  2017-11-10 04:00:00
             click_id            ip           app        device            os  \
  count  1.879047e+07  1.879047e+07  1.879047e+07  1.879047e+07  1.879047e+07
  mean   9.395234e+06  6.306921e+04  1.221480e+01  1.730513e+00  1.873312e+01
  std    5.424341e+06  3.688597e+04  1.164924e+01  2.597038e+01  1.135059e+01
  min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00
  25%    4.697617e+06  3.155800e+04  3.000000e+00  1.000000e+00  1.300000e+01
  50%    9.395234e+06  6.393600e+04  1.200000e+01  1.000000e+00  1.800000e+01
  75%    1.409285e+07  9.531600e+04  1.800000e+01  1.000000e+00  1.900000e+01
  max    1.879047e+07  1.264130e+05  5.210000e+02  3.031000e+03  6.040000e+02

              channel
  count  1.879047e+07
  mean   2.648059e+02
  std    1.355254e+02
  min    0.000000e+00
  25%    1.350000e+02
  50%    2.360000e+02
  75%    4.010000e+02
  max    4.980000e+02

  Characteristics:
  Column: app
    unique:      417
    maximum:     521
    minimum:     0
  Column: os
    unique:      395
    maximum:     604
    minimum:     0
  Column: channel
    unique:      178
    maximum:     498
    minimum:     0
  Column: ip
    unique:      93936
    maximum:     126413
    minimum:     0
  Column: device
    unique:      1985
    maximum:     3031
    minimum:     0
```

```
  TRAIN SAMPLE Data

    ip           app         device             os  \
  count  100000.000000  100000.00000  100000.000000  100000.000000
  mean    91255.879670      12.04788      21.771250      22.818280
  std     69835.553661      14.94150     259.667767      55.943136
  min         9.000000       1.00000       0.000000       0.000000
  25%     40552.000000       3.00000       1.000000      13.000000
  50%     79827.000000      12.00000       1.000000      18.000000
  75%    118252.000000      15.00000       1.000000      19.000000
  max    364757.000000     551.00000    3867.000000     866.000000

  channel  is_attributed
  count  100000.000000  100000.000000
  mean      268.832460       0.002270
  std       129.724248       0.047591
  min         3.000000       0.000000
  25%       145.000000       0.000000
  50%       258.000000       0.000000
  75%       379.000000       0.000000
  max       498.000000       1.000000

  Characteristics:
  Column: app
  unique:      161
  maximum:     551
  minimum:     1
  Column: os
  unique:      130
  maximum:     866
  minimum:     0
  Column: channel
  unique:      161
  maximum:     498
  minimum:     3
  Column: ip
  unique:      34857
  maximum:     364757
  minimum:     9
  Column: device
  unique:      100
  maximum:     3867
  minimum:     0
```

## Scores

```
  Small (Sample) Data Set (seed: 42):

                      (H)     (L)     (L)     (H)
  AUC-ACT  AUC        0/0     0/1     1/0     1/1   ITER RUN Configuration
  -------- -------- ------- ------- ------- ------- ---- --- -----------------------------------------------
  0.966351 0.952914   19176     773       8      43   20   X BASE: lr:1.0*1e-4; std data; DATA:10.418.395.179.10
  0.966349 0.952869   19184     765       8      43   20   2 Added ip_device group (REMOVE)
  0.967342 0.952293   18730    1219       5      46   20   3 Added ip_day (KEEP)
  0.967393 0.949396   18741    1208       5      46   20   4 Added ip_hour (KEEP)
  0.955577 0.909838   19243     706       6      45   20   5 Added app_channel (REMOVE)
  0.978309 0.953235   19381     568       7      44   20   6 Added app_qty (KEEP)
  0.943659 0.907197   19127     822       7      44   20   7 Added app_day (REMOVE)
  0.969461 0.932429   19287     662       6      45   20   8 Added app_hour (REMOVE)
  0.950983 0.953261   19022     927       8      43   20   9 Added os_qty (REMOVE)
  0.000000 0.000000   00000    0000      00      00   00
  0.000000 0.000000   00000    0000      00      00   00
  0.000000 0.000000   00000    0000      00      00   00
  0.000000 0.000000   00000    0000      00      00   00
  0.000000 0.000000   00000    0000      00      00   00
```


```
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
**0.967430 0.928058   18988     960       9      43  400 B:100/418/395/179/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-6; act:tanh (larger increase ITER)
  0.964460 0.933468   19026     922      12      40  600 B:100/418/395/179/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-6; act:tanh (larger increase ITER)
  0.919694 0.000000   00000    0000      00      00  200 B:100/418/395/179/100; LAY: 2000/2; DROP:0.5/0.5/0.5, LR: 1.0*1e-6; act:tanh (use probabilities instead of binary 1/0)
  0.000000 0.000000   00000    0000      00      00   00
  0.000000 0.000000   00000    0000      00      00   00
  0.000000 0.000000   00000    0000      00      00   00
  0.000000 0.000000   00000    0000      00      00   00
```


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
