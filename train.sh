#!/bin/bash

log_dir=/path/to/Terpene-former/log

data_dir=/path/to/Terpene-former/data/Retro-tero/random_split
intermediate_dir=/path/to/Terpene-former/data/intermediate
checkpoint_dir=/path/to/Terpene-former/ckpts

export PYTHONPATH=$(pwd)

python -u train.py \
  --log_dir $log_dir \
  --encoder_num_layers 8 \
  --decoder_num_layers 8 \
  --heads 8 \
  --max_step 400000 \
  --max_epoch 2000 \
  --batch_size_trn 8 \
  --batch_size_val 8 \
  --batch_size_token 4096 \
  --save_per_step 2500 \
  --val_per_step 2500 \
  --report_per_step 200 \
  --device cuda:1 \
  --rxnfp_class \
  --rxnfp_dist euc \
  --rxnfp_num_clusters 10 \
  --rxnfp_class_weights 0.982 3.2424528301886792 0.5069321533923303 1.4814655172413793 1.2144876325088338 1.1087096774193548 4.091666666666667 0.3764512595837897 1.8578378378378377 1.1611486486486486 \
  --shared_vocab \
  --data_dir $data_dir \
  --intermediate_dir $intermediate_dir \
  --checkpoint_dir $checkpoint_dir 2>&1 | tee -a log/out_train.txt

# bash train.sh