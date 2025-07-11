#!/bin/bash

data_dir=/path/to/Terpene-former/data/TeroRXN/random_split
intermediate_dir=/path/to/Terpene-former/data/intermediate
checkpoint_dir=/path/to/Terpene-former/ckpts

cd code
export PYTHONPATH=$(pwd)

python translate.py \
   --batch_size_val 8 \
   --shared_vocab \
   --device cuda:1 \
   --data_dir $data_dir \
   --intermediate_dir $intermediate_dir \
   --checkpoint_dir $checkpoint_dir \
   --ckpt_range \
   --range_begin 200000 \
   --range_end 400000 \
   --rxnfp_class \
   --rxnfp_dist euc \
   --rxnfp_num_clusters 10 \
   --beam_size 10 2>&1 | tee -a log/out_translate.txt

# bash translate.sh
