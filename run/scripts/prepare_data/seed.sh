#!/bin/bash

DATASET_NAME="seed"

python ./root/NeuroSketch/run/prepare_data/prepare_seed_data.py -m\
      dataset=${DATASET_NAME} \
      split_method='simple' \
      hydra.sweeper.params.dataset.id=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20

wait

python ./root/NeuroSketch/run/prepare_data/prepare_seed_data.py -m\
      dataset=${DATASET_NAME} \
      split_method='n_fold' \
      hydra.sweeper.params.dataset.id=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
      hydra.sweeper.params.dataset.fold=0,1,2
