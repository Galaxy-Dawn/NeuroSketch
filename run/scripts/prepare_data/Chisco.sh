#!/bin/bash

DATASET_NAME="Chisco"

python ./root/NeuroSketch/run/prepare_data/prepare_chisco_data.py -m\
      dataset=${DATASET_NAME} \
      dataset.task='read' \
      split_method='simple' \
      hydra.sweeper.params.dataset.id=1,2,3,4,5

wait

python ./root/NeuroSketch/run/prepare_data/prepare_chisco_data.py -m\
      dataset=${DATASET_NAME} \
      dataset.task='read' \
      split_method='n_fold' \
      hydra.sweeper.params.dataset.id=1,2,3,4,5 \
      hydra.sweeper.params.dataset.fold=0,1,2

wait

python ./root/NeuroSketch/run/prepare_data/prepare_chisco_data.py -m\
      dataset=${DATASET_NAME} \
      dataset.task='imagine' \
      split_method='simple' \
      hydra.sweeper.params.dataset.id=1,2,3,4,5

wait

python ./root/NeuroSketch/run/prepare_data/prepare_chisco_data.py -m\
      dataset=${DATASET_NAME} \
      dataset.task='imagine' \
      split_method='n_fold' \
      hydra.sweeper.params.dataset.id=1,2,3,4,5 \
      hydra.sweeper.params.dataset.fold=0,1,2