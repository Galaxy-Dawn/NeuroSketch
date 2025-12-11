#!/bin/bash

DATASET_NAME="OpenMIIR"

python ./root/NeuroSketch/run/prepare_data/prepare_openmiir_data.py -m\
      dataset=${DATASET_NAME} \
      dataset.task='imagination' \
      split_method='simple' \
      hydra.sweeper.params.dataset.id=1,2,3,4,5,6,7,8,9

wait

python ./root/NeuroSketch/run/prepare_data/prepare_openmiir_data.py -m\
      dataset=${DATASET_NAME} \
      dataset.task='imagination' \
      split_method='n_fold' \
      hydra.sweeper.params.dataset.id=1,2,3,4,5,6,7,8,9 \
      hydra.sweeper.params.dataset.fold=0,1,2

wait

python ./root/NeuroSketch/run/prepare_data/prepare_openmiir_data.py -m\
      dataset=${DATASET_NAME} \
      dataset.task='perception' \
      split_method='simple' \
      hydra.sweeper.params.dataset.id=1,2,3,4,5,6,7,8,9

wait

python ./root/NeuroSketch/run/prepare_data/prepare_openmiir_data.py -m\
      dataset=${DATASET_NAME} \
      dataset.task='perception' \
      split_method='n_fold' \
      hydra.sweeper.params.dataset.id=1,2,3,4,5,6,7,8,9 \
      hydra.sweeper.params.dataset.fold=0,1,2