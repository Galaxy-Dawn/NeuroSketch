#!/bin/bash

# 设置训练参数
MODEL="Transformer"
PER_DEVICE_TRAIN_BATCH_SIZE=256 #256
PER_DEVICE_EVAL_BATCH_SIZE=512 #512
GPI_ID_LIST=(0 1 2 3)
NUM_TRAIN_EPOCHS=500
LEARNING_RATE=3e-4
INPUT_TYPE=1

for id in {1..4}; do
  python /root/NeuroSketch/run/train.py \
    model=$MODEL \
    model.input_type=$INPUT_TYPE \
    dataset=OpenMIIR \
    dataset.task=perception\
    dataset.id="$id" \
    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    training.gpu_id="${GPI_ID_LIST[$id%4]}" \
    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
    training.learning_rate=$LEARNING_RATE \
    wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_OpenMIIR_perception_${id}" &
  done

wait

for id in {5..9}; do
  python /root/NeuroSketch/run/train.py \
    model=$MODEL \
    model.input_type=$INPUT_TYPE \
    dataset=OpenMIIR \
    dataset.task=perception\
    dataset.id="$id"\
    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    training.gpu_id="${GPI_ID_LIST[$id%4]}" \
    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
    training.learning_rate=$LEARNING_RATE \
    wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_OpenMIIR_perception_${id}" &
  done

wait

for id in {1..4}; do
  python /root/NeuroSketch/run/train.py \
    model=$MODEL \
    model.input_type=$INPUT_TYPE \
    dataset=OpenMIIR \
    dataset.task=imagination\
    dataset.id="$id"\
    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    training.gpu_id="${GPI_ID_LIST[$id%4]}" \
    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
    training.learning_rate=$LEARNING_RATE \
    wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_OpenMIIR_imagination_${id}" &
  done

wait

for id in {5..9}; do
  python /root/NeuroSketch/run/train.py \
    model=$MODEL \
    model.input_type=$INPUT_TYPE \
    dataset=OpenMIIR \
    dataset.task=imagination\
    dataset.id="$id"\
    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    training.gpu_id="${GPI_ID_LIST[$id%4]}" \
    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
    training.learning_rate=$LEARNING_RATE \
    wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_OpenMIIR_imagination_${id}" &
  done
#
#wait

#PER_DEVICE_TRAIN_BATCH_SIZE=64
#PER_DEVICE_EVAL_BATCH_SIZE=128
#
#python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=Chisco \
#      dataset.task=read\
#      dataset.id=1 \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[2]}" \
#      training.num_train_epochs=200 \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_Chisco_read_1" &
#
#python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=Chisco \
#      dataset.task=read\
#      dataset.id=2 \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[3]}" \
#      training.num_train_epochs=200 \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_Chisco_read_2" &
#
#
#PER_DEVICE_TRAIN_BATCH_SIZE=64
#PER_DEVICE_EVAL_BATCH_SIZE=128
#
## 运行训练脚本
#python /root/NeuroSketch/run/train.py \
#    model=$MODEL \
#    model.input_type=$INPUT_TYPE \
#    model.patch_len=10\
#    model.patch_stride=10\
#    dataset=ThingsEEG \
#    dataset.task=test\
#    dataset.id=1 \
#    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#    training.gpu_id="${GPI_ID_LIST[2]}" \
#    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#    training.learning_rate=$LEARNING_RATE \
#    wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_ThingsEEG_test_1" &
#
#python /root/NeuroSketch/run/train.py \
#    model=$MODEL \
#    model.input_type=$INPUT_TYPE \
#    model.patch_len=10\
#    model.patch_stride=10\
#    dataset=ThingsEEG \
#    dataset.task=test\
#    dataset.id=2 \
#    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#    training.gpu_id="${GPI_ID_LIST[3]}" \
#    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#    training.learning_rate=$LEARNING_RATE \
#    wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_ThingsEEG_test_2" &
#
#wait
#
#PER_DEVICE_TRAIN_BATCH_SIZE=64
#PER_DEVICE_EVAL_BATCH_SIZE=128
#
#for id in {3..5}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=Chisco \
#      dataset.task=read\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=200 \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_Chisco_read_${id}" &
#done
#
#python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=Chisco \
#      dataset.task=imagine\
#      dataset.id=1 \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[2]}" \
#      training.num_train_epochs=200 \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_Chisco_imagine_1" &
#
#wait
#
#for id in {2..5}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=Chisco \
#      dataset.task=imagine\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=200 \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_Chisco_imagine_${id}" &
#done
#
#wait
#
#PER_DEVICE_TRAIN_BATCH_SIZE=64
#PER_DEVICE_EVAL_BATCH_SIZE=128
#LEARNING_RATE=5e-5
#
#for id in {3..6}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      model.patch_len=10\
#      model.patch_stride=5\
#      dataset=ThingsEEG \
#      dataset.task=test\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_ThingsEEG_test_${id}" &
#done

#
#wait
#
#for id in {7..10}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      model.patch_len=10\
#      model.patch_stride=10\
#      dataset=ThingsEEG \
#      dataset.task=test\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_ThingsEEG_test_${id}" &
#done
#
#wait
#
#PER_DEVICE_TRAIN_BATCH_SIZE=64
#PER_DEVICE_EVAL_BATCH_SIZE=128
#NUM_TRAIN_EPOCHS=500
#
#for id in {1..4}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=duin \
#      dataset.task=word_classification\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_duin_word_classification_${id}" &
#done
#
#wait
#
#for id in {5..8}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=duin \
#      dataset.task=word_classification\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_duin_word_classification_${id}" &
#done
#
#wait
#
#for id in {9..12}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=duin \
#      dataset.task=word_classification\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_duin_word_classification_${id}" &
#done
#
#wait
#
#PER_DEVICE_TRAIN_BATCH_SIZE=64
#PER_DEVICE_EVAL_BATCH_SIZE=128
#NUM_TRAIN_EPOCHS=500
#
#for id in {1..4}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=seed \
#      dataset.task=concept_classification\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_seed_${id}" &
#done
#
#wait
#
#for id in {5..8}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=seed \
#      dataset.task=concept_classification\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_seed_${id}" &
#done
#
#wait
#
#for id in {9..12}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=seed \
#      dataset.task=concept_classification\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_seed_${id}" &
#done
#
#wait
#
#for id in {13..16}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=seed \
#      dataset.task=concept_classification\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_seed_${id}" &
#done
#
#wait
#
#for id in {17..20}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      dataset=seed \
#      dataset.task=concept_classification\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_seed_${id}" &
#done
#
#wait
#
#for id in {0..3}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      model.patch_len=20\
#      model.patch_stride=20\
#      dataset=faceshouses \
#      dataset.task=faceshouses\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_faceshouses_${id}" &
#done
#
#wait
#
#for id in {4..7}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      model.patch_len=20\
#      model.patch_stride=20\
#      dataset=faceshouses \
#      dataset.task=faceshouses\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_faceshouses_${id}" &
#done
#
#wait
#
#for id in {8..11}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      model.patch_len=20\
#      model.patch_stride=20\
#      dataset=faceshouses \
#      dataset.task=faceshouses\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_faceshouses_${id}" &
#done
#
#wait
#
#for id in {12..13}; do
#  # 运行训练脚本
#  python /root/NeuroSketch/run/train.py \
#      model=$MODEL \
#      model.input_type=$INPUT_TYPE \
#      model.patch_len=20\
#      model.patch_stride=20\
#      dataset=faceshouses \
#      dataset.task=faceshouses\
#      dataset.id="$id" \
#      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
#      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
#      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
#      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
#      training.learning_rate=$LEARNING_RATE \
#      wandb.exp_name="${MODEL}_input_type_${INPUT_TYPE}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_faceshouses_${id}" &
#done