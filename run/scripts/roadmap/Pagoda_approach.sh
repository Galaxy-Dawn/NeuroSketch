#!/bin/bash

# 设置训练参数[
MODEL="CNN_2D_downsample"
PER_DEVICE_TRAIN_BATCH_SIZE=256
PER_DEVICE_EVAL_BATCH_SIZE=512
GPI_ID_LIST=(4 5 6 7)
NUM_TRAIN_EPOCHS=500
LEARNING_RATE=1e-3
trial_id=1


for id in {1..4}; do
  python /root/NeuroSketch/run/train.py \
    model=$MODEL \
    dataset=OpenMIIR \
    dataset.task=perception\
    dataset.id="$id" \
    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    training.gpu_id="${GPI_ID_LIST[$id%4]}" \
    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
    training.learning_rate=$LEARNING_RATE \
    wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_OpenMIIR_perception_${id}_${trial_id}_d4" &
  done

wait

for id in {5..9}; do
  python /root/NeuroSketch/run/train.py \
    model=$MODEL \
    dataset=OpenMIIR \
    dataset.task=perception\
    dataset.id="$id"\
    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    training.gpu_id="${GPI_ID_LIST[$id%4]}" \
    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
    training.learning_rate=$LEARNING_RATE \
    wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_OpenMIIR_perception_${id}_${trial_id}_d4" &
  done

wait

for id in {1..4}; do
  python /root/NeuroSketch/run/train.py \
    model=$MODEL \
    dataset=OpenMIIR \
    dataset.task=imagination\
    dataset.id="$id"\
    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    training.gpu_id="${GPI_ID_LIST[$id%4]}" \
    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
    training.learning_rate=$LEARNING_RATE \
    wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_OpenMIIR_imagination_${id}_${trial_id}_d4" &
  done

wait

for id in {5..9}; do
  python /root/NeuroSketch/run/train.py \
    model=$MODEL \
    dataset=OpenMIIR \
    dataset.task=imagination\
    dataset.id="$id"\
    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    training.gpu_id="${GPI_ID_LIST[$id%4]}" \
    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
    training.learning_rate=$LEARNING_RATE \
    wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_OpenMIIR_imagination_${id}_${trial_id}_d4" &
  done

wait

PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=128

python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=Chisco \
      dataset.task=read\
      dataset.id=1 \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[2]}" \
      training.num_train_epochs=200 \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_Chisco_read_1_${trial_id}_d4" &

python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=Chisco \
      dataset.task=read\
      dataset.id=2 \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[3]}" \
      training.num_train_epochs=200 \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_Chisco_read_2_${trial_id}_d4" &

wait

PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=128

# 运行训练脚本
python /root/NeuroSketch/run/train.py \
    model=$MODEL \
    model.seg_dim=1 \
    dataset=ThingsEEG \
    dataset.task=test\
    dataset.id=1 \
    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    training.gpu_id="${GPI_ID_LIST[2]}" \
    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
    training.learning_rate=$LEARNING_RATE \
    wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_ThingsEEG_test_1_${trial_id}_d4" &

python /root/NeuroSketch/run/train.py \
    model=$MODEL \
    model.seg_dim=1 \
    dataset=ThingsEEG \
    dataset.task=test\
    dataset.id=2 \
    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    training.gpu_id="${GPI_ID_LIST[3]}" \
    training.num_train_epochs=$NUM_TRAIN_EPOCHS \
    training.learning_rate=$LEARNING_RATE \
    wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_ThingsEEG_test_2_${trial_id}_d4" &

wait


PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=128

for id in {3..5}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=Chisco \
      dataset.task=read\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=200 \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_Chisco_read_${id}_${trial_id}_d4" &
done


python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=Chisco \
      dataset.task=imagine\
      dataset.id=1 \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[2]}" \
      training.num_train_epochs=200 \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_Chisco_imagine_1_${trial_id}_d4" &

wait

for id in {2..5}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=Chisco \
      dataset.task=imagine\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=200 \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_Chisco_imagine_${id}_${trial_id}_d4" &
done

wait

PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=128

for id in {3..6}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      model.seg_dim=1 \
      dataset=ThingsEEG \
      dataset.task=test\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_ThingsEEG_test_${id}_${trial_id}_d4" &
done

wait

for id in {7..10}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      model.seg_dim=1 \
      dataset=ThingsEEG \
      dataset.task=test\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_ThingsEEG_test_${id}_${trial_id}_d4" &
done

wait

PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=128
NUM_TRAIN_EPOCHS=500

for id in {1..4}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=duin \
      dataset.task=word_classification\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_duin_word_classification_${id}_${trial_id}_d4" &
done

wait

for id in {5..8}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=duin \
      dataset.task=word_classification\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_duin_word_classification_${id}_${trial_id}_d4" &
done

wait

for id in {9..12}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=duin \
      dataset.task=word_classification\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_duin_word_classification_${id}_${trial_id}_d4" &
done

wait

PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=128
NUM_TRAIN_EPOCHS=500

for id in {1..4}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=seed \
      dataset.task=concept_classification\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_seed_${id}_${trial_id}_d4" &
done

wait

for id in {5..8}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=seed \
      dataset.task=concept_classification\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_seed_${id}_${trial_id}_d4" &
done

wait

for id in {9..12}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=seed \
      dataset.task=concept_classification\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_seed_${id}_${trial_id}_d4" &
done

wait

for id in {13..16}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=seed \
      dataset.task=concept_classification\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_seed_${id}_${trial_id}_d4" &
done

wait

for id in {17..20}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=seed \
      dataset.task=concept_classification\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_seed_${id}_${trial_id}_d4" &
done

wait

for id in {0..3}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=faceshouses \
      dataset.task=faceshouses\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_faceshouses_${id}_${trial_id}_d4" &
done

wait

for id in {4..7}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=faceshouses \
      dataset.task=faceshouses\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_faceshouses_${id}_${trial_id}_d4" &
done

wait

for id in {8..11}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=faceshouses \
      dataset.task=faceshouses\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_faceshouses_${id}_${trial_id}_d4" &
done

wait

for id in {12..13}; do
  # 运行训练脚本
  python /root/NeuroSketch/run/train.py \
      model=$MODEL \
      dataset=faceshouses \
      dataset.task=faceshouses\
      dataset.id="$id" \
      training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
      training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
      training.gpu_id="${GPI_ID_LIST[$id%4]}" \
      training.num_train_epochs=$NUM_TRAIN_EPOCHS \
      training.learning_rate=$LEARNING_RATE \
      wandb.exp_name="${MODEL}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_faceshouses_${id}_${trial_id}_d4" &
done