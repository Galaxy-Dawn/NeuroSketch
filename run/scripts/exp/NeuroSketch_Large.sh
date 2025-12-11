#!/bin/bash

# 设置训练参数
MODEL="NeuroSketch_Large"
PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=128
GPI_ID_LIST=(4 5 6 7)
NUM_TRAIN_EPOCHS=500
LEARNING_RATE=1e-3
fold_id_list=(0 1 2)
dataset_list=(OpenMIIR faceshouses duin seed ThingsEEG Chisco)
declare -A task_dict
task_dict["OpenMIIR"]="perception imagination"
task_dict["ThingsEEG"]="test"
task_dict["Chisco"]="read imagine"
task_dict["duin"]="word_classification"
task_dict["seed"]="concept_classification"
task_dict["faceshouses"]="faceshouses"

declare -A subject_num_dict
subject_num_dict["OpenMIIR"]=9
subject_num_dict["ThingsEEG"]=10
subject_num_dict["Chisco"]=5
subject_num_dict["duin"]=12
subject_num_dict["seed"]=20
subject_num_dict["faceshouses"]=14

declare -A dataset_epochs
dataset_epochs["Chisco"]=100

echo "开始生成参数组合..."
param_combinations=()

# 生成所有参数组合
for dataset in "${dataset_list[@]}"; do
    tasks=${task_dict[$dataset]}
    subject_num=${subject_num_dict[$dataset]}
    # 获取训练轮数（默认500）
    epochs=${dataset_epochs[$dataset]:-$NUM_TRAIN_EPOCHS}
    # 遍历任务
    for task in $tasks; do
        # 遍历受试者
        for ((subject_id=1; subject_id<=subject_num; subject_id++)); do
            # 遍历fold
            for fold_id in "${fold_id_list[@]}"; do
                param_combinations+=("$dataset $task $subject_id $fold_id $epochs")
            done
        done
    done
done

total_jobs=${#param_combinations[@]}
echo "总共生成 $total_jobs 个任务组合"

if [ "$total_jobs" -eq 0 ]; then
    echo "没有有效的参数组合，跳过"
    exit 1
fi

# 计算 GPU 相关参数
total_gpus=${#GPI_ID_LIST[@]}
batch_size=$((total_gpus*1))
current_job=0
total_batches=$(( (total_jobs + batch_size - 1) / batch_size ))

echo "GPU配置: ${total_gpus}个GPU (${GPI_ID_LIST[*]})"
echo "批处理大小: $batch_size"
echo "总批次数: $total_batches"
echo "=================================================="

# 逐批次执行任务
for ((batch=0; batch<total_batches; batch++)); do
    start=$((batch * batch_size))
    end=$((start + batch_size))
    if [ $end -gt $total_jobs ]; then
        end=$total_jobs
    fi

    echo "=================================================="
    echo "准备执行批次 $((batch+1))/$total_batches"
    echo "任务范围: $((start+1))-$end / $total_jobs"
    echo "剩余任务: $((total_jobs - current_job))"
    echo "=================================================="

    # 生成当前批次的GPU命令
    commands=()
    gpu_ids=()

    for ((i=start; i<end && i<total_jobs; i++)); do
        IFS=' ' read -r dataset task subject_id fold_id epochs <<< "${param_combinations[$i]}"
        gpu_index=$((i % total_gpus))
        gpu_id=${GPI_ID_LIST[$gpu_index]}

        # 显示当前任务详情
        echo "任务 $((i+1))/$total_jobs: 数据集=$dataset, 任务=$task, 受试者=$subject_id, Fold=$fold_id, 轮数=$epochs, GPU=$gpu_id"

        commands+=("CUDA_VISIBLE_DEVICES=$gpu_id python /root/NeuralSketch/run/train.py \
                    model=${MODEL} \
                    dataset=${dataset} \
                    dataset.task=${task} \
                    dataset.id=${subject_id} \
                    dataset.test_fold_id=${fold_id} \
                    training.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
                    training.per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
                    training.gpu_id=${gpu_id} \
                    training.num_train_epochs=${epochs} \
                    training.learning_rate=${LEARNING_RATE} \
                    wandb.exp_name=${MODEL}_${LEARNING_RATE}_${epochs}_${dataset}_${task}_${subject_id}_${fold_id}")
        gpu_ids+=($gpu_id)
    done

    echo "占用GPU: ${gpu_ids[*]}"
    echo "启动批次 $((batch+1)) 的 $((end-start)) 个任务..."

    # 启动当前批次的训练任务
    for cmd in "${commands[@]}"; do
        eval "$cmd &"
        ((current_job++))
    done

    echo "批次 $((batch+1)) 已启动，等待完成..."
    # 等待当前批次完成
    wait

    echo "批次 $((batch+1)) 已完成 | 累计完成 $current_job/$total_jobs"
    echo ""
done

echo "=================================================="
echo "🎉 全部任务完成！总执行组合数: $total_jobs"
echo "=================================================="
