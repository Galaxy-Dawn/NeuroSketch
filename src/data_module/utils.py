import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

def check_ieeg(ieeg):
    matrix = np.max(ieeg, axis=-1) - np.min(ieeg, axis=-1)
    mask = np.all(matrix != 0, axis=1)
    res = np.where(mask)[0]
    return res.tolist()


def get_split(data_args, ieeg, label=None):
    if label is None:
        label = ieeg
    train_ratio = data_args.train_ratio
    val_ratio = data_args.eval_ratio
    test_ratio = data_args.test_ratio
    stratify_flag = data_args.stratify_flag
    random_seed = data_args.random_seed
    if hasattr(data_args, 'selected_classes'):
        mask = np.isin(label, data_args.selected_classes)
        original_indices = np.arange(len(ieeg))[mask]
    else:
        original_indices = np.arange(len(ieeg))
    if stratify_flag:
        train_indices, temp_indices = train_test_split(
            original_indices,
            test_size=(1 - train_ratio),
            random_state=random_seed,
            shuffle=True,
            stratify=label[mask] if hasattr(data_args, 'selected_classes') else label
        )
        if test_ratio == 0:
            val_indices, test_indices = temp_indices, temp_indices
        else:
            val_size = val_ratio / (val_ratio + test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=(1 - val_size),
                random_state=random_seed,
                shuffle=True,
                stratify=label[temp_indices]
            )
    else:
        train_indices, temp_indices = train_test_split(
            original_indices,
            test_size=(1 - train_ratio),
            random_state=random_seed,
            shuffle=True,
        )
        if test_ratio == 0:
            val_indices, test_indices = temp_indices, temp_indices
        else:
            val_size = val_ratio / (val_ratio + test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=(1 - val_size),
                random_state=random_seed,
                shuffle=True,
            )
    return train_indices, val_indices, test_indices


def get_n_fold_split(data_args, ieeg, label=None):
    """
    先根据 test_ratio 划分出测试集，然后对剩余样本进行 n_fold 抽样（交叉验证分割）。
    参数:
        data_args: 包含参数的对象，必须有以下属性:
            - test_ratio: 测试集比例（0到1之间）。
            - n_fold: n_fold 交叉验证的折数。
            - stratify_flag: 是否采用分层抽样。
            - random_seed: 随机种子。
            - (可选) selected_classes: 如果存在，则只选择这些类别的数据。
        ieeg: 数据集（用于确定样本总数）。
        label: 每个样本的标签（如果为 None，则默认为 ieeg）。
    返回:
        test_indices: 测试集的索引数组。
        fold_splits: 长度为 n_fold 的列表，每个元素为一个 (train_idx, val_idx) 元组，
                     表示在剩余数据中训练集和验证集的划分。
    """
    if label is None:
        label = ieeg

    test_ratio = data_args.test_ratio
    stratify_flag = data_args.stratify_flag
    random_seed = data_args.random_seed

    # 如果设置了 selected_classes，则只选择特定类别的样本
    if hasattr(data_args, 'selected_classes'):
        mask = np.isin(label, data_args.selected_classes)
        original_indices = np.arange(len(ieeg))[mask]
    else:
        original_indices = np.arange(len(ieeg))

    # 先划分测试集
    if test_ratio > 0:
        if stratify_flag:
            # 分层抽样
            stratify_labels = label[mask] if hasattr(data_args, 'selected_classes') else label
            train_val_indices, test_indices = train_test_split(
                original_indices,
                test_size=test_ratio,
                random_state=random_seed,
                shuffle=True,
                stratify=stratify_labels
            )
        else:
            train_val_indices, test_indices = train_test_split(
                original_indices,
                test_size=test_ratio,
                random_state=random_seed,
                shuffle=True
            )
    else:
        # 若 test_ratio 为 0，则所有数据均作为训练/验证数据，测试集索引为全部索引
        train_val_indices = original_indices
        test_indices = original_indices

    # 对剩余的训练+验证数据进行 n_fold 交叉验证分割
    n_fold = data_args.n_fold
    fold_splits = []
    if stratify_flag:
        # 分层KFold需要提供对应的标签
        labels_subset = label[train_val_indices]
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_seed)
        for train_idx, val_idx in skf.split(train_val_indices, labels_subset):
            fold_splits.append((train_val_indices[train_idx], train_val_indices[val_idx]))
    else:
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=random_seed)
        for train_idx, val_idx in kf.split(train_val_indices):
            fold_splits.append((train_val_indices[train_idx], train_val_indices[val_idx]))

    return fold_splits, test_indices


# def get_split(data_args, ieeg, label=None):
#     if label is None:
#         label = ieeg
#     train_ratio = data_args.train_ratio
#     val_ratio = data_args.eval_ratio
#     test_ratio = data_args.test_ratio
#     stratify_flag = data_args.stratify_flag
#     random_seed = data_args.random_seed
#     if hasattr(data_args, 'selected_classes'):
#         mask = np.isin(label, data_args.selected_classes)
#         original_indices = np.arange(len(ieeg))[mask]
#     else:
#         original_indices = np.arange(len(ieeg))
#     if stratify_flag:
#         train_indices = []
#         unique_labels = np.unique(label)
#         for lbl in unique_labels:
#             class_indices = original_indices[label == lbl]
#             if len(class_indices) == 1:
#                 train_indices.append(class_indices[0])
#         remain_indices = np.setdiff1d(original_indices, train_indices)
#         train_indices_2, temp_indices = train_test_split(
#             remain_indices,
#             test_size=(1 - train_ratio),
#             random_state=random_seed,
#             shuffle=True,
#             stratify=label[mask] if hasattr(data_args, 'selected_classes') else label
#         )
#         if train_indices:
#             train_indices = np.concatenate((train_indices, train_indices_2))
#         else:
#             train_indices = train_indices_2
#         if test_ratio == 0:
#             val_indices, test_indices = temp_indices, temp_indices
#         else:
#             val_size = val_ratio / (val_ratio + test_ratio)
#             val_indices, test_indices = train_test_split(
#                 temp_indices,
#                 test_size=(1 - val_size),
#                 random_state=random_seed,
#                 shuffle=True,
#                 stratify=label[temp_indices]
#             )
#     else:
#         train_indices, temp_indices = train_test_split(
#             original_indices,
#             test_size=(1 - train_ratio),
#             random_state=random_seed,
#             shuffle=True,
#         )
#         if test_ratio == 0:
#             val_indices, test_indices = temp_indices, temp_indices
#         else:
#             val_size = val_ratio / (val_ratio + test_ratio)
#             val_indices, test_indices = train_test_split(
#                 temp_indices,
#                 test_size=(1 - val_size),
#                 random_state=random_seed,
#                 shuffle=True,
#             )
#     return train_indices, val_indices, test_indices


import numpy as np
from numpy.lib.stride_tricks import as_strided

def split_into_patches_2d(eeg_data: np.ndarray,
                          patch_len: int,
                          patch_stride: int) -> np.ndarray:
    """
    将 2D EEG 时间序列分块处理
    Args:
        eeg_data:  输入数据，形状 [channel, timestamp]
        patch_len:  每个时间块的长度
        patch_stride: 滑动步长
    Returns:
        patches: 分块后的数据，形状 [channel, num_patches, patch_len]
    """
    # 输入参数校验
    assert len(eeg_data.shape) == 2, "输入应为 2D 张量"
    assert patch_len > 0 and patch_stride > 0, "分块参数必须为正整数"

    channel, timestamp = eeg_data.shape
    assert timestamp >= patch_len, "时间长度必须 ≥ 分块长度"

    # 计算可分块数量 (公式：(L - patch_len)//stride + 1)
    num_patches = (timestamp - patch_len) // patch_stride + 1

    new_shape = (channel, num_patches, patch_len)

    original_stride = eeg_data.strides

    new_strides = (
        original_stride[0],  # channel 维度步幅不变
        original_stride[1] * patch_stride,  # 分块步幅
        original_stride[1]  # 时间点步幅
    )

    patches = as_strided(
        eeg_data,
        shape=new_shape,
        strides=new_strides,
        writeable=False  # 避免意外修改原始数据
    )
    return patches


from numpy.lib.stride_tricks import as_strided

def split_into_patches(trial_eeg: np.ndarray,
                       patch_len: int,
                       patch_stride: int) -> np.ndarray:
    """
    将 EEG 时间序列分块处理
    Args:
        trial_eeg:  输入数据，形状 [bsz, channel, timestamp]
        patch_len:  每个时间块的长度
        patch_stride: 滑动步长
    Returns:
        patches: 分块后的数据，形状 [bsz, channel, num_patches, patch_len]
    """
    # 输入参数校验
    assert len(trial_eeg.shape) == 3, "输入应为 3D 张量"
    assert patch_len > 0 and patch_stride > 0, "分块参数必须为正整数"

    bsz, channel, timestamp = trial_eeg.shape
    assert timestamp >= patch_len, "时间长度必须 ≥ 分块长度"

    # 计算可分块数量 (公式：(L - patch_len)//stride + 1)
    num_patches = (timestamp - patch_len) // patch_stride + 1

    new_shape = (bsz, channel, num_patches, patch_len)

    original_stride = trial_eeg.strides

    new_strides = (
        original_stride[0],  # bsz 维度步幅不变
        original_stride[1],  # channel 维度步幅不变
        original_stride[2] * patch_stride,  # 分块步幅
        original_stride[2]  # 时间点步幅
    )

    patches = as_strided(
        trial_eeg,
        shape=new_shape,
        strides=new_strides,
        writeable=False  # 避免意外修改原始数据
    )
    return patches