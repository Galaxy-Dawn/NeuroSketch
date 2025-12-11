from typing import List, Dict
import importlib
import importlib.util
import logging
from typing import Union, Tuple
import importlib.metadata
from hydra import compose, initialize
from omegaconf import DictConfig
import importlib
import os
import torch
from collections import OrderedDict

logger = logging.getLogger(__name__)


def yaml_to_dictconfig(config_path: str, config_name: str) -> DictConfig:
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)
    return cfg


def list2dict(input_list: List[str])-> Dict:
    # 创建一个键为索引，值为 input_list[index] 的字典
    index_value_dict = {index: value for index, value in enumerate(input_list)}
    # 创建一个键值对交换的字典
    value_index_dict = {value: index for index, value in enumerate(input_list)}

    # 返回两个字典
    return {
        "index_value_dict"      : index_value_dict,
        "value_index_dict": value_index_dict
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_module(model):
    """
    返回一个有序字典，包含模型中每个子模块的可训练参数数量
    格式：{模块名称: 参数量}
    """
    module_params = OrderedDict()

    # 遍历所有子模块（包含嵌套结构）
    for name, module in model.named_modules():
        # 跳过顶级模块（顶级模块名称为空字符串）
        if not name:
            continue

        # 计算当前模块自身参数的数量（不包括子模块）
        params = 0
        for p in module.parameters(recurse=False):  # 关键：recurse=False只查看当前模块参数
            if p.requires_grad:
                params += p.numel()

        # 只记录有参数的模块
        if params > 0:
            module_params[name] = params

    return module_params


def print_detailed_parameters(model):
    # 获取各模块参数
    params_dict = count_parameters_per_module(model)

    # 计算总量
    total = sum(params_dict.values())

    # 打印表格
    print("{:<20} {:<15} {:<15}".format('Module', 'Params', 'Percentage'))
    print('-' * 45)
    for name, params in params_dict.items():
        percent = 100 * params / total
        print("{:<20} {:<15,} {:>.2f}%".format(name, params, percent))
    print('-' * 45)
    print("{:<20} {:<15,} {:<15}".format('Total', total, '100%'))


def replace_eval_with_test(metrics):
    """
    将字典中的键从 'eval_' 替换为 'test_'。
    :param metrics: 包含评估指标的字典
    :return: 替换后的字典
    """
    new_metrics = {}
    for key, value in metrics.items():
        if key.startswith('eval_'):
            new_key = key.replace('eval_', 'test_')
        else:
            new_key = key
        new_metrics[new_key] = value
    return new_metrics


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    """
    检查一个包是否已经安装，并可选地返回包的版本。
    参数:
        pkg_name (str): 要检查的包名。
        return_version (bool): 如果为 True，返回一个元组 (包是否存在, 版本)。如果为 False，只返回包是否存在的布尔值。

    返回:
        Union[Tuple[bool, str], bool]: 如果 return_version 为 True，返回一个元组 (是否存在, 版本)。
                                        如果 return_version 为 False，仅返回布尔值表示包的存在与否。
    """
    # 使用 find_spec 检查包是否存在，这种方式对包存在性检查非常高效
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"  # 默认版本为 "N/A"

    if package_exists:
        try:
            # 首先尝试使用 importlib.metadata 获取包的版本
            package_version = importlib.metadata.version(pkg_name)
        except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError):
            # 如果无法通过 metadata 获取版本，进行回退处理
            if pkg_name == "torch":
                try:
                    # 针对 "torch" 包的特殊处理，直接访问其 __version__ 属性
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    if "dev" in temp_version:
                        package_version = temp_version  # 如果版本包含 "dev"，保持该版本
                    else:
                        package_exists = False  # 如果不是 "dev" 版本，则认为包不可用
                except ImportError:
                    # 如果无法导入 torch 包，则标记为不可用
                    package_exists = False
            else:
                # 对于其他包，如果无法通过 metadata 获取版本，则标记为不可用
                package_exists = False

        # 如果包存在，记录其版本
        if package_exists:
            logger.debug(f"检测到包 {pkg_name} 的版本: {package_version}")
        else:
            logger.warning(f"包 {pkg_name} 存在，但无法获取其版本信息。")
    else:
        logger.warning(f"包 {pkg_name} 没有安装。")

    # 如果需要返回版本信息，返回 (存在, 版本)
    if return_version:
        return package_exists, package_version
    else:
        # 否则仅返回包是否存在的布尔值
        return package_exists


# 定义函数 import_modules，接收两个参数:
# - models_dir: 存放模型文件的目录
# - namespace: 模块的命名空间，用于构建完整的模块路径
def import_modules(models_dir, namespace):
    # 遍历 models_dir 目录下的所有文件
    for file in os.listdir(models_dir):
        # 获取当前文件的完整路径
        path = os.path.join(models_dir, file)

        # 检查文件是否为有效的 Python 文件
        if (
                not file.startswith("_")  # 文件名不以 "_" 开头（一般表示内部文件）
                and not file.startswith(".")  # 文件名不以 "." 开头（隐藏文件）
                and file.endswith(".py")  # 文件以 ".py" 结尾（Python 源代码文件）
        ):
            # 从文件名中提取模块名（去掉 .py 后缀）
            model_name = file[: file.find(".py")] if file.endswith(".py") else file

            # 动态导入模块，构建完整的模块路径 (namespace + model_name)
            importlib.import_module(namespace + "." + model_name)