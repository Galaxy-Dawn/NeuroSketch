import logging
import sys
from contextlib import contextmanager
import time
import os
import psutil
import math

# 尝试导入 colorama，用于控制台输出彩色日志
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_INSTALLED = True
except ImportError:
    COLORAMA_INSTALLED = False

# 自定义文件处理器，继承自 logging.FileHandler
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # 每次记录后立即刷新

# 设置日志记录函数
def setup_logging(log_file=None, level=logging.INFO):
    # 定义新的日志级别 INFO_HIGH_LEVEL，比 INFO 更重要
    INFO_HIGH_LEVEL = 25
    logging.addLevelName(INFO_HIGH_LEVEL, "INFO_HIGH")

    # 定义一个新的日志记录方法
    def info_high(self, message, *args, **kwargs):
        if self.isEnabledFor(INFO_HIGH_LEVEL):
            self._log(INFO_HIGH_LEVEL, message, args, **kwargs)

    logging.Logger.info_high = info_high  # 给 Logger 类添加 info_high 方法

    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)  # 设置全局日志级别

    # 移除默认的处理器，避免重复日志输出
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 如果 colorama 安装了，使用带颜色的格式化器
    if COLORAMA_INSTALLED:
        class ColorFormatter(logging.Formatter):
            LEVEL_COLOR = {
                logging.DEBUG: Style.DIM + Fore.WHITE,
                logging.INFO: Fore.GREEN,
                INFO_HIGH_LEVEL: Fore.CYAN,  # INFO_HIGH 的颜色
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Style.BRIGHT + Fore.RED
            }

            def format(self, record):
                log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
                formatter = logging.Formatter(log_fmt)
                level_color = self.LEVEL_COLOR.get(record.levelno, Fore.WHITE)
                # 添加颜色到日志级别和消息中
                record.levelname = level_color + record.levelname + Style.RESET_ALL
                record.msg = level_color + record.getMessage() + Style.RESET_ALL
                return formatter.format(record)

        color_formatter = ColorFormatter()
        console_handler.setFormatter(color_formatter)  # 控制台使用颜色格式化器
    else:
        cprint("colorama 未安装，将使用普通格式化器")
        console_handler.setFormatter(formatter)  # 否则使用普通格式化器

    # 如果 log_file 是 None 或者为空字符串，则跳过文件处理器的创建
    if log_file and log_file != '':
        # 创建自定义文件处理器，输出到文件并确保每次记录后刷新
        file_handler = FlushFileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        # 添加文件处理器到日志记录器
        logger.addHandler(file_handler)

    # 添加控制台处理器到日志记录器
    logger.addHandler(console_handler)

    return logger  # 返回已配置好的 logger 对象


def cprint(text, color='WHITE', background=None, style="BOLD"):
    """
    彩色输出文本的自定义 print 函数
    :param text: 要打印的文本
    :param color: 文本的前景色（默认为 WHITE）
    :param background: 背景色（可选）
    :param style: 样式（如 BOLD, DIM， 可选）
    """
    if not COLORAMA_INSTALLED:
        # 如果没有安装 colorama，直接输出
        print(text)
        return

    # 定义前景色和背景色
    color_map = {
        'BLACK': Fore.BLACK, 'RED': Fore.RED, 'GREEN': Fore.GREEN, 'YELLOW': Fore.YELLOW,
        'BLUE': Fore.BLUE, 'MAGENTA': Fore.MAGENTA, 'CYAN': Fore.CYAN, 'WHITE': Fore.WHITE,
    }

    background_map = {
        'BLACK': Back.BLACK, 'RED': Back.RED, 'GREEN': Back.GREEN, 'YELLOW': Back.YELLOW,
        'BLUE': Back.BLUE, 'MAGENTA': Back.MAGENTA, 'CYAN': Back.CYAN, 'WHITE': Back.WHITE,
    }

    style_map = {
        'DIM': Style.DIM, 'NORMAL': Style.NORMAL, 'BOLD': Style.BRIGHT
    }

    # 获取指定颜色、背景色、样式的对应值
    text_color = color_map.get(color.upper(), Fore.WHITE)
    bg_color = background_map.get(background.upper(), '') if background else ''
    text_style = style_map.get(style.upper(), Style.NORMAL) if style else ''

    # 组合并输出
    output = f"{text_color}{bg_color}{text_style}{text}{Style.RESET_ALL}"
    print(output)

@contextmanager
def tracking(block_name, logger):
    logger.info(f"开始执行代码块: {block_name}")
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0 ** 30

    try:
        yield
        logger.info(f"代码块 {block_name} 执行成功")
    except Exception as e:
        logger.error(f"代码块 {block_name} 执行失败，异常: {e}")
        raise
    finally:
        end_time = time.time()
        elapsed_time = end_time - t0
        m1 = p.memory_info().rss / 2.0 ** 30
        delta = m1 - m0
        sign = "+" if delta >= 0 else "-"
        delta = math.fabs(delta)

        logger.info(f"代码块 {block_name} 内存使用: [{m1:.1f}GB({sign}{delta:.1f}GB)]")
        logger.info(f"代码块 {block_name} 执行时间: {elapsed_time:.4f} 秒")


# 测试 logging 配置
if __name__ == '__main__':
    logger = setup_logging(level=logging.DEBUG)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.info_high("This is an info_high message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")