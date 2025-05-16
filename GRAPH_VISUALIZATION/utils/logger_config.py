import logging
import os
from logging.handlers import RotatingFileHandler
import time


class SafeRotatingFileHandler(RotatingFileHandler):
    """A RotatingFileHandler subclass that handles file access errors during rotation"""

    def rotate(self, source, dest):
        """Rotate the file with error handling and retries"""
        for i in range(5):  # Try 5 times
            try:
                # If dest exists, try to delete it first
                if os.path.exists(dest):
                    try:
                        os.remove(dest)
                    except:
                        pass
                # Try to rename the file
                os.rename(source, dest)
                break
            except Exception as e:
                if i < 4:  # Don't sleep on the last attempt
                    time.sleep(0.1)  # Wait 100ms before retrying
                continue
        else:
            # If all retries failed, try to continue without rotation
            try:
                # Try to just truncate the file instead
                with open(source, 'w'):
                    pass
            except:
                pass


def setup_logger():
    """
    配置日志系统
    - 同时输出到控制台和文件
    - 日志文件位于 logs 目录下
    - 使用循环日志文件，每个文件最大 1MB，保留 5 个备份
    - 使用安全的文件轮转处理
    """
    # 创建 logs 目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件路径
    log_file = os.path.join(log_dir, 'graph_visualization.log')

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建并配置文件处理器，使用安全的轮转处理器
    try:
        file_handler = SafeRotatingFileHandler(
            log_file,
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5,
            encoding='utf-8'
        )
    except Exception as e:
        # 如果创建轮转处理器失败，尝试使用基本的文件处理器
        file_handler = logging.FileHandler(
            log_file,
            encoding='utf-8'
        )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # 创建并配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 移除所有现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 配置 graph_manager 的日志记录器
    graph_logger = logging.getLogger('GRAPH_VISUALIZATION')
    graph_logger.setLevel(logging.DEBUG)

    return graph_logger
