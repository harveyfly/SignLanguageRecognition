import logging
import os

class Logger(object):
    def __init__(self, log_path):
        if not os.path.exists(log_path):
            os.mkdir('./log')
            with open(log_path, "w") as f:
                pass
        # 设置logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s: - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # 使用FileHandler输出到文件
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # 使用StreamHandler输出到屏幕
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # 添加两个Handler
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

