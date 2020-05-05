"""
@Author  : vwonx
@Date    : 2020/4/28
"""

import os
import logging
import time
from datetime import datetime


def get_time():
    return datetime.now().strftime('%y-%m-%d %H:%M:%S')


class Timer(object):
    curr_record = None
    prev_record = None

    @classmethod
    def record(cls):
        cls.prev_record = cls.curr_record
        cls.curr_record = time.time()

    @classmethod
    def interval(cls):
        if cls.prev_record is None:
            return 0
        return cls.curr_record - cls.prev_record


def wrap_color(string, color):
    try:
        header = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'darkcyan': '\033[36m',
            'bold': '\033[1m',
            'underline': '\033[4m',
        }[color.lower()]
    except KeyError:
        raise ValueError('Unknow color: {}'.format(color))
    return header + string + '\033[0m'


def info(logger, msg, color=None):
    msg = '[{}]'.format(get_time()) + msg
    if logger is not None:
        logger.info(msg)
    if color is not None:
        msg = wrap_color(msg, color)
    print(msg)


def summary_args(logger, args, color=None):
    keys = [key for key in args.keys() if key[:2] != '__']
    keys.sort()
    length = max([len(key) for key in keys])
    msg = [('{:<' + str(length) + '}: {}').format(key, args[key]) for key in keys]

    msg = '\n' + '\n'.join(msg)
    info(logger, msg, color)


def get_logger(snapshot, model_name):
    if not os.path.exists(snapshot):
        os.makedirs(snapshot)
    logging.basicConfig(filename=os.path.join(snapshot, model_name + '.log'), level=logging.INFO)
    logger = logging.getLogger()
    return logger
