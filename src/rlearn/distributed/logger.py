import logging


def get_logger(name):
    logger = logging.getLogger(name)
    logger.propagate = False
    sh = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(filename)s:%(lineno)d - %(levelname)s | %(message)s")
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
