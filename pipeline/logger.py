import logging


def setup_logger(name):
    logging.basicConfig(level="INFO")

    logger = logging.getLogger(name)

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler("pipeline.log")

    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
