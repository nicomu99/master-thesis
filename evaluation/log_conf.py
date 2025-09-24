import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(name)s:%(funcName)-30s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False    # Remove log handlers from libraries

    return logger
