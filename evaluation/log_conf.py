import logging

# https://stackoverflow.com/questions/46008038/padding-multiple-fields-together-in-python-logger
factory = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
    record = factory(*args, **kwargs)
    class_name = record.name.split(".")[-1]
    record.origin = f'{class_name}:{record.funcName}'
    return record


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    logging.setLogRecordFactory(record_factory)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s  %(levelname)-8s %(origin)-30s %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False    # Remove log handlers from libraries

    return logger