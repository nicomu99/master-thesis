import logging

# https://stackoverflow.com/questions/46008038/padding-multiple-fields-together-in-python-logger
factory = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
    record = factory(*args, **kwargs)
    class_name = record.name.split(".")[-1]
    record.origin = f'{class_name}:{record.funcName}'
    return record


# def get_logger(name: str) -> logging.Logger:
root_log = logging.getLogger()
root_log.setLevel(logging.CRITICAL)


logging.setLogRecordFactory(record_factory)

for handler in list(root_log.handlers):
    root_log.removeHandler(handler)

if not root_log.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s  %(levelname)-8s %(origin)-35s %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch.setFormatter(formatter)
    root_log.addHandler(ch)
    root_log.propagate = False

#    return logger
