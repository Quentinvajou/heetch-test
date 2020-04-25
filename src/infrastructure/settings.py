import os, sys
import logging
import logmatic


def setup_logger_stdout(name, level=logging.INFO, additional_logger=[], removed_logger=[]):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    # formatter = logging.Formatter()
    formatter = logmatic.JsonFormatter(extra={"env":os.getenv('RZC_ENV', 'local')})

    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger_file = logging.getLogger('logger_file')
    logger_file.propagate = False

    for n in additional_logger:
        logger_n = logging.getLogger(n)
        logger_n.addHandler(handler)
        logger_n.propagate = False
    for m in removed_logger:
        logger_m = logging.getLogger(m)
        logger_m.propagate = False

    logging.basicConfig(level=level, handlers=[handler])
    return logger

LOGS_LEVEL = os.getenv('LOGS_LEVEL', 'info').lower()
log_level_from_env = LOGS_LEVEL
log_level = logging.INFO

if log_level_from_env == 'debug':
    log_level = logging.DEBUG
elif log_level_from_env == 'info':
    log_level = logging.INFO
elif log_level_from_env == 'warning':
    log_level = logging.WARNING
elif log_level_from_env == 'error':
    log_level = logging.ERROR

logger = setup_logger_stdout(name='logger', level=log_level, removed_logger=[])
