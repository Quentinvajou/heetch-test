import os, sys
import logging
import logmatic

ENV = os.getenv('ENV', 'local')
DATASET_SAMPLING_FRACTION = float(os.getenv('DATASET_SAMPLING_FRACTION', '.1'))
TRAINING_TYPE = os.getenv('TRAINING_TYPE', 'naive')

if TRAINING_TYPE == 'naive':
    continuous = ['driver_client_distance', 'ride_distance', 'workshift_duration', 'workshift_rides_count',
                  'workshift_rides_duration']
elif TRAINING_TYPE == 'e':
    continuous = ['driver_client_distance', 'ride_distance', 'workshift_duration', 'workshift_rides_count',
                  'workshift_rides_max','workshift_mean_rides_duration', 'workshift_sum_rides_duration',
                  'workshift_rides_ratio']
else:
    continuous = ['driver_client_distance', 'ride_distance', 'workshift_duration', 'workshift_prefered_duration',
                  'workshift_rides_count', 'workshift_rides_prefered',
                  'workshift_mean_rides_duration', 'workshift_sum_rides_duration',
                  'ride_distance_cumul', 'ride_distance_prefered',
                  'workshift_rides_ratio', 'workshift_duration_ratio', 'distance_ratio',
                  'count_booking_requests_received']

discrete = ['driver_accepted']

columns_timestamp = ['workshift_duration', 'workshift_prefered_duration', 'workshift_mean_rides_duration',
                     'workshift_sum_rides_duration']

look_for_best_params = False

default_parameters = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_weight': 2,
    'n_estimators': 800,
    'gamma': 0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_boost_round': 900
    # 'num_boost_round': 900
}

list_of_parameters = {'nthread': [4],
  'objective':['binary:logistic'],
  'learning_rate': [0.03, 0.05, 0.07],  # so called `eta` value
  'max_depth': [3, 5, 7],
  # 'min_child_weight': [2, 3, 4],
  # 'silent': [1],
  # 'subsample': [0.3, 0.7],
  # 'colsample_bytree': [0.5, 0.7],
  # 'n_estimators': [10, 100, 500, 800],
  # 'num_boost_round' : 900
}

dict_modeling_params = {
    'continuous': continuous,
    'discrete': discrete,
    'default_parameters': default_parameters,
    'list_of_parameters': list_of_parameters,
    'look_for_best_params': look_for_best_params,
    'columns_timestamp': columns_timestamp
}

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
