import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import math

from src.infrastructure.settings import logger

tqdm.pandas()
cores = cpu_count() #Number of CPU cores on your system
partitions = cores #Define as many partitions as you want



class Preprocessing:
    def __init__(self, dict_modeling_params):
        self.list_raw_files = ['rideRequests.log', 'bookingRequests.log', 'driver.log']
        self.continuous = dict_modeling_params['continuous']
        self.discrete = dict_modeling_params['discrete']
        self.df_d = None

    def load_and_merge_datasets(self):
        if os.path.isfile('data/trusted/preprocessed_dataset.csv') and os.access('data/trusted/preprocessed_dataset.csv', os.R_OK):
            is_preprocessed = True
            df_m = pd.read_csv('data/trusted/preprocessed_dataset.csv')
        else:
            is_preprocessed = False
            df_rr = pd.read_csv('data/raw/rideRequests.log')
            df_br = pd.read_csv('data/raw/bookingRequests.log')

            df_m = df_br.merge(df_rr, how='inner', left_on=['ride_id'], right_on=['ride_id'])
            df_m = self.timestamp_preprocessing(df_m)

            # df_t = df_m.loc[df_m['driver_accepted'] == True].sample(500)
            # df_f = df_m.loc[df_m['driver_accepted'] == False].sample(500)
            # df_m = pd.concat([df_t, df_f], axis=0)

        self.df_d = pd.read_csv('data/raw/drivers.log')
        self.df_d = self.timestamp_preprocessing(self.df_d)
        self.df_d = self.df_d.sort_values('logged_at')
        logger.info("datasets loaded")

        return df_m.sample(frac=1), is_preprocessed

    def timestamp_preprocessing(self, df):
        """
        naive preprocessing only modifies data to make it readable
        :param df:
        :return:
        """
        if 'logged_at' in df.columns:
            df['logged_at'] = pd.to_datetime(df['logged_at'], unit='s')
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], unit='s')
        return df

    def feature_engineering(self, df):
        logger.info("feature engineering started")

        # df = self.parallelize(df, self.feature_driver_availability)
        df['driver_client_distance'] = df.apply(lambda row: self.distance_between_point(
            (row['origin_lat'], row['origin_lon']), (row['driver_lat'], row['driver_lon'])), axis=1)
        df['ride_distance'] = df.apply(lambda row: self.distance_between_point(
            (row['origin_lat'], row['origin_lon']), (row['destination_lat'], row['destination_lon'])), axis=1)

        df = self.parallelize(df, self.features_workshift_state)

        logger.info("feature engineering ended")
        return df

    def features_workshift_state(self, df):
        df['workshift_duration'], df['workshift_rides_count'], df['workshift_rides_duration'] = \
            zip(*df[['logged_at', 'driver_id']].progress_apply(lambda row: self.workshift_state(row), axis=1))
        return df

    def workshift_state(self, row):
        booking_request_time = row['logged_at']
        df_driver = self.df_d.loc[self.df_d['driver_id'] == row['driver_id'], :]
        duration_workshift = booking_request_time - df_driver.iloc[0, -2]
        list_rides_time = []
        for index, row in df_driver.iterrows():
            if row['new_state'] == 'began_ride':
                begin = row['logged_at']
            elif row['new_state'] == 'ended_ride':
                end = row['logged_at']
                list_rides_time.append(end-begin)
        count_ride = len(list_rides_time)
        mean_ride_duration = np.mean(list_rides_time)
        return duration_workshift, count_ride, mean_ride_duration

    def feature_driver_availability(self, df):
        df['is_driver_available'] = df[['logged_at', 'driver_id']].progress_apply(lambda row: self.is_driver_available(row), axis=1)
        return df

    def is_driver_available(self, row):
        booking_request_time = row['logged_at']
        df_driver = self.df_d.loc[self.df_d['driver_id'] == row['driver_id'], :]
        lt = df_driver['logged_at'].tolist()
        lt.append(booking_request_time)
        lt_sorted = sorted(lt)
        index_in_list = lt_sorted.index(booking_request_time)
        if index_in_list == 0:
            is_driver_available = False
        elif index_in_list == len(lt) - 1:
            is_driver_available = False
        else:
            driver_next_state = df_driver['new_state'].iloc[index_in_list]
            if driver_next_state == "connected":
                is_driver_available = False
            elif driver_next_state == "began_ride":
                is_driver_available = True
            elif driver_next_state == "ended_ride":
                is_driver_available = False
            elif driver_next_state == "disconnected":
                is_driver_available = True
        return is_driver_available

    def distance_between_point(self, origin, destination):
        lat1, lon1 = origin
        lat2, lon2 = destination
        radius = 6371  # km

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = radius * c
        return d

    def parallelize(self, df, func):
        data_split = np.array_split(df, partitions)
        pool = Pool(cores)
        data = pd.concat(pool.map(func, data_split))
        pool.close()
        pool.join()
        return data

    def convert_and_format_dataset(self, df):
        if 'workshift_duration' in df.columns:
            df['workshift_duration'] = df['workshift_duration'].apply(lambda x: x.seconds/60)
        if 'workshift_rides_duration' in df.columns:
            df['workshift_rides_duration'] = df['workshift_rides_duration'].apply(lambda x: x.seconds/60)
        return df

    def preprocessing_rules(self, df):
        return df

    def prepare_dataset_for_training(self, df, save_processed_dataset=False):
        df = self.feature_engineering(df)
        df = self.convert_and_format_dataset(df)
        df = self.preprocessing_rules(df)
        if save_processed_dataset:
            df.to_csv('data/trusted/preprocessed_dataset.csv')
        return df

    def filter_dataset_for_training(self, df):
        df_con = df.loc[:, self.continuous]
        df_cat = df.loc[:, self.discrete]
        df_cat = pd.get_dummies(df_cat)
        df = pd.concat([df_con, df_cat], axis=1)
        return df

    def main(self):
        pp = Preprocessing()
        df = pp.load_and_merge_datasets()
        df = pp.feature_engineering(df)
        # df_temp = pp.df_d.loc[pp.df_d['driver_id']==df.loc[9713, 'driver_id']]
