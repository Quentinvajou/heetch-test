import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import math
import datetime
from copy import deepcopy

from src.infrastructure.settings import logger, DATASET_SAMPLING_FRACTION

tqdm.pandas()
cores = cpu_count() #Number of CPU cores on your system
partitions = cores #Define as many partitions as you want



class Preprocessing:
    def __init__(self, dict_modeling_params):
        self.list_raw_files = ['rideRequests.log', 'bookingRequests.log', 'driver.log']
        self.continuous = dict_modeling_params['continuous']
        self.discrete = dict_modeling_params['discrete']
        self.df_d = None
        self.df_m = None
        self.columns_timestamp = dict_modeling_params['columns_timestamp']

    def load_and_merge_datasets(self, frac=.1):
        if os.path.isfile('data/trusted/preprocessed_dataset'+str(frac)+'.csv')\
                and os.access('data/trusted/preprocessed_dataset'+str(frac)+'.csv', os.R_OK):
            is_preprocessed = True
            df_m = pd.read_csv('data/trusted/preprocessed_dataset'+str(frac)+'.csv')
        else:
            is_preprocessed = False
            df_rr = pd.read_csv('data/raw/rideRequests.log')
            df_br = pd.read_csv('data/raw/bookingRequests.log')

            df_m = df_br.merge(df_rr, how='inner', left_on=['ride_id'], right_on=['ride_id'])
            df_m = self.timestamp_preprocessing(df_m)
            df_m = df_m.sample(frac=frac)

        self.df_d = pd.read_csv('data/raw/drivers.log')
        self.df_d = self.timestamp_preprocessing(self.df_d)
        self.df_d = self.df_d.sort_values('logged_at')

        logger.info("datasets loaded")

        return df_m, is_preprocessed

    def timestamp_preprocessing(self, df):
        """
        naive preprocessing only modifies data to make it readable
        :param df:
        :return:
        """
        if 'logged_at' in df.columns:
            try:
                df['logged_at'] = pd.to_datetime(df['logged_at'], unit='s')
            except ValueError as e:
                pass
        if 'created_at' in df.columns:
            try:
                df['created_at'] = pd.to_datetime(df['created_at'], unit='s')
            except Exception as e:
                pass
        return df

    def feature_engineering(self, df):
        logger.info("feature engineering started")

        # df = self.parallelize(df, self.feature_driver_availability)
        df['driver_client_distance'] = df.apply(lambda row: self.distance_between_point(
            (row['origin_lat'], row['origin_lon']), (row['driver_lat'], row['driver_lon'])), axis=1)
        # This features weakens model
        # df['driver_destination_distance'] = df.apply(lambda row: self.distance_between_point(
        #     (row['destination_lat'], row['destination_lon']), (row['driver_lat'], row['driver_lon'])), axis=1)
        df['ride_distance'] = df.apply(lambda row: self.distance_between_point(
            (row['origin_lat'], row['origin_lon']), (row['destination_lat'], row['destination_lon'])), axis=1)

        # necessary for distance features extraction
        self.df_m = deepcopy(df)

        df = self.parallelize(df, self.workshift_state_extraction)

        df['workshift_rides_count'], df['workshift_rides_prefered'],\
            df['workshift_mean_rides_duration'], df['workshift_sum_rides_duration'] =\
            zip(*df[['list_rides_time', 'list_all_rides_time']].apply(
                lambda row: self.features_workshift_state(row), axis=1))

        df = self.convert_and_format_dataset(df)

        df = self.parallelize(df, self.feature_ride_distance)
        # df = self.feature_ride_distance(df)

        df['workshift_rides_ratio'] = df[['workshift_rides_count', 'workshift_rides_prefered']].apply(
            lambda row: row[0]/row[1], axis=1)
        df['workshift_duration_ratio'] = df[['workshift_duration', 'workshift_prefered_duration']].apply(
            lambda row: row[0]/row[1], axis=1)
        df['distance_ratio'] = df[['ride_distance_cumul', 'ride_distance_prefered']].apply(
            lambda row: row[0]/row[1], axis=1)

        logger.info("feature engineering ended")
        return df

    def workshift_state_extraction(self, df):
        df['workshift_duration'], df['workshift_prefered_duration'],\
            df['list_rides_time'], df['list_all_rides_time'] = \
            zip(*df[['logged_at', 'driver_id']].progress_apply(lambda row: self.workshift_state(row), axis=1))
        return df

    def workshift_state(self, row):
        booking_request_time = row['logged_at']
        df_driver = self.df_d.loc[self.df_d['driver_id'] == row['driver_id'], :]
        duration_workshift = booking_request_time - df_driver.iloc[0, -2]
        duration_prefered_workshift = df_driver.iloc[-1, -2] - df_driver.iloc[0, -2]
        list_rides_time = []
        list_all_rides_time = []
        for index, row_d in df_driver.iterrows():
            if row_d['logged_at'] <= booking_request_time:
                if row_d['new_state'] == 'began_ride':
                    begin = row_d['logged_at']
                elif row_d['new_state'] == 'ended_ride':
                    end = row_d['logged_at']
                    list_rides_time.append(end-begin)
                    list_all_rides_time.append(end-begin)
            else:
                if row_d['new_state'] == 'began_ride':
                    begin = row_d['logged_at']
                elif row_d['new_state'] == 'ended_ride':
                    end = row_d['logged_at']
                    list_all_rides_time.append(end-begin)
        return duration_workshift, duration_prefered_workshift, list_rides_time, list_all_rides_time

    def features_workshift_state(self, row):
        list_rides_time = row['list_rides_time']
        list_all_rides_time = row['list_all_rides_time']
        count_rides = len(list_rides_time)
        count_all_rides = len(list_all_rides_time)
        if count_rides > 0:
            mean_ride_duration = np.mean(list_rides_time)
            sum_ride_duration = np.sum(list_rides_time)
        else:
            mean_ride_duration = datetime.timedelta()
            sum_ride_duration = datetime.timedelta()
        return count_rides, count_all_rides, mean_ride_duration, sum_ride_duration

    def feature_ride_distance(self, df):
        df['ride_distance_cumul'], df['ride_distance_prefered'], df['count_booking_requests_received'] = \
            zip(*df[['logged_at', 'driver_id']].progress_apply(
                lambda row: self.ride_distances(row), axis=1))
        return df

    def ride_distances(self, row):
        distance_prefered = self.df_m.loc[(self.df_m['driver_accepted'] == True) &
                                          (self.df_m['driver_id'] == row['driver_id']), 'ride_distance'].sum()
        distance_cumul = self.df_m.loc[(self.df_m['logged_at'] < row['logged_at']) &
                                       (self.df_m['driver_accepted'] == True) &
                                       (self.df_m['driver_id'] == row['driver_id']), 'ride_distance'].sum()
        count_booking_requests_received = self.df_m.loc[(self.df_m['logged_at'] == row['logged_at']) &
                                                        (self.df_m['driver_id'] == row['driver_id']), 'request_id'].count()

        return distance_cumul, distance_prefered, count_booking_requests_received

    def feature_driver_availability(self, df):
        df['is_driver_available'], df['driver_next_state'] = zip(*df[['logged_at', 'driver_id']].progress_apply(lambda row: self.is_driver_available(row), axis=1))
        return df

    def is_driver_available(self, row):
        booking_request_time = row['logged_at']
        df_driver = self.df_d.loc[self.df_d['driver_id'] == row['driver_id'], :]
        lt = df_driver['logged_at'].tolist()
        lt.append(booking_request_time)
        lt_sorted = sorted(lt)
        index_in_list = lt_sorted.index(booking_request_time)
        driver_next_state = 'err'
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
        return is_driver_available, driver_next_state

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
        for col in self.columns_timestamp:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x.seconds/60)
        return df

    def subset_by_iqr_range(self, df, column, whisker_width=1.5):
        ''' remove outliers from continuous serie in DataFrame
        :param df:
            dataframe wit all data
        :param column:
            serie within dataframe to scan for outliers
        :param whisker_width:
            distance from the quartiles that separate inliers from outliers
        :return:
            complete dataframe with rows of outliers removed
        :rtype:dataframe
        '''
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        # Apply filter with respect to IQR, including optional whiskers
        # filter = (df[column] >= q1 - whisker_width * iqr) & (df[column] <= q3 + whisker_width * iqr)
        filter = (df[column] <= q3 + whisker_width * iqr)
        return df.loc[filter]

    def preprocessing_rules(self, df):
        '''
        processing rules is defined to apply specific business rules. they're not part of preprocessing since
        they don't necessarly come from data exploration and modeling improvement.
        :param df:
        :return:
        '''
        return df

    def prepare_dataset_for_training(self, df, save_processed_dataset=False):
        df = self.feature_engineering(df)
        df = self.preprocessing_rules(df)
        if save_processed_dataset:
            df.to_csv('data/trusted/preprocessed_dataset'+str(DATASET_SAMPLING_FRACTION)+'.csv', index=False)
        return df

    def filter_dataset_for_training_column_wise(self, df):
        df_con = df.loc[:, self.continuous]
        df_cat = df.loc[:, self.discrete]
        df_cat = pd.get_dummies(df_cat)
        df = pd.concat([df_con, df_cat], axis=1)
        return df

    def filter_dataset_for_training_row_wise(self, df):
        for iqr_col in self.continuous:
            df = self.subset_by_iqr_range(df, iqr_col)
        return df

    def main(self):
        self = Preprocessing(dict_modeling_params)
        df, _ = self.load_and_merge_datasets(frac=.1)
        # df = pp.prepare_dataset_for_training(df)
        # driver_id = df.loc[12000, 'driver_id']
        driver_id = df.loc[3400, 'driver_id']
        df_temp = df.loc[df['driver_id'] == driver_id, :]
        df_temp = df_temp.sort_values(['logged_at'])
        df_driver = self.df_d.loc[self.df_d['driver_id'] == driver_id, :]

