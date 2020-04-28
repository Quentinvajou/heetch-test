import os, sys
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from src.infrastructure.settings import logger, continuous
from src.modeling.preprocessing import Preprocessing


class EDA:
    def __init__(self, preprocessing=None):
        self.preprocessing = preprocessing

    def available_local_datasets(self):
        """
        explore data dir to find possible data sets
        :return: list of paths
        """
        all_files = []
        for path, subdirs, files in os.walk('data'):
            for name in files:
                if name.endswith('.log') or name.endswith('.csv'):
                    all_files.append(os.path.join(path, name))
        return all_files

    st.cache
    def load_data(self, dataset_path):
        """
        Load datasets

        :param dataset_path:
        :return: pandas DataFrame
        """
        df = pd.read_csv(dataset_path)
        df = self.preprocessing.timestamp_preprocessing(df)
        return df

    def main_data_explorer(self):
        st.title('Heetch ride booking Case')
        available_dataset = self.available_local_datasets()
        option_dataset = st.selectbox('dataset : ', available_dataset)
        df = self.load_data(option_dataset)
        st.write(df)
        self.explore_per_dataset(option_dataset, df)

    def explore_per_dataset(self, dataset_path, df):
        dataset_name = dataset_path.split('/')[-1]
        if dataset_name == 'bookingRequests.log':
            self.explore_br_dataset(df)
        elif dataset_name == 'rideRequests.log':
            self.explore_rr_dataset(df)
        elif dataset_name == 'drivers.log':
            self.explore_d_dataset(df)
        elif dataset_name.split('.')[0] == 'preprocessed_dataset0':
            self.explore_preprocessed_dataset(df)
        elif dataset_name == 'metrics.csv':
            self.explore_metrics(df)
        else:
            st.write("exploration not prepared for dataset : %s" % dataset_name)
            logger.info("exploration not prepared for dataset : %s" % dataset_name)

    def explore_br_dataset(self, df):
        st.subheader('Number of booking requests by hour')
        hist_values = np.histogram(df['logged_at'].dt.hour, bins=24, range=(0, 24))[0]
        st.bar_chart(hist_values)

        hour_to_filter = st.slider('hour', 0, 23, 21)
        st.subheader('Map of driver positions')
        st.map(df.loc[df['logged_at'].dt.hour == hour_to_filter, ['driver_lon', 'driver_lat']]
               .rename(columns={'driver_lon': 'lon', 'driver_lat': 'lat'}))
        sns.jointplot('driver_lon', 'driver_lat', data=df.loc[df['logged_at'].dt.hour == hour_to_filter, ['driver_lat', 'driver_lon']],
                      kind="hex", color="#4CB391")
        st.pyplot()

        st.subheader('distribution of the target')
        hist_values = np.histogram(df['driver_accepted'], bins=2)[0]
        st.bar_chart(hist_values)


    def explore_rr_dataset(self, df):
        st.subheader('Number of requests by hour')
        # TODO:
        #  - hist from 12 to 12
        hist_values = np.histogram(df['created_at'].dt.hour, bins=24, range=(0, 24))[0]
        st.bar_chart(hist_values)

        hour_to_filter = st.slider('hour', 0, 23, 21)

        st.subheader('Map of pickup points')
        st.map(df.loc[df['created_at'].dt.hour == hour_to_filter, ['origin_lon', 'origin_lat']]
               .rename(columns={'origin_lat': 'lat', 'origin_lon': 'lon'}))

        st.subheader('Map of drop off points')
        st.map(df.loc[df['created_at'].dt.hour == hour_to_filter, ['destination_lon', 'destination_lat']]
               .rename(columns={'destination_lat': 'lat', 'destination_lon': 'lon'}))
        # TODO:
        #  - build deckGL chart with directions of ride
        #  - build histogram by city (origin, dest)

    def explore_d_dataset(self, df):
        st.subheader('Distribution of drivers by hour')
        hist_values = np.histogram(df['logged_at'].dt.hour, bins=24, range=(0, 24))[0]
        st.bar_chart(hist_values)

        st.subheader('Distribution drivers states')
        sns.catplot('new_state', kind='count', data=df)
        st.pyplot()

        # TODO:
        #  - graph timeline drivers (#rides, mean, var..)

    def explore_preprocessed_dataset(self, df):
        st.subheader('distribution of the target')
        hist_values = np.histogram(df['driver_accepted'], bins=2)[0]
        st.bar_chart(hist_values)

        for option_col in continuous:
            g = sns.FacetGrid(df, hue="driver_accepted", margin_titles=True)
            g.fig.set_figwidth(10)
            g = g.map(sns.distplot, option_col)
            g.add_legend()
            plt.show()
            st.pyplot()

        # for option_col in continuous:
        #     sns.boxplot(df[option_col])
        #     st.pyplot()

        st.subheader('Distribution of bookingRequests by hour')
        hist_values = np.histogram(pd.to_datetime(df['logged_at']).dt.hour, bins=24, range=(0, 24))[0]
        st.bar_chart(hist_values)

        st.write("There is  %s drivers id in this sample" % str(df['driver_id'].nunique()))
        st.write("There is  %s bookingRequest id in this sample" % str(df['request_id'].nunique()))
        st.write("There is  %s rideRequest id in this sample" % str(df['ride_id'].nunique()))

    def explore_metrics(self, df):
        available_models = df.loc[:, 'model']
        option_model1 = st.selectbox('compare model : ', available_models)
        option_model2 = st.selectbox('and model : ', available_models)

        string_dict_f = df.loc[df['model'] == option_model1, 'features'].values[0]
        dict_f = eval(string_dict_f)
        string_dict_f2 = df.loc[df['model'] == option_model2, 'features'].values[0]
        dict_f2 = eval(string_dict_f2)
        df_print = pd.DataFrame([dict_f, dict_f2], index=[option_model1, option_model2]).T
        st.table(df_print)


    def main(self):
        eda = EDA()
        eda.preprocessing = Preprocessing()
        dataset_path = 'data/raw/rideRequests.log'
        dataset_path = 'data/raw/bookingRequests.log'
        dataset_path = 'data/raw/drivers.log'
        dataset_path = 'data/trusted/test.csv'
        df = eda.load_data(dataset_path)
