import pandas as pd


class Preprocessing:
    def __init__(self):
        self.list_raw_files = ['rideRequests.log', 'bookingRequests.log', 'driver.log']

    def load_and_merge_datasets(self):
        for file_name in self.list_raw_files:
            pass

    def eda_preprocessing(self, df):
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

    def main(self):
        pass