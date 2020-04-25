

class ModelInteractor:
    def __init__(self, eda=None, preprocessing=None):
        self.preprocessing = preprocessing
        self.eda = eda

    def build_data_set(self):
        """
        check if necessary data is there, otherwise download it
        :return:
        """
        pass

    def stream_eda(self):
        """
        opens streamlit UI for extraction & data analysis
        :return:
        """
        self.eda.main_data_explorer()

    def train_model(self):
        pass

