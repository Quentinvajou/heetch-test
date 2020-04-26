

class ModelInteractor:
    def __init__(self, eda=None, preprocessing=None, train_model=None, test_model=None):
        self.preprocessing = preprocessing
        self.eda = eda
        self.train_model = train_model
        self.test_model = test_model

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

    def launch_model_training(self):
        """
        run training
        :return:
        """
        best_model, metrics = self.train_model.run_training()
        self.train_model.save_model_and_metrics(best_model, metrics)
