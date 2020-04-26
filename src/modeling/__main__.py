from src.use_case.interactor import ModelInteractor
from src.modeling.preprocessing import Preprocessing
from src.modeling.train_model import TrainModel
from src.modeling.test_model import TestModel
from src.infrastructure.settings import dict_modeling_params

mi = ModelInteractor(preprocessing=Preprocessing(dict_modeling_params), train_model=TrainModel(dict_modeling_params), test_model=TestModel())
mi.train_model.preprocessing = mi.preprocessing
mi.train_model.test_model = mi.test_model
mi.launch_model_training()
