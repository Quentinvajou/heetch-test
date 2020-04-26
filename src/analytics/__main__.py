from src.use_case.interactor import ModelInteractor
from src.modeling.preprocessing import Preprocessing
from src.analytics.eda import EDA
from src.infrastructure.settings import dict_modeling_params

mi = ModelInteractor(eda=EDA(), preprocessing=Preprocessing(dict_modeling_params))
mi.eda.preprocessing = mi.preprocessing
mi.stream_eda()
