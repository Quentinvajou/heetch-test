from src.use_case.interactor import ModelInteractor
from src.modeling.preprocessing import Preprocessing
from src.analytics.eda import EDA

mi = ModelInteractor(eda=EDA(), preprocessing=Preprocessing())
mi.eda.preprocessing = mi.preprocessing
mi.stream_eda()
