import joblib
import numpy as np
from tensorflow.keras.models import load_model
from . utils import load_yml
import pandas as pd

class ModelHandler:
    
    def __init__(self):
        self.CONFIG = load_yml('config/config.yml')
        self.xgboost_model = joblib.load(open(self.CONFIG['xgboost_model_path'], "rb"))
        self.ann_model = load_model(self.CONFIG['ann_model_path'])
        self.std_scaler = joblib.load(self.CONFIG['std_scaler_path'])

    def preprocessing(self, data):

        df = pd.DataFrame(data, index=[0])

        return self.std_scaler.transform(df)
    
    def xgb_pred(self, data):
        df = self.preprocessing(data)
        result = self.xgboost_model.predict(df)[0]
        return result
    
    def ann_pred(self, data):
        label_list = ['1-Octanol', '1-Propanol', '2-Butanol', '2-propanol', '1-isobutanol']
        df = self.preprocessing(data)
        print(df)
        result = label_list[np.argmax(self.ann_model.predict(df),axis=1)[0]]
        print(result)
        return result
