import numpy as np
import pandas as pd

class MakeMeanRegressin:
    def __init__(self):
        self.global_mean_ = None
        self.group_means_ = {}

    def fit(self, X, y):
        self.global_mean_ = y.mean()

        df = pd.DataFrame(
            {'make' : X['make'],
             'price' : y
            })

        group_means = df.groupby('make')['price'].mean().to_dict()

        self.group_means_ = group_means

    def predict(self, X):
        # 1. Подставляем средние цены по маркам
        preds = X['make'].map(self.group_means_)
        
        # 2. Закрываем пропуски глобальным средним
        preds = preds.fillna(self.global_mean_)
        
        # 3. Возвращаем одномерный массив предсказаний
        return preds.to_numpy()

    