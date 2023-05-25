import numpy as np

import torch

torch.manual_seed(99)
np.random.seed(99)
torch.cuda.empty_cache()


import numpy as np


from enum import Enum


import pandas as pd

import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader


class Tasks(Enum):
    prediction = "prediction"
    reconstruction = "reconstruction"


class TimeSeriesDataset(object):
    def __init__(self, task: Tasks, data_path: str, seq_length: int, batch_size: int, prediction_window: int = 1):
        """
        :param task: name of the task
        :param data_path: path to datafile
        :param seq_length: window length to use
        :param batch_size:
        :param prediction_window: window length to predict
        """
        self.task = task.value

        # data_path = pkg_resources.resource_filename("tsa", data_path)
        self.data = np.load(data_path)

        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.batch_size = batch_size


    def preprocess_data(self):
        """Preprocessing function"""

        X_train, X_test = train_test_split(self.data, train_size=0.8, shuffle=False)

        return X_train, X_test

    def frame_series(self, X, y=None):
        """
        Function used to prepare the data for time series prediction
        :param X: set of features
        :param y: targeted value to predict
        :return: TensorDataset
        """
        nb_obs, nb_features = X.shape
        features, target = [], []

        for i in range(0, nb_obs - self.seq_length - self.prediction_window):
            features.append(torch.FloatTensor(X[i:i + self.seq_length, :]).unsqueeze(0))

            # # lagged output used for prediction
            # y_hist.append(torch.FloatTensor(y[i:i + self.seq_length,:]).unsqueeze(0))
            # shifted target
            target.append(
                torch.FloatTensor(X[i + self.seq_length:i + self.prediction_window + self.seq_length,:]).unsqueeze(0))

        features_var = torch.cat(features)
        target_var = torch.cat(target)

        return TensorDataset(features_var, target_var)

    def get_loaders(self):
        """
        Preprocess and frame the dataset

        :return: DataLoaders associated to training and testing data
        """
        X_train, X_test = self.preprocess_data()
        nb_features = X_train.shape[1]
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        # y_train = np.array(y_train).reshape(-1,1)
        # y_test = np.array(y_train).reshape(-1,1)

        train_dataset = self.frame_series(X_train)
        test_dataset = self.frame_series(X_test)

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_iter = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return train_iter, test_iter, nb_features

    # def invert_scale(self, predictions):
    #     """
    #     Inverts the scale of the predictions
    #     """
    #     if isinstance(predictions, torch.Tensor):
    #         predictions = predictions.numpy()
    #
    #     if predictions.ndim == 1:
    #         predictions = predictions.reshape(-1, 1)
    #
    #     if self.task == "prediction":
    #         unscaled = self.y_scaler.inverse_transform(predictions)
    #     else:
    #         unscaled = self.preprocessor.named_transformers_["scaler"].inverse_transform(predictions)
    #     return torch.Tensor(unscaled)