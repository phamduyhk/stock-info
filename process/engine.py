import pandas as pd
import xgboost as xgb
import numpy as np
import sys
import os
sys.path.append('./../dataProcessing/')
from dataProcessing import GetData
from sklearn.model_selection import train_test_split
import configparser
from os import path
import sys
import json

import matplotlib.pyplot as plt

## for Deep-learing
import keras
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Print dot when training
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".", end="")


class Engine:
    def __init__(self):
        config = configparser.ConfigParser()
        config.optionxform = str  # differ with large and small character
        # Read config
        current_dir = path.dirname(path.abspath(__file__))
        config_file = current_dir + '/engine-config.ini'
        abs_path_config = os.path.abspath(config_file)
        config.read(config_file, encoding="utf-8")

        # hard code
        paths = ['/Users/phamduy/github-res/stock-info/dataProcessing/Data/ext/01_01_2015_31_08_2019']
        instance = GetData()
        self.input_data = instance.read_excel(paths)

    def get_input(self):
        print(self.input_data)
        upcom = self.input_data.get('upcom')
        hose = self.input_data.get('hose')
        hnx = self.input_data.get('hnx')


    def get_index(self):
        return self.input_data.get('index')

    def LSTM(self,length_of_sequence):
        # Model building
        # length_of_sequence = x_train.shape[1]
        in_out_neurons = 1
        n_hidden = 128
        model = Sequential()
        model.add(LSTM(n_hidden, input_shape=(length_of_sequence, in_out_neurons), return_sequences=True))
        model.add(LSTM(n_hidden, return_sequences=True))
        model.add(LSTM(n_hidden, return_sequences=False))
        model.add(Dense(in_out_neurons))
        # model.add(Activation("linear"))
        optimizer = Adam(lr=0.001)
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        model.summary()

        return model

    def fit_model(self,model,x_train,y_train,epochs=100):
        # early stop
        early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=10)
        # load model parameters
        if os.path.isfile("param_goto.hdf5"):
            model.load_weights('param_goto.hdf5')
        # Learning
        history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_split=0.1, verbose=0,
                            callbacks=[early_stop, PrintDot()])

        json_string = model.to_json()
        with open('model_goto.json', 'w') as outfile:
            json.dump(json_string, outfile)
        # save weight
        model.save_weights('param_goto.hdf5')
        return history

    def LSTM_predict(self,x,model):
        predicted = model.predict(x)
        print(predicted[:5])

    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist["epoch"] = history.epoch

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.plot(hist["epoch"], hist["loss"], label="Train loss")
        plt.plot(hist["epoch"], hist["val_loss"], label="Val loss")
        plt.legend()

def create_dataset(df):
    x = []
    y = []
    for i in range(50,df.shape[0]):
        x.append(df[i-50:i,])
        y.append(df[i])
    x = np.array(x)
    y = np.array(y)
    return x,y

if __name__ == '__main__':
    engine = Engine()
    index = engine.get_index()
    vnindex = index.loc[index.TICKER == 'VNINDEX', :]
    print(index)

    # model = LSTM()