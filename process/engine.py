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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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


    def LSTM(self):

        pass



if __name__ == '__main__':
    engine = Engine()