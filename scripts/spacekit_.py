import os
import zipfile
import numpy as np
import spacekit
from spacekit.transformer import Transformer
from spacekit.builder import Builder

class Prep:
    def __init__(self):
        HOME = os.path.curdir
        DATA = os.path.abspath(HOME+'/data/')
        self.HOME = HOME
        self.DATA = DATA
        T = Transformer()
        self.T = T

    def unzip(self, train_zip, test_zip):
        DATA = self.DATA

        with zipfile.ZipFile(DATA+train_zip, 'r') as zip_ref:
            zip_ref.extractall(self.DATA)
        with zipfile.ZipFile(DATA+test_zip, 'r') as zip_ref:
            zip_ref.extractall(self.DATA)
        print('Data Extraction Successful')
        return os.listdir(DATA)

    def split_data(self, train, test):
        DATA = self.DATA
        T = self.T
        print('Train-Test Split Successful')
        X_train, X_test, y_train, y_test = T.hypersonic_pliers(DATA+train, DATA+test)
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        T = self.T
        print('Data Scaled to Zero Mean and Unit Variance')
        X_train, X_test = T.thermo_fusion_chisel(X_train, X_test)
        return X_train, X_test

    def add_filter(self, X_train, X_test):
        T = self.T
        print('Noise filter added!')
        X_train, X_test = T.babel_fish_dispenser(X_train, X_test)
        return X_train, X_test

class Launch:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        builder = Builder(X_train, X_test, y_train, y_test)
        self.builder = builder

    def deploy(self):
        builder = self.builder
        cnn = builder.build_cnn()
        return cnn

    def takeoff(self, cnn):
        builder = self.builder
        history = builder.fit_cnn(cnn)

if __name__ == '__main__':
    import spacekit
    prep = Prep()
    prep.unzip('/exoTrain.csv.zip', '/exoTest.csv.zip')
    X_train, X_test, y_train, y_test = prep.split_data('/exoTrain.csv', '/exoTest.csv')
    X_train, X_test = prep.scale_data(X_train, X_test)
    X_train, X_test = prep.add_filter(X_train, X_test)

    launch = Launch(X_train, X_test,  y_train, y_test)
    cnn = launch.deploy()
    history = launch.takeoff(cnn)
