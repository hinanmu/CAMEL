#@Time      :2019/3/29 10:44
#@Author    :zhounan
# @FileName: main.py
#import numpy as cp
import cupy as cp
from sklearn.metrics.pairwise import rbf_kernel
from utils.data import Data
import camel_GPU
import numpy as np
from decimal import Decimal

def train_image():
    datasets = ['yeast', 'scene', 'enron', 'image']
    dataset = datasets[3]
    data = Data(dataset, label_type=0)
    x, y = data.load_data()
    x_train = x[0:1800]
    y_train = y[0:1800]
    x_test = x[1800:2000]
    y_test = y[1800:2000]
    camel_GPU.train_image(dataset, x_train, y_train, x_test, y_test, rho=1, alpha=0.1, alpha_ban=0.5, lam2=0.1)

def train_val():
    # trade-off para
    rho_list = [1]
    alpha_list = cp.arange(0, 1.1, 0.1)
    alpha_ban_list = cp.arange(0, 1.1, 0.1)
    lam2_list = cp.array([0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1])

    datasets = ['yeast', 'scene', 'enron', 'image']
    dataset = datasets[2]
    data = Data(dataset, label_type=0)
    x, y = data.load_data()
    camel_GPU.train_val(dataset, x, y, rho_list, alpha_list, alpha_ban_list, lam2_list)

def train():
    datasets = ['yeast', 'scene', 'enron', 'image']
    dataset = datasets[2]
    data = Data(dataset, label_type=0)
    x, y = data.load_data()

    camel_GPU.train(dataset, x, y, rho=1, alpha=0.1, alpha_ban=0.5, lam2=0.1)

if __name__ == '__main__':
    #train()
    train_val()
    #train_image()


