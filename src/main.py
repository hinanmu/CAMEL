#@Time      :2019/3/29 10:44
#@Author    :zhounan
# @FileName: main.py
#import numpy as np
import numpy as np
import time
from utils.data import Data
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold, train_test_split

#ADMM algriothm for label correlation
def ADMM(y_not_j, y_j, rho=0):
    #w, _ = np.linalg.eig(y_not_j.T.dot(y_not_j))
    #rho = 1 / (np.amax(np.absolute(w)))
    rho = 1
    #rho = 0.05

    S_j = np.zeros(y_not_j.shape[1]).reshape(-1, 1)
    z = np.zeros_like(S_j)
    l1norm = np.max(np.abs(np.dot(y_j.T, y_not_j))) / 100
    u = np.zeros_like(S_j)
    I = np.eye(y_not_j.shape[1])

    for iter in range(max_iter):
        S_j = np.dot(np.linalg.inv(np.dot(y_not_j.T, y_not_j) + rho * I), np.dot(y_not_j.T, y_j) + rho*(z - u))

        omega = l1norm / rho
        a = S_j + u
        z = np.maximum(0, a - omega) - np.maximum(0, -a - omega)

        u = u + S_j - z

    return S_j

def CAMEL(S, x, y, alpha, lam2):
    num_instances = x.shape[0]
    num_labels = y.shape[1]

    G = (1-alpha)*np.eye(num_labels)+ alpha*S #size num_labels * num_labels
    Z = y #size num_instances * num_labels
    lam1 = 1

    sigma = np.mean(np.log(rbf_kernel(x, gamma=-1)))
    gamma = 1 / (2 * sigma * sigma)
    K = rbf_kernel(x, gamma=gamma) #size num_instances * num_instances

    H = (1 / lam2) * K + np.eye(num_instances) #size num_instances * num_instances

    for iter in range(max_iter):
        #b_T is row vector
        H_inv = np.linalg.inv(H) #size num_instances * num_instances
        b_T = np.sum(np.dot(H_inv, Z), axis=0) / np.sum(H_inv)
        b_T = b_T.reshape(1, -1)#size 1 * num_labels

        A = np.dot(H_inv, Z - b_T.repeat(num_instances, axis=0)) #size num_instances * num_labels

        T = (1 / lam2) * np.dot(K, A) + b_T.repeat(num_instances, axis=0) #size num_instances * num_labels
        Z = np.dot((T + lam1 * np.dot(y, G.T)), np.linalg.inv(np.eye(num_labels) + lam1 * np.dot(G, G.T)))

    return G, A, gamma, b_T

def predict(G, A, gamma, x, x_test, b_T):
    num_instances = x.shape[0]
    num_labels = G.shape[0]
    b = b_T.T
    #a col vector
    K_test = rbf_kernel(x, x_test, gamma=gamma) # size n_train * n_test
    temp = np.dot(K_test.T, A) #(n_test * n_train) * (n_train * n_label) = (n_test * n_label)
    temp = temp.T # (n_label * n_test)

    logits = np.dot(G.T, temp + b) # (n_label * n_test)
    logits[logits > 0] = 1
    logits[logits < 0] = -1
    logits = logits.T #(n_test * n_label)
    return logits

#Mean square error
def loss(predcit, y):
    loss = np.sum(np.square(y-predcit), axis=1)
    loss = np.mean(loss)
    return loss

if __name__ == '__main__':
    data = Data('yeast', label_type=0)
    x, y = data.load_data()
    num_labels = y.shape[1]
    S = np.zeros((num_labels, num_labels))
    max_iter = 1000
    kf = KFold(n_splits=5, random_state=1)

    #trade-off para
    rho_list = [1]
    alpha_list = np.arange(0, 1, 0.1)
    lam2_list = np.array([0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1])

    for rho in rho_list:
        for alpha in alpha_list:
            for lam2 in lam2_list:
                print('time{}, CAME rho:{}, alpha:{}, lam2:{}'.format(time.strftime('%H:%M:%S', time.localtime(time.time())), rho, alpha, lam2))

                fold = 0
                for train_idx, test_idx in kf.split(x):
                    x_trainval = x[train_idx]
                    y_trainval = y[train_idx]
                    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2,
                                                                      random_state=0)

                    x_test = x[test_idx]
                    y_test = y[test_idx]

                    # get label correlation matrix
                    for j in range(num_labels):
                        y_j = y_train[:, j].reshape(-1, 1)
                        y_not_j = np.delete(y_train, j, axis=1)
                        S_j = ADMM(y_not_j, y_j, rho)
                        S[:, j] = np.insert(S_j, j, 0)

                    # get caml parameter
                    G, A, gamma, b_T = CAMEL(S, x_train, y_train, alpha, lam2)

                    #evalue
                    train_predict = predict(G, A, gamma, x_train, x_train, b_T)
                    val_predict = predict(G, A, gamma, x_train, x_val, b_T)
                    test_predict = predict(G, A, gamma, x_train, x_test, b_T)
                    train_loss = loss(train_predict, y_train)
                    val_loss = loss(val_predict, y_val)
                    test_loss = loss(test_predict, y_test)

                    print('time{}, Kflod {}, train loss:{}, val loss:{}, test loss:{}'.format(time.strftime('%H:%M:%S', time.localtime(time.time())), fold, train_loss, val_loss, test_loss))
                    fold += 1

                print()






