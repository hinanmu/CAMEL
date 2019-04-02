#@Time      :2019/3/31 16:13
#@Author    :zhounan
# @FileName: camel_GPU.py

import numpy as np
import cupy as cp
import time
from utils.util import path_exists
from utils.data import Data
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from utils.evaluate import evaluate

def ADMM(y_not_j, y_j, rho, alpha):
    """
    ADMM algriothm for label correlation
    :param y_not_j: cupy, dim {n_instances, n_labels - 1}
        train label data which not contain j col
    :param y_j: cupy, dim {n_instances, 1}
        train label data which only contain j col
    :param rho: ADMM augmented lagrangian parameter(not mentioned in the paper)
    :return: cupy, dim {n_labels - 1, 1}
    """
    #w, _ = cp.linalg.eig(y_not_j.T.dot(y_not_j))
    #rho = 1 / (cp.amax(cp.absolute(w)))
    #rho = 0.05
    max_iter = 1000
    AB1 = 1e-4
    AB2 = 1e-2
    n = y_not_j.shape[1]
    
    S_j = cp.zeros(y_not_j.shape[1]).reshape(-1, 1)
    z = cp.zeros_like(S_j)
    l1norm = cp.max(cp.abs(cp.dot(y_j.T, y_not_j))) / 100
    u = cp.zeros_like(S_j)
    I = cp.eye(y_not_j.shape[1])

    for iter in range(max_iter):
        S_j = cp.dot(cp.linalg.inv(cp.dot(y_not_j.T, y_not_j) + rho * I), cp.dot(y_not_j.T, y_j) + rho*(z - u))

        z_old = z
        omega = l1norm / rho
        S_j_hat = alpha * S_j + (1 - alpha) * z_old
        a = S_j_hat + u
        z = cp.maximum(0, a - omega) - cp.maximum(0, -a - omega)
        u = u + S_j_hat - z

        r_norm = cp.linalg.norm(S_j - z)
        s_norm = cp.linalg.norm(-rho * (z - z_old))
        eps_pri = cp.sqrt(n) * AB1 + AB2 * cp.maximum(cp.linalg.norm(S_j), cp.linalg.norm(-z))
        eps_dual = cp.sqrt(n) * AB1 + AB2 * cp.linalg.norm(rho * u)

        if r_norm < eps_pri and s_norm < eps_dual:
            break
    return z

def CAMEL(S, x, y, alpha, lam2):
    """
    get caml parameter
    :param S: cupy, dim {n_labels, n_labels}
        label correlation matrix
    :param x: numpy, dim {n_instances, n_features}
    :param y: numpy, dim {n_instances, n_labels}
    :param alpha: number
        alpha is the tradeoff parameter that controls the collaboration degree
    :param lam2: number
        λ1 and λ2 are the tradeoff parameters determining the relative importance of the above three terms
        λ1 is given in the paper, equal to 1
    :return:
    """
    max_iter = 51
    num_instances = x.shape[0]
    num_labels = y.shape[1]
    y = cp.array(y)

    G = (1-alpha)*cp.eye(num_labels)+ alpha*S #size num_labels * num_labels
    Z = y #size num_instances * num_labels
    lam1 = 1

    sigma = cp.sum(euclidean_distances(x)) / (num_instances * (num_instances - 1))
    gamma = np.array((1 / (2 * sigma * sigma)).tolist())
    K = cp.array(rbf_kernel(x, gamma=gamma)) #size num_instances * num_instances

    H = (1 / lam2) * K + cp.eye(num_instances) #size num_instances * num_instances

    for iter in range(max_iter):
        #b_T is row vector
        H_inv = cp.linalg.inv(H) #size num_instances * num_instances
        b_T = cp.sum(cp.dot(H_inv, Z), axis=0) / cp.sum(H_inv)
        b_T = b_T.reshape(1, -1)#size 1 * num_labels

        A = cp.dot(H_inv, Z - b_T.repeat(num_instances, axis=0)) #size num_instances * num_labels

        T = (1 / lam2) * cp.dot(K, A) + b_T.repeat(num_instances, axis=0) #size num_instances * num_labels
        Z = cp.dot((T + lam1 * cp.dot(y, G.T)), cp.linalg.inv(cp.eye(num_labels) + lam1 * cp.dot(G, G.T)))

    return G, A, gamma, b_T

def predict(G, A, gamma, x, x_test, b_T, lam2):
    num_instances = x.shape[0]
    num_labels = G.shape[0]
    b = b_T.T
    #a col vector
    K_test = cp.array(rbf_kernel(x, x_test, gamma=gamma)) # size n_train * n_test
    temp = cp.dot(K_test.T, A) #(n_test * n_train) * (n_train * n_label) = (n_test * n_label)
    temp = (1 / lam2) * temp.T + b # (n_label * n_test)

    output = cp.dot(temp.T, G) # (n_label * n_test)
    pred = cp.copy(output)
    pred[pred > 0] = 1
    pred[pred < 0] = -1
    return output, pred

#Mean square error
def loss(predcit, y):
    y = cp.array(y)
    loss = cp.sum(cp.square(y-predcit), axis=1)
    loss = cp.mean(loss)
    return loss.tolist()

def evaluate_mean(kf_metrics):
    hammingloss = []
    averge_precision = []
    converge = []
    one_error = []
    ranking_loss = []
    micro_f1 = []
    micro_precision = []
    micro_recall = []
    macro_f1 = []
    macro_precision = []
    macro_recall = []
    for metric in kf_metrics:
        hammingloss.append(metric['hamming_loss'])
        averge_precision.append(metric['average_precision'])
        converge.append(metric['coverage'])
        one_error.append(metric['one_error'])
        ranking_loss.append(metric['ranking_loss'])
        micro_f1.append(metric['micro_f1'])
        micro_precision.append(metric['micro_precision'])
        micro_recall.append(metric['micro_recall'])
        macro_f1.append(metric['macro_f1'])
        macro_precision.append(metric['macro_precision'])
        macro_recall.append(metric['macro_recall'])

    output = 'hammingloss:{:.4f}±{:.4f}\n'.format(np.mean(hammingloss), np.std(hammingloss))
    output += 'averge_precision:{:.4f}±{:.4f}\n'.format(np.mean(averge_precision), np.std(averge_precision))
    output += 'converge:{:.4f}±{:.4f}\n'.format(np.mean(converge), np.std(converge))
    output += 'one_error:{:.4f}±{:.4f}\n'.format(np.mean(one_error), np.std(one_error))
    output += 'ranking_loss:{:.4f}±{:.4f}\n'.format(np.mean(ranking_loss), np.std(ranking_loss))
    output += 'micro_f1:{:.4f}±{:.4f}\n'.format(np.mean(micro_f1), np.std(micro_f1))
    output += 'micro_precision:{:.4f}±{:.4f}\n'.format(np.mean(micro_precision), np.std(micro_precision))
    output += 'micro_recall:{:.4f}±{:.4f}\n'.format(np.mean(micro_recall), np.std(micro_recall))
    output += 'macro_f1:{:.4f}±{:.4f}\n'.format(np.mean(macro_f1), np.std(macro_f1))
    output += 'macro_precision:{:.4f}±{:.4f}\n'.format(np.mean(macro_precision), np.std(macro_precision))
    output += 'macro_recall:{:.4f}±{:.4f}\n'.format(np.mean(macro_recall), np.std(macro_recall))
    print(output)

def train(dataset, x, y, rho, alpha, alpha_ban, lam2):
    num_labels = y.shape[1]
    S = cp.zeros((num_labels, num_labels))
    kf = KFold(n_splits=5, random_state=1)
    kf_metrics = []
    for train_idx, test_idx in kf.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        # get label correlation matrix
        for j in range(num_labels):
            y_j = cp.array(y_train[:, j].reshape(-1, 1))
            y_not_j = cp.array(np.delete(y_train, j, axis=1))
            S_j = ADMM(y_not_j, y_j, rho, alpha_ban)
            S[j, :] = cp.array(np.insert(np.array(S_j.tolist()), j, 0))

        # get caml parameter
        G, A, gamma, b_T = CAMEL(S, x_train, y_train, alpha, lam2)

        # evalue
        test_output, test_predict = predict(G, A, gamma, x_train, x_test, b_T, lam2)
        y_test[y_test==-1] = 0
        test_predict[test_predict==-1] = 0
        metric = evaluate(y_test, np.array(test_output.tolist()), np.array(test_predict.tolist()))
        kf_metrics.append(metric)
    evaluate_mean(kf_metrics)

def train_image(dataset, x_train, y_train, x_test, y_test, rho, alpha, alpha_ban, lam2):
    train_log_path = '../train_log/' + dataset + '/'
    log_name = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())) + '.log'
    num_labels = y_train.shape[1]
    S = cp.zeros((num_labels, num_labels))

    # get label correlation matrix
    for j in range(num_labels):
        y_j = cp.array(y_train[:, j].reshape(-1, 1))
        y_not_j = cp.array(np.delete(y_train, j, axis=1))
        S_j = ADMM(y_not_j, y_j, rho, alpha_ban)
        S[:, j] = cp.array(np.insert(np.array(S_j.tolist()), j, 0))

    # get caml parameter
    G, A, gamma, b_T = CAMEL(S, x_train, y_train, alpha, lam2)

    # evalue
    output, test_predict = predict(G, A, gamma, x_train, x_test, b_T, lam2)
    test_predict[test_predict==-1] = 0
    y_test[y_test==-1] = 0
    metrics = evaluate(y_test, np.array(output.tolist()), np.array(test_predict.tolist()))
    print(metrics)

def train_val(dataset, x, y, rho_list, alpha_list, alpha_ban_list, lam2_list):
    train_log_path = '../train_log/' + dataset + '/'
    log_name = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())) + '.log'

    num_labels = y.shape[1]
    S = cp.zeros((num_labels, num_labels))
    kf = KFold(n_splits=5, random_state=1)

    step = 0
    best_val_loss = 1e6
    all_paras = []
    print('dataset:', dataset)
    for rho in rho_list:
        for alpha in alpha_list:
            for alpha_ban in alpha_ban_list:
                for lam2 in lam2_list:
                    output = 'step:{}, time:{}, CAME rho:{}, alpha:{}, alpha_ban:{},lam2:{}'.format(step, time.strftime('%H:%M:%S', time.localtime(time.time())), rho, alpha, alpha_ban, lam2)
                    output_ = output
                    print(output)

                    fold = 0
                    train_loss_list = []
                    val_loss_list = []
                    test_loss_list = []
                    for train_idx, test_idx in kf.split(x):
                        x_trainval = x[train_idx]
                        y_trainval = y[train_idx]
                        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2,
                                                                          random_state=0)

                        x_test = x[test_idx]
                        y_test = y[test_idx]

                        # get label correlation matrix
                        for j in range(num_labels):
                            y_j = cp.array(y_train[:, j].reshape(-1, 1))
                            y_not_j = cp.array(np.delete(y_train, j, axis=1))
                            S_j = ADMM(y_not_j, y_j, rho, alpha_ban)
                            S[j, :] = cp.array(np.insert(np.array(S_j.tolist()), j, 0))

                        # get caml parameter
                        G, A, gamma, b_T = CAMEL(S, x_train, y_train, alpha, lam2)

                        # evalue
                        _, train_predict = predict(G, A, gamma, x_train, x_train, b_T, lam2)
                        _, val_predict = predict(G, A, gamma, x_train, x_val, b_T, lam2)
                        _, test_predict = predict(G, A, gamma, x_train, x_test, b_T, lam2)
                        train_loss = loss(train_predict, y_train)
                        val_loss = loss(val_predict, y_val)
                        test_loss = loss(test_predict, y_test)

                        train_loss_list.append(train_loss)
                        val_loss_list.append(val_loss)
                        test_loss_list.append(test_loss)

                        output = 'time:{}, Kflod {}, train loss:{}, val loss:{}, test loss:{}'.format(
                            time.strftime('%H:%M:%S', time.localtime(time.time())), fold, train_loss, val_loss, test_loss)
                        output_ = output_ + '\n' + output
                        print(output)
                        fold += 1

                    mean_train_loss, mean_val_loss, mean_test_loss = cp.mean(cp.array(train_loss_list)), cp.mean(
                        cp.array(val_loss_list)), cp.mean(cp.array(test_loss_list))
                    output = 'mean train loss:{}, val loss:{}, test loss:{}'.format(mean_train_loss, mean_val_loss,
                                                                                    mean_test_loss)
                    output_ = output_ + '\n' + output
                    print(output)

                    paras = (mean_val_loss, mean_test_loss, mean_train_loss, rho, alpha, alpha_ban, lam2)
                    all_paras.append(paras)
                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        output = 'current best val loss para, rho:{}, alpha:{},  alpha_ban:{}, lam2:{}'.format(rho, alpha, alpha_ban, lam2)
                        output_ = output_ + '\n' + output
                        print(output)

                    step += 1
                    output_ = output_ + '\n\n'
                    print()

                    path_exists(train_log_path)
                    with open(train_log_path + log_name, 'a+') as f:
                        f.write(output_)

    with open(train_log_path + log_name, 'a+') as f:
        all_paras.sort()
        f.write('all_paras\n')
        for paras in all_paras:
            f.write('val loss: {} '.format(paras[0]))
            f.write('test loss: {} '.format(paras[1]))
            f.write('train loss: {}'.format(paras[2]))
            f.write('rho: {} '.format(paras[3]))
            f.write('alpha: {} '.format(paras[4]))
            f.write('alpha_ban: {} '.format(paras[5]))
            f.write('lam2: {}\n'.format(paras[6]))
