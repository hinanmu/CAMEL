#@Time      :2019/3/31 16:13
#@Author    :zhounan
# @FileName: camel.py
import numpy as np
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
    :param y_not_j: numpy, dim {n_instances, n_labels - 1}
        train label data which not contain j col
    :param y_j: numpy, dim {n_instances, 1}
        train label data which only contain j col
    :param rho: ADMM augmented lagrangian parameter(not mentioned in the paper)
    :return: numpy, dim {n_labels - 1, 1}
    """
    #w, _ = np.linalg.eig(y_not_j.T.dot(y_not_j))
    #rho = 1 / (np.amax(np.absolute(w)))
    #rho = 0.05
    max_iter = 1000
    AB1 = 1e-4
    AB2 = 1e-2
    n = y_not_j.shape[1]

    S_j = np.zeros(y_not_j.shape[1]).reshape(-1, 1)
    z = np.zeros_like(S_j)
    l1norm = np.max(np.abs(np.dot(y_j.T, y_not_j))) / 100
    u = np.zeros_like(S_j)
    I = np.eye(y_not_j.shape[1])

    for iter in range(max_iter):
        S_j = np.dot(np.linalg.inv(np.dot(y_not_j.T, y_not_j) + rho * I), np.dot(y_not_j.T, y_j) + rho*(z - u))

        z_old = z
        omega = l1norm / rho
        S_j_hat = alpha * S_j + (1 - alpha) * z_old
        a = S_j_hat + u
        z = np.maximum(0, a - omega) - np.maximum(0, -a - omega)
        u = u + S_j_hat - z

        r_norm = np.linalg.norm(S_j-z)
        s_norm = np.linalg.norm(-rho*(z - z_old))
        eps_pri = np.sqrt(n) * AB1 + AB2 * np.maximum(np.linalg.norm(S_j), np.linalg.norm(-z))
        eps_dual = np.sqrt(n) * AB1 + AB2 * np.linalg.norm(rho*u)

        if r_norm < eps_pri and s_norm < eps_dual:
            break
    return z

def CAMEL(S, x, y, alpha, lam2):
    """
    get caml parameter
    :param S: numpy, dim {n_labels, n_labels}
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

    G = (1-alpha)*np.eye(num_labels)+ alpha*S #size num_labels * num_labels
    Z = y #size num_instances * num_labels
    lam1 = 1

    sigma = np.sum(euclidean_distances(x)) / (num_instances*(num_instances-1))
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

def predict(G, A, gamma, x, x_test, b_T, lam2):
    num_instances = x.shape[0]
    num_labels = G.shape[0]
    b = b_T.T
    #a col vector
    K_test = rbf_kernel(x, x_test, gamma=gamma) # size n_train * n_test
    temp = np.dot(K_test.T, A) #(n_test * n_train) * (n_train * n_label) = (n_test * n_label)
    temp = (1 / lam2) * temp.T + b # (n_label * n_test)

    output = np.dot(temp.T, G) # (n_label * n_test)
    pred = np.copy(output)
    pred[pred > 0] = 1
    pred[pred < 0] = -1
    return output, pred

#Mean square error
def loss(predcit, y):
    loss = np.sum(np.square(y-predcit), axis=1)
    loss = np.mean(loss)
    return loss

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
    S = np.zeros((num_labels, num_labels))
    kf = KFold(n_splits=5, random_state=1)
    kf_metrics = []
    for train_idx, test_idx in kf.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        # get label correlation matrix
        for j in range(num_labels):
            y_j = np.array(y_train[:, j].reshape(-1, 1))
            y_not_j = np.array(np.delete(y_train, j, axis=1))
            S_j = ADMM(y_not_j, y_j, rho, alpha_ban)
            S[j, :] = np.array(np.insert(np.array(S_j.tolist()), j, 0))

        # get caml parameter
        G, A, gamma, b_T = CAMEL(S, x_train, y_train, alpha, lam2)

        # evalue
        test_output, test_predict = predict(G, A, gamma, x_train, x_test, b_T, lam2)
        y_test[y_test == -1] = 0
        test_predict[test_predict == -1] = 0
        metric = evaluate(y_test, test_output, test_predict)
        kf_metrics.append(metric)
    evaluate_mean(kf_metrics)
    
def train_image(dataset, x_train, y_train, x_test, y_test, rho, alpha, alpha_ban, lam2):
    train_log_path = '../train_log/' + dataset + '/'
    log_name = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())) + '.log'
    num_labels = y_train.shape[1]
    S = np.zeros((num_labels, num_labels))

    # get label correlation matrix
    for j in range(num_labels):
        y_j = np.array(y_train[:, j].reshape(-1, 1))
        y_not_j = np.array(np.delete(y_train, j, axis=1))
        S_j = ADMM(y_not_j, y_j, rho, alpha_ban)
        S[j, :] = np.array(np.insert(np.array(S_j.tolist()), j, 0)) #col

    # get caml parameter
    G, A, gamma, b_T = CAMEL(S, x_train, y_train, alpha, lam2)

    # evalue
    output, test_predict = predict(G, A, gamma, x_train, x_test, b_T, lam2)
    test_predict[test_predict==-1] = 0
    y_test[y_test==-1] = 0
    metrics = evaluate(y_test, output, test_predict)
    print(metrics)
    
def train_val(dataset, x, y, rho_list, alpha_list, alpha_ban_list, lam2_list):
    train_log_path = '../train_log/' + dataset + '/'
    log_name = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())) + '.log'
    num_labels = y.shape[1]
    S = np.zeros((num_labels, num_labels))
    kf = KFold(n_splits=5, random_state=1)

    step = 0
    best_val_loss = 1e6
    print('dataset:', dataset)
    all_paras = []
    for rho in rho_list:
        for alpha in alpha_list:
            for alpha_ban in alpha_ban_list:
                for lam2 in lam2_list:
                    output = 'step:{}, time:{}, CAME rho:{}, alpha:{}, alpha_ban:{},lam2:{}'.format(step, time.strftime(
                        '%H:%M:%S', time.localtime(time.time())), rho, alpha, alpha_ban, lam2)
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
                            y_j = y_train[:, j].reshape(-1, 1)
                            y_not_j = np.delete(y_train, j, axis=1)
                            S_j = ADMM(y_not_j, y_j, rho, alpha)
                            S[j, :] = np.insert(S_j, j, 0)

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

                    mean_train_loss, mean_val_loss, mean_test_loss = np.mean(train_loss_list), np.mean(
                        val_loss_list), np.mean(test_loss_list)
                    output = 'mean train loss:{}, val loss:{}, test loss:{}'.format(mean_train_loss, mean_val_loss,
                                                                                    mean_test_loss)
                    output_ = output_ + '\n' + output
                    print(output)

                    paras = (mean_val_loss, mean_test_loss, mean_train_loss, rho, alpha, lam2)
                    all_paras.append(paras)
                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        output = 'current best val loss para, rho:{}, alpha:{},  alpha_ban:{}, lam2:{}'.format(rho, alpha,
                                                                                                               alpha_ban,
                                                                                                               lam2)
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