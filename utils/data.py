#@Time      :2019/2/21 15:21
#@Author    :zhounan
# @FileName: data.py
import numpy as np
import random
from sklearn.decomposition import PCA
class Data():
    def __init__(self, dataset_name, label_type, path = '../dataset/'):
        """

        :param dataset_name:
        :param label_type: if equal to 0 represent y = {-1,1} if equal to 1 represent y = {0,1}
        :param path:
        """
        self.path = path
        self.dataset_name = dataset_name
        self.label_type = label_type
        self.x, self.y, self.x_train, self.y_train, self.x_test, self.y_test = \
            None, None, None, None, None, None

    def load_data(self):
        """load numpy data x and y
        ----------
        Returns
        -------
        numpy size (n_samples, n_features)
        numpy size (n_samples, n_labels), value {-1,1}
        """
        path = self.path
        dataset_name = self.dataset_name
        x = np.load(path + dataset_name + '/x.npy')
        y = np.load(path + dataset_name + '/y.npy')

        if self.label_type == 0:
            y[y == 0] = -1  # 0 to -1
        if self.label_type == 1:
            pass
        return x, y

    def load_data_separated(self):
        """load numpy data x_train, x_test and y_train, y_test
        ----------
        Returns
        -------
        numpy size (n_train_samples, n_features)
            train x data
        numpy size (n_train_samples, n_labels), value {-1,1}
            trian y data
        numpy size (n_test_samples, n_features)
            test x data
        numpy size (n_test_samples, n_labels), value {-1,1}
            test y data
        """
        path = self.path
        dataset_name = self.dataset_name
        x_train = np.load(path + dataset_name + '/x_train.npy')
        y_train = np.load(path + dataset_name + '/y_train.npy')
        x_test = np.load(path + dataset_name + '/x_test.npy')
        y_test = np.load(path + dataset_name + '/y_test.npy')

        if self.label_type == 0:
            y_train[y_train == 0] = -1  # 0 to -1
            y_test[y_test == 0] = -1  # 0 to -1
        if self.label_type == 1:
            pass
        return x_train, y_train, x_test, y_test

    def adjust_data(self, X, y):
        """
        for bpmll_exp_main method 5
        :param X:
        :param y:
        :return:
        """
        [rows, cols] = y.shape
        min = cols
        for i in range(rows - 1):
            reverse = i + 1
            for j in range(i + 1, rows):
                sum = np.sum(y[i] + y[j])
                if sum < min:
                    min = sum
                    reverse = j
            X[[i + 1, reverse], :] = X[[reverse, i + 1], :]
            y[[i + 1, reverse], :] = y[[reverse, i + 1], :]
        return X, y

    def adjust_data(self, X, y, batch_size):
        """
        for bpmll_exp_main method 6
        调整y的数据集，依次选取与y某一行i相反量最多的行让这一行排列在i的下一行
        :param X: numpy
            feature data
        :param y: numpy {-1,1}
            label data
        :return: numpy, numpy
        """
        [rows, cols] = y.shape
        min = cols
        for i in range(rows - 1):
            reverse = i + 1
            if (i+1 // batch_size) % 2 == 0:
                for j in range(i + 1, rows):
                    sum = np.sum(y[i] - y[j])

                    if sum < min:
                        min = sum
                        reverse = j

                X[[i + 1, reverse], :] = X[[reverse, i + 1], :]
                y[[i + 1, reverse], :] = y[[reverse, i + 1], :]
            else:
                for j in range(i + 1, rows):
                    sum = np.sum(y[i-batch_size-1] + y[j])
                    if sum < min:
                        min = sum
                        reverse = j

                X[[i + 1, reverse], :] = X[[reverse, i + 1], :]
                y[[i + 1, reverse], :] = y[[reverse, i + 1], :]
        return X, y

    def change_dimension(self, X):
        """
        change x dimension for cnn network, n_features to n_features*n_features
        :param X:
        :return:
        """
        num_features = X.shape[1]
        num_instances = X.shape[0]
        grid = [[(x + y) % num_features for x in range(num_features)] for y in range(num_features)]

        n = num_features
        random.seed(0)
        for x in range(n - 1):
            swapRow = random.randrange(x + 1, n)
            for i in range(n):
                temp = grid[x][i]
                grid[x][i] = grid[swapRow][i]
                grid[swapRow][i] = temp

        random.seed(1)
        # Randomize col
        for x in range(n - 1):
            swapCol = random.randrange(x + 1, n)
            for i in range(n):
                temp = grid[i][x]
                grid[i][x] = grid[i][swapCol]
                grid[i][swapCol] = temp

        temp = np.zeros(shape=(num_instances, num_features, num_features))
        for i, val in enumerate(grid):
            for j in range(num_instances):
                temp[j, i] = X[:, val][j]

        return temp[:,:,:, np.newaxis]

    def change_dimension_1(self, X, n_components = 32):
        """
        change x dimension for cnn network, n_features to 32 ,and then to 32*32
        :param X:
        :return:
        """
        estimator = PCA(n_components=n_components)
        X = estimator.fit_transform(X)
        num_features = X.shape[1]
        num_instances = X.shape[0]
        grid = [[(x + y) % num_features for x in range(num_features)] for y in range(num_features)]

        n = num_features
        random.seed(0)
        for x in range(n - 1):
            swapRow = random.randrange(x + 1, n)
            for i in range(n):
                temp = grid[x][i]
                grid[x][i] = grid[swapRow][i]
                grid[swapRow][i] = temp

        random.seed(1)
        # Randomize col
        for x in range(n - 1):
            swapCol = random.randrange(x + 1, n)
            for i in range(n):
                temp = grid[i][x]
                grid[i][x] = grid[i][swapCol]
                grid[i][swapCol] = temp

        temp = np.zeros(shape=(num_instances, num_features, num_features))
        for i, val in enumerate(grid):
            for j in range(num_instances):
                temp[j, i] = X[:, val][j]

        return temp[:,:,:, np.newaxis]

    def change_dimension_2(self, X, n_components=1024):
        """
        change x dimension for cnn network, n_features to n_features*n_features ,and then to 32*32
        :param X:
        :return:
        """
        num_features = X.shape[1]
        num_instances = X.shape[0]
        grid = [[(x + y) % num_features for x in range(num_features)] for y in range(num_features)]

        n = num_features
        random.seed(0)
        for x in range(n - 1):
            swapRow = random.randrange(x + 1, n)
            for i in range(n):
                temp = grid[x][i]
                grid[x][i] = grid[swapRow][i]
                grid[swapRow][i] = temp

        random.seed(1)
        # Randomize col
        for x in range(n - 1):
            swapCol = random.randrange(x + 1, n)
            for i in range(n):
                temp = grid[i][x]
                grid[i][x] = grid[i][swapCol]
                grid[i][swapCol] = temp

        temp = np.zeros(shape=(num_instances, num_features, num_features))
        for i, val in enumerate(grid):
            for j in range(num_instances):
                temp[j, i] = X[:, val][j]

        print(temp.shape)
        temp = temp.reshape(-1, num_features*num_features)
        print(temp.shape)
        estimator = PCA(n_components=n_components)
        temp = estimator.fit_transform(temp)
        print(temp.shape)
        temp = temp.reshape(num_instances, 32, 32)

        return temp[:,:,:, np.newaxis]

    # def change_dimension_3(self, X, ):



# data = Data('yeast')
# # x = np.array([[1,-1,1,1],[-1,1,-1,-1],[1,-1,1,1],[-1,1,-1,-1]])
# # y = np.array([[-1,-1,1,1],[-1,1,-1,1],[1,1,1,1],[-1,1,-1,-1]])
# # x,y = data.adjust_data(x,y, batch_size=2)
# # print(y)