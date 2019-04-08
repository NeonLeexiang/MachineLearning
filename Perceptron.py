# import list
from sklearn.model_selection import train_test_split
import numpy as np

"""
    created by neonleexiang
    data : 2019-03-27
    Perceptron.py
"""


"""
    refer to Statistic Learning
    chapter 2
"""


class Perceptron:
    """
        the class of perceptron
    """
    # init def
    def __init__(self, n):
        """"""
        self._w = None
        self._b = None
        self._n = n
        self._data_set = None
        self._is_trained = False

    # ---------------- Data class ----------------
    """ To store the split of the data """
    class Data:
        def __init__(self, train_x, train_y, test_x, test_y):
            self._train_x = train_x
            self._train_y = train_y
            self._test_x = test_x
            self._test_y = test_y

        def train_x(self):
            return self._train_x

        def train_y(self):
            return self._train_y

        def test_x(self):
            return self._test_x

        def test_y(self):
            return self._test_y

    # ---------------- End of Data Class ----------

    # ------------------ input method --------
    def data_split(self, x, y):
        """
            data split
            while using sklearn to split the data into train set and test set
        """
        try:
            _train_x, _train_y, _test_x, _test_y = train_test_split(x, y)
            print("data is successfully input")

            # store the train_set and test_set into the Data
            self._data_set = Perceptron.Data(_train_x, _train_y, _test_x, _test_y)

            return _train_x, _train_y, _test_x, _test_y
        except ValueError:
            print("data input Error")

    @staticmethod
    def loss_function(x, y, w, b):
        return y * (np.sum(w * x) + b)

    def has_negative(self, _train_x, _train_y):
        negative_count = 0
        for _x, _y in zip(_train_x, _train_y):
            if self.loss_function(_x, _y, self._w, self._b) <= 0:
                negative_count += 1
        if negative_count != 0:
            return True
        else:
            return False

    def _train(self, _train_x, _train_y):
        _train_x = np.array(data_set.return_train_x())
        _train_y = np.array(data_set.return_test_y())

        self._w = np.zeros(_train_x.shape[1])
        self._b = np.zeros(1)

        for i in range(1, len(_train_x) + 1):
            while self.has_negative(_train_x[:i], _train_y[:i]):
                for _x, _y in zip(_train_x[:i], _train_y[:i]):
                    if self.loss_function(_x, _y, self._w, self._b) <= 0:
                        self._w = self._w + self._n * _x * _y
                        self._b = self._b + self._n * _y
        self._is_trained = True

    def train(self):
        if self._is_trained:
            print("the Perceptron has been trained")
        else:
            self.data_split(data_x, data_y)
            self._train(self._data_set)
        print("w is", self._w, "b is", self._b)


def statistical_learning_perceptron(data_array, label_array, iter_num=50, n=1):
    """

    :param data_array:  the training data set
    :param label_array:  the label set
    :param iter: the number of iter
    :param n:  the learning rate (gradient step)
    :return:
    """

    def statistical_learning_loss(_x, _y, _w, _b):
        return -1 * _y * (_w * _x.T + _b)

    data_mat = np.mat(data_array)   # change vectors into np matrix
    label_mat = np.mat(label_array).T   # l^T

    row_data_mat, col_data_mat = np.shape(data_mat)   # the row and col of data_mat

    w = np.zeros((1, col_data_mat))  # create the w vector
    b = 0   # init bias=0

    for k in range(iter_num):
        for i in range(row_data_mat):
            _x = data_mat[i]
            _y = label_mat[i]
            if statistical_learning_loss(_x, _y, w, b) >= 0:
                w = w + n * _y * _x
                b = b + n * _y

        print('Round %d:%d training' % (k, iter_num))
    return w, b


if __name__ == '__main__':
    # if you want a test like the code below, you should undo some coding
    """
    test_perceptron = Perceptron(n=1)
    data_x = np.array([[3, 3], [4, 3], [1, 1]])
    data_y = np.array([[1], [1], [-1]])
    test_perceptron._train(data_x, data_y)
    test_perceptron.train()
    """

    data_x = np.array([[3, 3], [4, 3], [1, 1]])
    data_y = np.array([[1], [1], [-1]])

    w, b = statistical_learning_perceptron(data_x, data_y)
    print('the w is %d, the b is %d' % (w, b))
    
