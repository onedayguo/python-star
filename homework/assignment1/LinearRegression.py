import numpy as np
from .metrics import r2_score


class SimpleLoopLinearRegression1:
    def __init__(self):
        """初始化Simple Linear Regression模型"""
        self.a = None
        self.b = None

    def fit(self, x_train, y_train):
        '''根据训练数据集x_train，y_train训练Simple linear Regression模型'''
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of a_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0  # 分子
        d = 0.0  # 分母
        for x, y in zip(x_train,y_train):
            num += (x-x_mean) * (y-y_mean)
            d += (x-x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        '''给定待预测数据集x_predict,返回表示x_predict的结果向量'''
        assert x_predict.ndim == 1, \
            "Simple Linear Regresson can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        return np.array([self.predict(x) for x in x_predict])

    def _predict(self, x_single):
        '''给定单个带预测数据x_single,返回x_single的预测结果值'''
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleMatrixLinearRegression2:
    def __init__(self):
        """初始化Simple Linear Regression模型"""
        self.a = None
        self.b = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train，y_train训练Simple linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of a_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        # 分子,向量点乘
        num = (x_train - x_mean).dot(y_train-y_mean)
        # 分母，向量点乘
        d = (x_train-x_mean).dot(x_train-x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict,返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regresson can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"
        return np.array([self.predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个带预测数据x_single,返回x_single的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression2()"


class MultiLinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    @staticmethod
    def get_data_from_file():
        # 1*50
        train_x = np.loadtxt("D:/GitHub/python-star/PA-1-data-text/polydata_data_sampx.txt")
        # 50*1
        train_y = np.loadtxt("D:/GitHub/python-star/PA-1-data-text/polydata_data_sampy.txt")
        # 1*100
        test_x = np.loadtxt("D:/GitHub/python-star/PA-1-data-text/polydata_data_polyx.txt")
        # 100*1
        test_y = np.loadtxt("D:/GitHub/python-star/PA-1-data-text/polydata_data_polyy.txt")

        # 统一转换成列向量
        # 50*1
        train_x = train_x.reshape(len(train_x), 1)
        # 50*1
        train_y = train_y.reshape(len(train_y), 1)
        # 100*1
        test_x = test_x.reshape(len(test_x), 1)
        # 100*1
        test_y = test_y.reshape(len(test_y), 1)
        return train_x, train_y, test_x, test_y

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        # np.vstack():在竖直方向上堆叠  np.hstack():在水平方向上平铺
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # X_b原始数据矩阵，但是多加了一列，表示截距，值全为1
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)  # 求导后的参数表达式
        self.intercept_ = self._theta[0]  # 截距第0列
        self.coef_ = self._theta[1:]   # 系数第一列到最后
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])  # 测试值维度调整,首部多加一列 1
        return X_b.dot(self._theta)  # 得到预测值，向量或者矩阵

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度，使用R-square"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"

