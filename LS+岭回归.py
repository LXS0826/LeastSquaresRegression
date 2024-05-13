# encoding:utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
import scipy.io as scio


# 加载数据函数
def load_data(path):
    data = scio.loadmat(path)
    data_x = np.array(data['X'])
    data_y = np.array(data['Y'][:, 0] - 1)
    train_data, test_data, train_label, test_label = train_test_split(data_x, data_y, test_size=0.25, random_state=10)
    return train_data, test_data, train_label, test_label


# 定义 Least Squares (LS) 算法
class LeastSquaresRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加 x0 = 1 到每个实例
        self.theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)


if __name__ == '__main__':
    datasets = ['MNIST.mat', 'lung.mat', 'Yale.mat']
    for dataset in datasets:
        print(dataset[:-4] + ':')
        # 加载数据集
        train_data, test_data, train_label, test_label = load_data('datasets/' + dataset)

        # 使用 LS 算法找到最佳参数
        ls = LeastSquaresRegression()
        ls.fit(train_data, train_label)

        # 使用岭回归分类器进行分类
        ridge_classifier = RidgeClassifier(alpha=0.2)  # 设置 alpha 参数作为正则化强度
        ridge_classifier.coef_ = ls.theta[1:].reshape(1, -1)  # 设置岭回归分类器的权重
        ridge_classifier.intercept_ = ls.theta[0].reshape(1)  # 设置岭回归分类器的截距
        ridge_classifier.fit(train_data, train_label)  # 进一步调整模型参数
        ridge_pred_labels = ridge_classifier.predict(test_data)

        # 计算准确率和NMI
        ridge_acc = accuracy_score(test_label, ridge_pred_labels)
        ridge_nmi = normalized_mutual_info_score(test_label, ridge_pred_labels)
        print('准确率：', ridge_acc)
        print('NMI：', ridge_nmi)
        print()

