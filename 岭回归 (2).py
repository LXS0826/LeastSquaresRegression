# encoding:utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import metrics
import scipy.io as scio


def load_data(path):
    data = scio.loadmat(path)
    data_x = np.array(data['X'])
    data_y = np.array(data['Y'][:, 0] - 1)
    train_data, test_data, train_label, test_label = train_test_split(data_x, data_y, test_size=0.25, random_state=10)
    return train_data, test_data, train_label, test_label


def acc_rate(gt_s, s):
    err_x = np.sum(gt_s == s)  # 比较预测标签和真实标签的相同元素数量
    acc = err_x.astype(float) / gt_s.shape[0]  # 计算准确率
    return acc


if __name__ == '__main__':
    mnist_path = 'datasets/MNIST.mat'
    lung_path = 'datasets/lung.mat'
    yale_path = 'datasets/Yale.mat'

    print('mnist:')
    train_data, test_data, train_label, test_label = load_data(mnist_path)
    model = Ridge(alpha=0.2)  # 设置 L2 正则化参数
    model.fit(train_data, train_label)
    pred = model.predict(test_data)
    pred_labels = np.round(pred)
    print('准确率：', acc_rate(test_label, pred_labels))
    NMI = metrics.normalized_mutual_info_score(pred_labels, test_label)
    print('NMI: ', NMI)

    print('lung:')
    train_data, test_data, train_label, test_label = load_data(lung_path)
    model = Ridge(alpha=0.2)  # 设置 L2 正则化参数
    model.fit(train_data, train_label)
    pred = model.predict(test_data)
    pred_labels = np.round(pred)
    print('准确率：', acc_rate(test_label, pred_labels))
    NMI = metrics.normalized_mutual_info_score(pred_labels, test_label)
    print('NMI: ', NMI)

    print('yale:')
    train_data, test_data, train_label, test_label = load_data(yale_path)
    model = Ridge(alpha=0.2)  # 设置 L2 正则化参数
    model.fit(train_data, train_label)
    pred = model.predict(test_data)
    pred_labels = np.round(pred)
    print('准确率：', acc_rate(test_label, pred_labels))
    NMI = metrics.normalized_mutual_info_score(pred_labels, test_label)
    print('NMI: ', NMI)
