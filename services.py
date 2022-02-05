import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sympy import *
import enum


class KMeansType(enum.Enum):
    SIMPLE_KMEANS = 1
    PLUS_PLUS_KMEANS = 2


class Info:
    def __init__(self, label, predict, data, cost_value):
        self.label = label
        self.predict = predict
        self.data = data
        self.cost_value = cost_value


class Data:
    trainData: np.array
    testData: np.array
    trainLabel: np.array
    testLabel: np.array

    def __init__(self, data, split_rate=0.2, bias=True, normal=False):
        self.trainData: np.array
        self.testData: np.array
        self.trainLabel: np.array
        self.testLabel: np.array
        self.split_rare = split_rate
        self.data = data
        self.bias = bias
        self.normal = normal
        self.prepare_data()

    def prepare_data(self):
        if self.normal:
            self.normalizer()

        # self.data[:, -1] = np.unique(self.data[:, -1], return_inverse=True)[1]
        if self.bias:
            self.data = np.insert(self.data, 0, 1, axis=1)

        self.trainData, self.testData, self.trainLabel, self.testLabel = train_test_split(self.data[:, :-1],
                                                                                          self.data[:, -1],
                                                                                          test_size=self.split_rare,
                                                                                          random_state=42)

    def normalizer(self):
        norm = np.linalg.norm(self.data[:, :-1])
        self.data[:, :-1] = self.data[:, :-1] / norm


def calculate_metrics(predicted, gold):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for p, g in zip(predicted, gold):

        if p == 1 and g == 1:
            true_pos += 1
        if p == 0 and g == 0:
            true_neg += 1
        if p == 1 and g == 0:
            false_pos += 1
        if p == 0 and g == 1:
            false_neg += 1

    recall = true_pos / float(true_pos + false_neg)

    precision = true_pos / float(true_pos + false_pos)

    fscore = 2 * precision * recall / (precision + recall)

    # accuracy = (true_pos + true_neg) / float(len(gold)) if gold else 0

    return precision, recall, fscore


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def mse(actual, predict):
    diff = np.subtract(predict, actual)
    ms = np.power(diff, 2)
    return np.mean(ms)


def init_weights(size):
    x = np.random.uniform(-0.01, 0.01, size=size)
    return x


def load_data(path, array=True, show_data=False):
    data = cv2.imread(path)
    if show_data:
        plt.imshow(data)
        plt.show()
    if array:
        data = np.array(data)

    data = data / 255
    return data
