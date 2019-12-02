'''
Created on Nov 30st, 2019
Tree-Based Regression Methods
@author: Shihao Liu
'''

import numpy as np
import logging 

from decision_tree import DecisionTree, Data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GBDT():
    def __init__(self, lr=0.8, iterations=15):
        self.forest = []
        self.learning_rate = lr
        self.iterations = iterations
        self.lr_list = []

    def fit(self, dataset):
        base_constant_prediction = np.mean(dataset[:, -1])
        base_tree = DecisionTree(is_leaf=True, split_value=base_constant_prediction)
        residuals = dataset[:, -1] - base_constant_prediction
        self.lr_list.append(self.learning_rate)
        self.forest.append(base_tree)
        for iter in range(self.iterations-1):
            dataset = np.concatenate((dataset[:, :-1], residuals[:, None]), axis=1)
            tree = DecisionTree()
            tree.build_tree(tree, dataset)
            for i, example in enumerate(dataset):
                residuals[i] -= tree.predict(tree, example)
            self.forest.append(tree)
            self.lr_list.append(self.learning_rate * self.lr_list[iter])

    def predict(self, dataset):
        loss = 0
        for example in dataset:
            prediction = 0
            for (lr, tree) in zip(self.lr_list, self.forest):
                prediction += lr * tree.predict(tree, example)
            loss += (example[-1] - prediction) ** 2
            logger.info("***** 特征向量为{}，真实输出为{:.2f}，预测值为{:.2f}".format(example[:-1], example[-1], prediction))
        logger.info("********** 模型的loss为{:.2f} **********".format(loss))

if __name__ == '__main__':
    gbdt = GBDT()
    data = Data('bikeSpeedVsIq_train.txt')
    training_set = data.load_file()
    gbdt.fit(training_set)

    data = Data('./bikeSpeedVsIq_test.txt')
    test_set = data.load_file()
    gbdt.predict(test_set)