from numpy import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-maximum_height', default=30, type=int)
parser.add_argument('-tree_mode', default='cart_regression', type=str)

args = parser.parse_args()

class DecisionTree:
    def __init__(self, is_leaf=False, left=None, right=None, height=0):
        self.left = left
        self.right = right
        self.height = height
        self.is_leaf = is_leaf
        self.split_feature = None
        self.split_value = None
        self.leaf_nums = 0
        self.threshold = 1

    def build_tree(self, tree, dataset):
        best_split_feature, split_value = self.choose_best_feature(tree, dataset)
        tree.split_feature = best_split_feature
        tree.split_value = split_value
        if tree.is_leaf is True or best_split_feature is None:
            return
        dataset_1, dataset_2 = self.split_dataset(dataset, best_split_feature, split_value)
        tree.left = DecisionTree(height=tree.height+1)
        tree.right = DecisionTree(height=tree.height+1)

        self.build_tree(tree.left, dataset_1)
        self.build_tree(tree.right, dataset_2)

    def split_dataset(self, dataset, feature, value):
        dataset_1 = dataset[nonzero(dataset[:, feature] <= value)[0]]
        dataset_2 = dataset[nonzero(dataset[:, feature] > value)[0]]
        return dataset_1, dataset_2

    def choose_best_feature(self, tree, dataset):
        if len(set(dataset[-1])) == 1 or tree.height >= args.maximum_height:
            tree.is_leaf = True
            return None, mean(dataset[-1])
        maximum_gain = 0
        feature_nums = len(dataset[0]) - 1 # 最后一列默认为label
        loss = self.compute_loss(dataset, args.tree_mode)
        for f in range(feature_nums):
            for val in dataset[:, f]:
                dataset_1, dataset_2 = self.split_dataset(dataset, f, val)
                loss_1 = self.compute_loss(dataset_1, args.tree_mode)
                loss_2 = self.compute_loss(dataset_2, args.tree_mode)
                new_loss = loss_1 + loss_2
                gain = loss - new_loss
                if gain < tree.threshold:
                    continue
                if gain > maximum_gain:
                    best_split_feature, split_value = f, val
                    maximum_gain = gain
                    self.loss = new_loss
        if maximum_gain == 0 or maximum_gain < tree.threshold:
            tree.is_leaf = True
            return None, mean(dataset[-1])
        else:
            return best_split_feature, split_value

    def compute_loss(self, dataset, mode):
        if mode == 'cart_regression':
            return var(dataset[:, -1]) * shape(dataset)[0]

    def predict(self, tree, dataset):
        row, col = dataset.shape
        result = 0
        for i in range(row):
            features = dataset[i, :-1]
            tmp_tree = tree
            while not tmp_tree.is_leaf:
                if features[tmp_tree.split_feature] <= tmp_tree.split_value:
                    tmp_tree = tmp_tree.left
                else:
                    tmp_tree = tmp_tree.right
            result += (tmp_tree.split_value - dataset[i, -1]) ** 2
            print("***** 测试样例为{}，输出结果为{:.2f} *****".format(features, tmp_tree.split_value))
        return result
            
    def tree_illustration(self, tree):
        queue = [tree]
        while queue:
            tree = queue.pop(0)
            if tree.is_leaf or not type(tree.split_feature) == int:
                print(tree.split_value)
                continue
            print(tree.split_feature, tree.split_value)
            queue.extend([tree.left, tree.right])

        return
class Data:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_file(self):
        dataset = loadtxt(self.file_path)
        return dataset

if __name__ == '__main__':
    data = Data('./bikeSpeedVsIq_train.txt')
    training_set = data.load_file()
    cart = DecisionTree(height=1)
    cart.build_tree(cart, training_set)

    # cart.tree_illustration(cart)

    data = Data('./bikeSpeedVsIq_test.txt')
    test_set = data.load_file()
    loss = cart.predict(cart, test_set)

    print("模型在test_set上的最终MSE值为{:.2f}".format(loss))
