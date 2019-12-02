#include<iostream>
#include<vector>
#include<cmath>
#include<unordered_set>

#include "data.h"

using namespace std;

class SplitResults{
    public:
    int feature;
    float value;
};

class SplitData{
    public:
    Data subdata_1;
    Data subdata_2;
};

class DecisionTree
{
    public:
    DecisionTree *left;
    DecisionTree *right;
    int height;
    int maximum_height = 15;
    bool is_leaf;
    int split_feature;
    float split_value;
    float threshold;

    // DecisionTree(): left(NULL), right(NULL), height(1), is_leaf(false), split_feature(NULL), split_value(NULL) {};
    DecisionTree(
                 int h=1,
                 DecisionTree *l = NULL, 
                 DecisionTree *r = NULL, 
                 bool is_leaf = false,
                 int feat = -1,
                 float val = 0,
                 float t = 1
                 ):
                 left(l), right(r), height(h), is_leaf(is_leaf), split_feature(feat), split_value(val), threshold(t) {};
    
    void build_tree(DecisionTree* tree, vector<vector<float>>& dataset)
    {
        // cout<<tree->height<<endl;
        // if (dataset.size() == 0)
        // {
        //     return;
        // }
        SplitResults feature_and_val = choose_best_feature(tree, dataset);
        tree -> split_feature = feature_and_val.feature;
        tree -> split_value = feature_and_val.value;
        tree -> height += 1;
        if (tree -> is_leaf == true || tree -> split_feature == -1)
        {
            return;
        }
        SplitData new_dataset = split_dataset(dataset, split_feature, split_value);
        vector<vector<float>> dataset_1 = new_dataset.subdata_1;
        vector<vector<float>> dataset_2 = new_dataset.subdata_2;
        DecisionTree left_tree(tree -> height + 1);
        // cout<<"left_tree height = "<<left_tree.height<<endl;
        DecisionTree right_tree(tree -> height + 1);
        // cout<<"right_tree height = "<<right_tree.height<<endl;

        tree -> left = &left_tree;
        tree -> right = &right_tree;
        build_tree(tree -> left, dataset_1);
        build_tree(tree -> right, dataset_2);
        return;
    }

    SplitResults choose_best_feature(DecisionTree* tree, vector<vector<float>>& dataset){
        SplitResults result;
        unordered_set<float> set;
        // cout<<tree -> height<<endl;
        if (tree -> height >= tree -> maximum_height)
        {
            result.feature = -1;
            result.value = compute_mean(dataset);
            tree -> is_leaf = true;
            return result;
        }
        for (size_t i = 0; i < dataset.size(); ++i)
        {
            set.insert(dataset[i].back());
        }
        if (set.size() <= 1) 
        {
            result.feature = -1;
            result.value = compute_mean(dataset);
            tree -> is_leaf = true;
            return result;
        }
        float loss = compute_loss(dataset);
        float maximum_gain = 0;
        for (size_t i = 0; i < dataset[0].size() - 1; ++i) // 最后一列默认为label
        {
            for (size_t j = 0; j < dataset.size(); ++j)
            {
                SplitData new_dataset = split_dataset(dataset, i, dataset[j][i]);
                vector<vector<float>> dataset_1 = new_dataset.subdata_1;
                vector<vector<float>> dataset_2 = new_dataset.subdata_2;
                float new_loss = (compute_loss(dataset_1) + compute_loss(dataset_2))/2;
                float gain = loss - new_loss;
                // cout<<"new_loss = "<<new_loss<<endl;
                if (gain < tree -> threshold)
                {
                    continue;
                }
                if (gain > maximum_gain)
                {
                    result.feature = i;
                    result.value = dataset[j][i];
                    maximum_gain = gain;
                }
            }
        }
        if (maximum_gain == 0 || maximum_gain < tree -> threshold)
        {
            result.feature = -1;
            result.value = compute_mean(dataset);
            tree -> is_leaf = true;
        }
        return result;
    }

    SplitData split_dataset(vector<vector<float>>& dataset, int feature, float value){
        SplitData new_dataset;
        for (size_t i = 0; i < dataset.size(); ++i)
        {
            if (dataset[i][feature] <= value)
            {
                new_dataset.subdata_1.push_back(dataset[i]);
            }
            else
            {
                new_dataset.subdata_2.push_back(dataset[i]);
            }
        }
        return new_dataset;
    }

    float compute_mean(Data& dataset){
        float total = 0;
        for (size_t i = 0; i < dataset.size(); ++i)
        {
            total += dataset[i].back();
        }
        return total / dataset.size();
    }

    float compute_loss(Data& dataset){
        float loss = 0;
        float mean = compute_mean(dataset);
        for (size_t i = 0; i < dataset.size(); ++i)
        {
            loss += (dataset[i].back() - mean) * (dataset[i].back() - mean);
        }
        loss = sqrt(loss / dataset.size());
        return loss;
    }

    float predict(DecisionTree* tree, vector<float>& dataset)
    {
        if (tree -> is_leaf == true || tree -> split_feature == -1)
        {
            return tree -> split_value;
        }
        if (dataset[tree -> split_feature] <= tree -> split_value)
        {
            // if(!tree -> left)
            //     cout<<"!!!!!!!!!!!!!"<<endl;
            return predict(tree -> left, dataset);
        }
        else
        {
            // if(!tree -> right)
            //     cout<<"!!!!!!!!!!!!!"<<endl;
            return predict(tree -> right, dataset);
        }
    }
};

int main()
{
    Data training_set = LoadData("/Users/liushihao/Desktop/搜索引擎基础/GBDT/bikeSpeedVsIq_train.txt");
    DecisionTree tree;
    tree.build_tree(&tree, training_set);

    // vector<DecisionTree> queue;
    // while (tree.is_leaf == false)
    // {
        
    //     queue.push_back(tree.left);
    //     queue.push_back(tree.right);
    // }

    Data test_set = LoadData("/Users/liushihao/Desktop/搜索引擎基础/GBDT/bikeSpeedVsIq_test.txt");
    for (size_t i = 0; i < test_set.size(); ++i)
    {
        cout << tree.predict(&tree, test_set[i]) << endl;
    }
    return 0;
}
