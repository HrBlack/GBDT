#include<iostream>
#include<vector>
#include<cmath>

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
    bool is_leaf;
    int split_feature;
    float split_value;
    float threshold = 1;

    // DecisionTree(): left(NULL), right(NULL), height(1), is_leaf(false), split_feature(NULL), split_value(NULL) {};
    DecisionTree(
                 int h=1,
                 DecisionTree *l=NULL, 
                 DecisionTree *r=NULL, 
                 bool is_leaf=false,
                 int fea=NULL,
                 int val=NULL
                 ):
                 left(l), right(r), height(h), is_leaf(is_leaf), split_feature(fea), split_value(val) {};
    
    void build_tree(DecisionTree* tree, vector<vector<float>>& dataset){
        SplitResults feature_and_val = choose_best_feature(dataset);
        tree -> split_feature = feature_and_val.feature;
        tree -> split_value = feature_and_val.value;
        tree -> height += 1;
        SplitData new_dataset = split_dataset(dataset, split_feature, split_value);
        vector<vector<float>> dataset_1 = new_dataset.subdata_1;
        vector<vector<float>> dataset_2 = new_dataset.subdata_2;
        build_tree(tree -> left, dataset_1);
        build_tree(tree -> right, dataset_2);
    }

    SplitResults choose_best_feature(vector<vector<float>>& dataset){
        SplitResults result;
        float loss = MAXFLOAT;
        for (size_t i = 0; i < dataset[0].size(); ++i)
        {
            for (size_t j = 0; j < dataset.size(); ++j)
            {
                SplitData new_dataset = split_dataset(dataset, i, dataset[i][j]);
                vector<vector<float>> dataset_1 = new_dataset.subdata_1;
                vector<vector<float>> dataset_2 = new_dataset.subdata_2;
                float new_loss = compute_loss(dataset_1) + compute_loss(dataset_2);
                if((loss - new_loss) > threshold) 
                {
                    result.feature = i;
                    result.value = dataset[i][j];
                    loss = new_loss;
                }
            }
        }
        if (loss == MAXFLOAT)
        {
            result.feature = NULL;
            result.value = compute_mean(dataset);
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

    float predict(vector<float>& dataset)
    {
        if (is_leaf == true)
        {
            return split_value;
        }
        if (dataset[split_feature] <= split_value)
        {
            return left -> predict(dataset);
        }
        else
        {
            return right -> predict(dataset);
        }
    }
};

int main()
{
    Data dataset = LoadData("bikeSpeedVsIq_train.txt");
    DecisionTree tree = DecisionTree();
    tree.build_tree(& tree, dataset);
    for (size_t i = 0; i < dataset.size(); ++i)
    {
        cout << tree.predict(dataset[i]);
    }
    return 0;
}