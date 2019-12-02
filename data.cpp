#include "data.h"

Data LoadData(const char* input_file)
{
    Data data;
    string tmp_line;
    ifstream inputs;
    vector<float> row_data;
    float val;
    inputs.open(input_file); // 这里的形参input_file必须是指针
    while (!inputs.eof())
    {
        getline(inputs, tmp_line, '\n');
        if (tmp_line == "\0")
        {
            return data;
        }
        stringstream input_line(tmp_line);
        while (input_line >> val)
        {
            row_data.push_back(val);
        }
        data.push_back(row_data);
        row_data.clear();
    }
    inputs.close();
    return data;
}

// int main()
// {
//     Data dataset = LoadData("bikeSpeedVsIq_train.txt");
// }