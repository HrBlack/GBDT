#ifndef LoadData_H
#define LoadData_H

#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>

using namespace std;

typedef vector<vector<float>> Data;

Data LoadData(const char* input_file);

#endif