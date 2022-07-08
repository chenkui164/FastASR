#include "Vocab.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

Vocab::Vocab(const char *filename)
{
    ifstream in(filename);
    string line;

    if (in) // 有该文件
    {
        while (getline(in, line)) // line中不包括每行的换行符
        {
            vocab.push_back(line);
        }
        // cout << vocab[1719] << endl;
    } else // 没有该文件
    {
        cout << "no such file" << endl;
    }
}
Vocab::~Vocab()
{
}

string Vocab::vector2string(vector<int> in)
{
    int i;
    stringstream ss;
    for (auto it = in.begin(); it != in.end(); it++)
        ss << vocab[*it];

    return ss.str();
}
