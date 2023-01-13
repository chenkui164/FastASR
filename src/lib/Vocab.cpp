#include "Vocab.h"

#include <fstream>
#include <iostream>
#include <list>
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
    }
    // else // 没有该文件
    //{
    //     cout << "no such file" << endl;
    // }
}
Vocab::~Vocab()
{
}

string Vocab::vector2string(vector<int> in)
{
    int i;
    stringstream ss;
    for (auto it = in.begin(); it != in.end(); it++) {
        ss << vocab[*it];
    }

    return ss.str();
}

int str2int(string str)
{
    const char *ch_array = str.c_str();
    if (((ch_array[0] & 0xf0) != 0xe0) || ((ch_array[1] & 0xc0) != 0x80) ||
        ((ch_array[2] & 0xc0) != 0x80))
        return 0;

    int val = ((ch_array[0] & 0x0f) << 12) | ((ch_array[1] & 0x3f) << 6) |
              (ch_array[2] & 0x3f);
    return val;
}

bool Vocab::isChinese(string ch)
{
    if (ch.size() != 3) {
        return false;
    }

    int unicode = str2int(ch);
    if (unicode >= 19968 && unicode <= 40959) {
        return true;
    }

    return false;
}

string Vocab::vector2stringV2(vector<int> in)
{
    int i;
    list<string> words;

    int strstatus = 0;
    bool isAllChinese = true;
    bool isAllAlpha = true;

    string combine = "";
    int is_combining = false;
    int is_pre_english = false;
    int pre_english_len = 0;

    for (auto it = in.begin(); it != in.end(); it++) {
        string word = vocab[*it];
        // words.push_back(word);
        int sub_word = !(word.find("@@") == string::npos);
        // cout << word << "," << word.size() << "," << isChinese(word) << ","
        //      << sub_word << endl;
        if (word == "<s>" || word == "</s>" || word == "<unk>")
            continue;
        if (sub_word) {
            combine += word.erase(word.length() - 2);
            is_combining = true;
            continue;
        } else if (is_combining) {
            combine += word;
            is_combining = false;
            // cout << combine << endl;
            word = combine;
            combine = "";
        }

        if (isChinese(word)) {
            words.push_back(word);
            is_pre_english = false;
        } else {
            if (is_pre_english && pre_english_len > 1) {
                words.push_back(" ");
                if (word.size() == 1) {
                    word[0] = word[0] - 32;
                }
                words.push_back(word);
            } else {
                if (word.size() == 1) {
                    word[0] = word[0] - 32;
                }
                words.push_back(word);
            }
            is_pre_english = true;
            pre_english_len = word.size();
        }
    }

    // for (auto it = words.begin(); it != words.end(); it++) {
    //     cout << *it << endl;
    // }

    stringstream ss;
    for (auto it = words.begin(); it != words.end(); it++) {
        ss << *it;
    }

    return ss.str();
}

int Vocab::size()
{
    return vocab.size();
}
