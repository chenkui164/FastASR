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

    int is_pre_english = false;
    int pre_english_len = 0;

    int is_combining = false;
    string combine = "";

    for (auto it = in.begin(); it != in.end(); it++) {
        string word = vocab[*it];

        // step1 空白字符不处理
        if (word == "<s>" || word == "</s>" || word == "<unk>")
            continue;

        // step2 将音素拼成完整的单词
        {
            int sub_word = !(word.find("@@") == string::npos);

            //处理单词起始和中间部分
            if (sub_word) {
                combine += word.erase(word.length() - 2);
                is_combining = true;
                continue;
            }
            //处理单词结束部分, combine结束
            else if (is_combining) {
                combine += word;
                is_combining = false;
                word = combine;
                combine = "";
            }
        }

        // step3 处理英文单词，单词之间需要加入空格，缩写转成大写。
        {

            //输入是汉字不需要处理
            if (isChinese(word)) {
                words.push_back(word);
                is_pre_english = false;
            }
            //输入是英文单词
            else {

                // 如果前面是汉字，不论当前是多个字母还是单个字母的单词，都不需要加空格
                if (!is_pre_english) {
                    word[0] = word[0] - 32;
                    words.push_back(word);
                    pre_english_len = word.size();

                }

                // 如果前面是单词
                else {

                    // 单个字母的单词变大写
                    if (word.size() == 1) {
                        word[0] = word[0] - 32;
                    }

                    // 前面单词的长度是大于1的，说明当前输入不属于缩写的部分，需要和前面的单词分割开，加空格
                    if (pre_english_len > 1) {
                        words.push_back(" ");
                        words.push_back(word);
                        pre_english_len = word.size();
                    } 
                    // 前面单词的长度是等于1，可能是属于缩写
                    else {
                        // 当前长度大于1, 不可能是缩写，所以需要插入空格
                        if (word.size() > 1) {
                            words.push_back(" ");
                        }
                        words.push_back(word);
                        pre_english_len = word.size();
                    }
                }

                is_pre_english = true;

            }
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
