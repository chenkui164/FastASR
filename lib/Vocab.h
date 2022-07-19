
#ifndef VOCAB_H
#define VOCAB_H

#include <stdint.h>
#include <string>
#include <vector>
using namespace std;

class Vocab {
  private:
    vector<string> vocab;

  public:
    Vocab(const char *filename);
    ~Vocab();
    int size();
    string vector2string(vector<int> in);
};

#endif
