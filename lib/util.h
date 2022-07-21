

#ifndef UTIL_H
#define UTIL_H
#include "Tensor.h"
#include <iostream>

using namespace std;

extern void SaveDataFile(const char *filename, void *data, uint32_t len);

extern void relu(Tensor<float> *din);
extern void swish(Tensor<float> *din);

extern void softmax(float *din, int mask, int len);

extern void log_softmax(float *din, int len);

string pathAppend(const string &p1, const string &p2);

#endif
