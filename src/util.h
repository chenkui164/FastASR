

#ifndef UTIL_H
#define UTIL_H
#include "Tensor.h"
extern void SaveDataFile(const char *filename, void *data, uint32_t len);

extern void relu(Tensor<float> *din);
extern void swish(Tensor<float> *din);

extern void softmax(float *din, int mask, int len);

extern void log_softmax(float *din, int len);

#endif
