

#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../util.h"
#include "FeedForward.h"

using namespace kaldi2;

FeedForward::FeedForward(FeedForwardParams *params) : params(params)
{
}

FeedForward::~FeedForward()
{
}

void FeedForward::forward(Tensor<float> *din)
{
    int nn = din->size[3];
    int mm = din->buff_size / nn;
    int i;
    Tensor<float> tmp(mm, 2048);

    for (i = 0; i < mm; i++) {
        int offset = i * 2048;
        memcpy(tmp.buff + offset, params->w1_bias, 2048 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 2048, 512, 1,
                din->buff, 512, params->w1_weight, 512, 1, tmp.buff, 2048);


    doubleswish(&tmp);

    for (i = 0; i < mm; i++) {
        int offset = i * 512;
        memcpy(din->buff + offset, params->w2_bias, 512 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 512, 2048, 1,
                tmp.buff, 2048, params->w2_weight, 2048, 1, din->buff, 512);
}
