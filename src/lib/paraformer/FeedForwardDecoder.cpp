

#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../util.h"
#include "FeedForwardDecoder.h"

using namespace paraformer;

FeedForwardDecoder::FeedForwardDecoder(DecoderFeedForwardParams *params)
    : params(params)
{
    activate = &relu;
    norm = new LayerNorm(&params->norm, 1e-12, 2048);
}

FeedForwardDecoder::~FeedForwardDecoder()
{
}

void FeedForwardDecoder::forward(Tensor<float> *din)
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

    activate(&tmp);
    norm->forward(&tmp);

    din->zeros();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 512, 2048, 1,
                tmp.buff, 2048, params->w2_weight, 2048, 1, din->buff, 512);
}
