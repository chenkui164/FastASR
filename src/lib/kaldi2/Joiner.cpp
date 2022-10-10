
#include "Joiner.h"
#include "../util.h"
#include <cblas.h>
#include <math.h>
#include <stdio.h>

using namespace kaldi2;

Joiner::Joiner(JoinerParams *params) : params(params)
{
}

Joiner::~Joiner()
{
}

void Joiner::encoder_forward(Tensor<float> *&din)
{
    int mm = din->size[2];
    Tensor<float> *dout = new Tensor<float>(mm, 512);

    int i;
    for (i = 0; i < mm; i++) {
        int offset = i * 512;
        memcpy(dout->buff + offset, params->decoder_proj_bias,
               sizeof(float) * 512);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 512, 512, 1,
                din->buff, 512, params->encoder_proj_weight, 512, 1, dout->buff,
                512);
    delete din;
    din = dout;
}

void Joiner::decoder_forward(Tensor<float> *&din)
{
    Tensor<float> *dout = new Tensor<float>(1, 512);
    memcpy(dout->buff, params->decoder_proj_bias, sizeof(float) * 512);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, 512, 512, 1,
                din->buff, 512, params->decoder_proj_weight, 512, 1, dout->buff,
                512);
    delete din;
    din = dout;
}

void Joiner::linear_forward(float *encoder, float *decoder, Tensor<float> *dout)
{

    float din[512];

    int i;
    for (i = 0; i < 512; i++) {
        float tmp = encoder[i] + decoder[i];
        din[i] = tanh(tmp);
    }

    memcpy(dout->buff, params->output_linear_bias, 5537 * sizeof(float));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, 5537, 512, 1,
                din, 512, params->output_linear_weight, 512, 1,
                dout->buff, 5537);
}
