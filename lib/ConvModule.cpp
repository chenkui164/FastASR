#include "ConvModule.h"
#include "util.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

void glu(Tensor<float> *din, Tensor<float> *dout)
{
    int mm = din->buff_size / 1024;
    int i, j;
    for (i = 0; i < mm; i++) {
        for (j = 0; j < 512; j++) {
            int in_off = i * 1024 + j;
            int out_off = i * 512 + j;
            float a = din->buff[in_off];
            float b = din->buff[in_off + 512];
            dout->buff[out_off] = a / (1 + exp(-b));
        }
    }
}

ConvModule::ConvModule(EncConvParams *params) : params(params)
{
    norm = new LayerNorm(&params->norm, 1e-5f);
    conv_cache = new Tensor<float>(14, 512);

    Tensor<float> tmp(14, 1024);
    int i, mm;
    for (i = 0; i < 14; i++) {
        int offset = i * 1024;
        memcpy(tmp.buff + offset, params->pointwise_conv1_bias,
               sizeof(float) * 1024);
    }

    glu(&tmp, conv_cache);
}

ConvModule::~ConvModule()
{
}

void ConvModule::forward(Tensor<float> *din)
{
    int mm = din->size[2];
    Tensor<float> tmp(mm, 1024);
    int i, j;
    for (i = 0; i < mm; i++) {
        int offset = i * 1024;
        memcpy(tmp.buff + offset, params->pointwise_conv1_bias,
               sizeof(float) * 1024);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 1024, 512, 1,
                din->buff, 512, params->pointwise_conv1_weight, 512, 1,
                tmp.buff, 1024);
    glu(&tmp, din);
    

    Tensor<float> conv_in(1, mm + 14);
    Tensor<float> blasin(mm, 15);
    Tensor<float> tmp_din(din);
    // conv_in.zeros();

    for (i = 0; i < 512; i++) {
        for (j = 0; j < 14; j++) {
            int ii = j * 512 + i;
            conv_in.buff[j] = conv_cache->buff[ii];
        }

        for (j = 14; j < 14 + mm; j++) {
            int ii = (j - 14) * 512 + i;
            conv_in.buff[j] = din->buff[ii];
            din->buff[ii] = params->depthwise_conv_bias[i];
        }

        for (j = 0; j < mm; j++) {
            int offset = j * 15;
            memcpy(blasin.buff + offset, conv_in.buff + j, 15 * sizeof(float));
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mm, 1, 15, 1,
                    blasin.buff, 15, params->depthwise_conv_weight + i * 15, 1,
                    1, din->buff + i, 512);
    }
    int t_offset = (mm - 14) * 512;
    memcpy(conv_cache->buff, tmp_din.buff + t_offset, 14 * 512 * sizeof(float));

    norm->forward(din);
    swish(din);

    Tensor<float> tmp2(din);
    for (i = 0; i < mm; i++) {
        int offset = i * 512;
        memcpy(din->buff + offset, params->pointwise_conv2_bias,
               sizeof(float) * 512);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 512, 512, 1,
                tmp2.buff, 512, params->pointwise_conv2_weight, 512, 1,
                din->buff, 512);
}
