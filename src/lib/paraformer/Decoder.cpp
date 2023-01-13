

#include <cblas.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../Tensor.h"
#include "../util.h"
#include "Decoder.h"
#include "ModelParams.h"

using namespace std;
using namespace paraformer;

Decoder::Decoder(DecoderParams *params) : params(params)
{
    conv_im2col = NULL;
    int i;
    for (i = 0; i < 16; i++) {
        sub_decoders[i] = new SubDecoder(&params->sub_decoders[i]);
    }

    decoder3_norm = new LayerNorm(&params->sub_decoders3.norm1, 1e-12, 512);
    feedforward = new FeedForwardDecoder(&params->sub_decoders3.feedforward);
    after_norm = new LayerNorm(&params->after_norm, 1e-12, 512);
}

Decoder::~Decoder()
{
}

void Decoder::get_conv_im2col(int mm)
{

    int idxs_size = mm * 11;
    if (conv_im2col != NULL)
        free(conv_im2col);
    conv_im2col = (int *)malloc(sizeof(int) * idxs_size);
    int step = 512;
    int i, j;
    int ii = 0;
    for (i = 0; i < mm; i++) {
        int start_idx = -5 + i;
        for (j = 0; j < 11; j++) {
            int val = start_idx + j;
            if (val >= 0 && val < mm)
                conv_im2col[ii++] = val * step;
            else
                conv_im2col[ii++] = -1;
        }
    }
}

void Decoder::forward(Tensor<float> *&din, Tensor<float> *enc)
{
    int mm = din->size[2];
    get_conv_im2col(mm);
    int i;
    for (i = 0; i < 16; i++) {
        sub_decoders[i]->forward(din, enc, conv_im2col);
    }
    decoder3_norm->forward(din);
    feedforward->forward(din);
    after_norm->forward(din);

    Tensor<float> *tmp = new Tensor<float>(mm, 8404);

    for (i = 0; i < mm; i++) {
        int offset = i * 8404;
        memcpy(tmp->buff + offset, params->linear_out_bias,
               8404 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 8404, 512, 1,
                din->buff, 512, params->linear_out_weight, 512, 1, tmp->buff,
                8404);

    int j;
    for (j = 0; j < mm; j++) {
        int offset = j * 8404;
        log_softmax(tmp->buff + offset, 8404);
    }
    delete din;
    din = tmp;
}
