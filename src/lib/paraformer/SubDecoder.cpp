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

SubDecoder::SubDecoder(SubDecoderParams *params) : params(params)
{
    norm1 = new LayerNorm(&params->norm1, 1e-12, 512);
    feedforward = new FeedForwardDecoder(&params->feedforward);
    norm2 = new LayerNorm(&params->norm2, 1e-12, 512);
    norm3 = new LayerNorm(&params->norm3, 1e-12, 512);
    src_attn = new DecoderSrcAttn(&params->src_attn);
}

SubDecoder::~SubDecoder()
{

    delete norm1;
    delete feedforward;
    delete norm2;
    delete norm3;
    delete src_attn;
}

void SubDecoder::forward_fsmn(Tensor<float> *din, int *conv_im2col)
{

    int mm = din->size[2];
    int v_offset = 0;

    Tensor<float> blasin(mm, 11);
    int i, j;

    for (i = 0; i < 512; i++) {
        for (j = 0; j < mm * 11; j++) {
            int tmp_idx = conv_im2col[j];
            if (tmp_idx == -1)
                blasin.buff[j] = 0;
            else
                blasin.buff[j] = din->buff[tmp_idx + v_offset];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mm, 1, 11, 1,
                    blasin.buff, 11, params->fsmn_block_weight + i * 11, 1, 1,
                    din->buff + v_offset, 512);

        v_offset++;
    }
}

void SubDecoder::forward(Tensor<float> *din, Tensor<float> *enc,
                         int *conv_im2col)
{
    Tensor<float> residual(din);
    norm1->forward(din);
    feedforward->forward(din);
    norm2->forward(din);
    forward_fsmn(din, conv_im2col);
    din->add(&residual);
    residual.reload(din);
    norm3->forward(din);
    src_attn->forward(din, enc);
    din->add(&residual);
}
