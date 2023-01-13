
#include <cblas.h>
#include <math.h>
#include <string.h>

#include "../util.h"
#include "EncSelfAttn.h"

using namespace paraformer;

EncSelfAttn::EncSelfAttn(EncSelfAttnParams *params) : params(params)
{
}

EncSelfAttn::~EncSelfAttn()
{
}

void EncSelfAttn::linear_qkv_forward(Tensor<float> *din, Tensor<float> *dout,
                                     float *weight, float *bias)
{
    int mm = din->size[2];
    int nn = din->size[3];
    int i;
    int offset = 0;
    for (i = 0; i < mm; i++) {
        memcpy(dout->buff + offset, bias, sizeof(float) * 1536);
        offset += 1536;
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 1536, nn, 1,
                din->buff, nn, weight, nn, 1, dout->buff, 1536);
}

void EncSelfAttn::forward_fsmn(Tensor<float> *din, int *conv_im2col)
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

void EncSelfAttn::forward(Tensor<float> *&din, Tensor<float> *&fsmn_memory,
                          int *conv_im2col)
{
    int Tmax = din->size[2];
    Tensor<float> qkv(Tmax, 1536);

    // din->disp();
    linear_qkv_forward(din, &qkv, params->linear_qkv_weight,
                       params->linear_qkv_bias);

    fsmn_memory = new Tensor<float>(Tmax, 512);

    int i;
    int offset0 = 0;
    int offset1 = 0;
    for (i = 0; i < Tmax; i++) {
        memcpy(fsmn_memory->buff + offset0, qkv.buff + 1024 + offset1,
               512 * sizeof(float));
        offset0 += 512;
        offset1 += 1536;
    }
    forward_fsmn(fsmn_memory, conv_im2col);

    Tensor<float> scores(Tmax, Tmax);
    Tensor<float> attnout(Tmax, 512);
    attnout.zeros();

    int head_step = 512 / 4;
    int q_offset = 0;
    int k_offset = 512;
    int v_offset = 1024;
    int next_column = 1536;

    for (i = 0; i < 4; i++) {
        float *sub_q = qkv.buff + i * head_step + q_offset;
        float *sub_k = qkv.buff + i * head_step + k_offset;
        float *sub_v = qkv.buff + i * head_step + v_offset;
        float *sub_attn = attnout.buff + i * head_step;

        scores.zeros();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Tmax, Tmax,
                    head_step, 1, sub_q, next_column, sub_k, next_column, 1,
                    scores.buff, Tmax);


        int j;
        for (j = 0; j < Tmax; j++) {
            int offset = j * Tmax;
            softmax(scores.buff + offset, scores.size[3], scores.size[3]);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Tmax, head_step,
                    Tmax, 1, scores.buff, Tmax, sub_v, next_column, 1, sub_attn,
                    512);
    }

    Tensor<float> *tmp_out = new Tensor<float>(Tmax, 512);
    for (i = 0; i < Tmax; i++) {
        int offset = i * 512;
        memcpy(tmp_out->buff + offset, params->linear_out_bias,
               512 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Tmax, 512, 512, 1,
                attnout.buff, 512, params->linear_out_weight, 512, 1,
                tmp_out->buff, 512);
    delete din;
    din = tmp_out;
}
