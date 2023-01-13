

#include <cblas.h>
#include <math.h>
#include <string.h>

#include "../util.h"
#include "DecoderSrcAttn.h"

using namespace paraformer;

DecoderSrcAttn::DecoderSrcAttn(DecSelfAttnParams *params) : params(params)
{
}

DecoderSrcAttn::~DecoderSrcAttn()
{
}

void DecoderSrcAttn::linear_forward(Tensor<float> *din, Tensor<float> *dout,
                                    float *weight, float *bias)
{
    int mm = din->size[2];
    int o_size = dout->size[3];
    int offset = 0;
    int i;

    for (i = 0; i < mm; i++) {
        memcpy(dout->buff + offset, bias, sizeof(float) * o_size);
        offset += o_size;
    }

    // disp_params(weight, 10);
    // disp_params(bias, 10);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, o_size, 512, 1,
                din->buff, 512, weight, 512, 1, dout->buff, o_size);
}

void DecoderSrcAttn::forward(Tensor<float> *&din, Tensor<float> *enc)
{
    int m1 = din->size[2];
    int m2 = enc->size[2];
    Tensor<float> q = new Tensor<float>(m1, 512);
    Tensor<float> kv = new Tensor<float>(m2, 1024);

    linear_forward(din, &q, params->linear_q_weight, params->linear_q_bias);
    linear_forward(enc, &kv, params->linear_kv_weight, params->linear_kv_bias);

    Tensor<float> scores(m1, m2);
    Tensor<float> attnout(m1, 512);
    attnout.zeros();

    int head_step = 512 / 4;
    int k_offset = 0;
    int v_offset = 512;
    
    int i;

    for (i = 0; i < 4; i++) {
        float *sub_q = q.buff + i * head_step;
        float *sub_k = kv.buff + i * head_step + k_offset;
        float *sub_v = kv.buff + i * head_step + v_offset;
        float *sub_attn = attnout.buff + i * head_step;

        scores.zeros();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m1, m2, head_step,
                    1, sub_q, 512, sub_k, 1024, 1, scores.buff, m2);

        int j;
        for (j = 0; j < m1; j++) {
            int offset = j * m2;
            softmax(scores.buff + offset, scores.size[3], scores.size[3]);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1, head_step,
                    m2, 1, scores.buff, m2, sub_v, 1024, 1, sub_attn, 512);
    }

    linear_forward(&attnout, din, params->linear_out_weight, params->linear_out_bias);



}
