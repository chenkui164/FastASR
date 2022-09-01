#include <cblas.h>
#include <math.h>
#include <string.h>

#include "EncSelfAttn.h"
#include "../util.h"

EncSelfAttn::EncSelfAttn(EncSelfAttnParams *params) : params(params)
{
}

EncSelfAttn::~EncSelfAttn()
{
}

void linear_forward(Tensor<float> *din, Tensor<float> *dout, float *weight,
                    float *bias)
{
    int mm = din->buff_size / 512;
    int i;
    if (bias != 0) {
        for (i = 0; i < mm; i++) {
            int offset = i * 512;
            memcpy(dout->buff + offset, bias, sizeof(float) * 512);
        }
    } else {
        dout->zeros();
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mm, 512, 512, 1,
                din->buff, 512, weight, 512, 1, dout->buff, 512);
}

void EncSelfAttn::forward(Tensor<float> *query, Tensor<float> *key,
                          Tensor<float> *value, Tensor<float> *pe)
{
    Tensor<float> q(query->size[2], 8, query->size[3] / 8);
    Tensor<float> k(key->size[2], 8, key->size[3] / 8);
    Tensor<float> v(value->size[2], 8, value->size[3] / 8);
    Tensor<float> p(pe->size[2], 8, pe->size[3] / 8);

    linear_forward(query, &q, params->linear0.linear_q_weight,
                   params->linear0.linear_q_bias);

    linear_forward(key, &k, params->linear0.linear_k_weight,
                   params->linear0.linear_k_bias);

    linear_forward(value, &v, params->linear0.linear_v_weight,
                   params->linear0.linear_v_bias);

    linear_forward(pe, &p, params->linear_pos_weight, NULL);

    Tensor<float> q_with_bias_u(&q);
    Tensor<float> q_with_bias_v(&q);

    int i, j;
    for (i = 0; i < q.buff_size; i++) {
        int ii = i % 512;
        q_with_bias_u.buff[i] += params->pos_bias_u[ii];
        q_with_bias_v.buff[i] += params->pos_bias_v[ii];
    }

    Tensor<float> scores(q.size[1], 8, k.size[1]);

    scores.zeros();

    for (i = 0; i < 8; i++) {
        int offset1 = q.size[3] * i;
        int offset2 = scores.size[3] * i;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, q.size[1],
                    k.size[1], q.size[3], 1, q_with_bias_u.buff + offset1, 512,
                    k.buff + offset1, 512, 1, scores.buff + offset2,
                    k.size[1] * 8);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, q.size[1],
                    k.size[1], q.size[3], 1, q_with_bias_v.buff + offset1, 512,
                    p.buff + offset1, 512, 1, scores.buff + offset2,
                    k.size[1] * 8);
    }

    for (i = 0; i < scores.buff_size; i++) {
        scores.buff[i] = scores.buff[i] / 8;
    }
    int mm = scores.buff_size / scores.size[3];

    for (i = 0; i < mm; i++) {
        int offset = i * scores.size[3];
        softmax(scores.buff + offset, scores.size[3], scores.size[3]);
    }

    Tensor<float> tmp(query->size[2], query->size[3]);
    tmp.zeros();

    for (i = 0; i < 8; i++) {
        int offset1 = scores.size[3] * i;
        int offset2 = v.size[3] * i;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, scores.size[1],
                    v.size[3], v.size[1], 1, scores.buff + offset1,
                    scores.size[3] * 8, v.buff + offset2, 512, 1,
                    tmp.buff + offset2, 512);
    }

    for (i = 0; i < query->size[2]; i++) {
        int offset = i * 512;
        memcpy(query->buff + offset, params->linear0.linear_out_bias,
               512 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, query->size[2], 512,
                512, 1, tmp.buff, 512, params->linear0.linear_out_weight, 512,
                1, query->buff, 512);
}
