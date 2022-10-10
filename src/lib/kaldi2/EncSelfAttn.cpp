#include <cblas.h>
#include <math.h>
#include <string.h>

#include "../util.h"
#include "EncSelfAttn.h"

EncSelfAttn::EncSelfAttn(EncSelfAttnParams *params) : params(params)
{
}

EncSelfAttn::~EncSelfAttn()
{
}


void EncSelfAttn::forward(Tensor<float> *din, Tensor<float> *pe)
{
    int Tmax = din->size[2];
    int Pmax = pe->size[2];
    int nn = 512 * 3;
    Tensor<float> linear_out(Tmax, nn);
    Tensor<float> p(Pmax, 512);

    int i;
    for (i = 0; i < Tmax; i++) {
        int offset = i * nn;
        memcpy(linear_out.buff + offset, params->in_proj_bias,
               sizeof(float) * nn);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Tmax, nn, 512, 1,
                din->buff, 512, params->in_proj_weight, 512, 1, linear_out.buff,
                nn);

    p.zeros();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Pmax, 512, 512, 1,
                pe->buff, 512, params->linear_pos_weight, 512, 1, p.buff, 512);

    Tensor<float> q_with_bias_u(Tmax, 512);
    Tensor<float> q_with_bias_v(Tmax, 512);

    int j;
    for (i = 0; i < Tmax; i++) {
        for (j = 0; j < 512; j++) {
            int idx = i * 512 + j;
            int ii = i * 512 * 3 + j;
            float val = linear_out.buff[ii] / 8;
            q_with_bias_u.buff[idx] = val + params->pos_bias_u[j];
            q_with_bias_v.buff[idx] = val + params->pos_bias_v[j];
        }
    }

    Tensor<float> matrix_ac(Tmax, 8, Tmax);
    Tensor<float> matrix_bd(Tmax, 8, Pmax);
    Tensor<float> matrix_bd_new(Tmax, 8, Tmax);

    matrix_ac.zeros();
    matrix_bd.zeros();

    for (i = 0; i < 8; i++) {
        int offset1 = 64 * i;
        int offset2 = Tmax * i;
        int offset3 = Pmax * i;
        float *k_base_addr = linear_out.buff + 512;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Tmax, Tmax, 64, 1,
                    q_with_bias_u.buff + offset1, 512, k_base_addr + offset1,
                    512 * 3, 1, matrix_ac.buff + offset2, Tmax * 8);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Tmax, Pmax, 64, 1,
                    q_with_bias_v.buff + offset1, 512, p.buff + offset1, 512, 1,
                    matrix_bd.buff + offset3, Pmax * 8);
    }

    for (i = 0; i < Tmax; i++) {
        int offset3 = (Pmax) / 2 - i;
        for (j = 0; j < 8; j++) {
            int offset1 = (i * 8 + j) * Tmax;
            int offset2 = (i * 8 + j) * Pmax;
            memcpy(matrix_bd_new.buff + offset1,
                   matrix_bd.buff + offset2 + offset3, sizeof(float) * Tmax);
        }
    }

    Tensor<float> scores(Tmax, 8, Tmax);
    for (i = 0; i < scores.buff_size; i++) {
        scores.buff[i] = matrix_bd_new.buff[i] + matrix_ac.buff[i];
    }

    for (i = 0; i < Tmax * 8; i++) {
        int offset = i * Tmax;
        softmax(scores.buff + offset, Tmax, Tmax);
    }

    Tensor<float> tmp(Tmax, 512);
    tmp.zeros();

    for (i = 0; i < 8; i++) {
        int offset1 = Tmax * i;
        int offset2 = 64 * i;
        float *v_base_addr = linear_out.buff + 1024;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Tmax, 64, Tmax,
                    1, scores.buff + offset1, Tmax * 8, v_base_addr + offset2,
                    512 * 3, 1, tmp.buff + offset2, 512);
    }

    for (i = 0; i < din->size[2]; i++) {
        int offset = i * 512;
        memcpy(din->buff + offset, params->out_proj_bias, 512 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, din->size[2], 512, 512,
                1, tmp.buff, 512, params->out_proj_weight, 512, 1, din->buff,
                512);

}
