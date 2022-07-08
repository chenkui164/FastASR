#include "DecSelfAttn.h"
#include "util.h"
#include <cblas.h>

DecSelfAttn::DecSelfAttn(SelfAttnParams *params) : params(params)
{
}

DecSelfAttn::~DecSelfAttn()
{
}

extern void linear_forward(Tensor<float> *din, Tensor<float> *dout,
                           float *weight, float *bias);
// {
//     int mm = din->buff_size / 512;
//     int i;
//     if (bias != 0) {
//         for (i = 0; i < mm; i++) {
//             int offset = i * 512;
//             memcpy(dout->buff + offset, bias, sizeof(float) * 512);
//         }
//     } else {
//         dout->zeros();
//     }

//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mm, 512, 512, 1,
//                 din->buff, 512, weight, 512, 1, dout->buff, 512);
// }

void DecSelfAttn::forward(Tensor<float> *query, Tensor<float> *key,
                          Tensor<float> *value, Tensor<int> *mask,
                          Tensor<float> *dout)
{

    Tensor<float> q(query->size[1], query->size[2], 8, query->size[3] / 8);
    Tensor<float> k(key->size[1], key->size[2], 8, key->size[3] / 8);
    Tensor<float> v(value->size[1], value->size[2], 8, value->size[3] / 8);

    linear_forward(query, &q, params->linear_q_weight, params->linear_q_bias);

    linear_forward(key, &k, params->linear_k_weight, params->linear_k_bias);

    linear_forward(value, &v, params->linear_v_weight, params->linear_v_bias);

    // SaveDataFile("/home/ck/matlab/conformer/test.bin", v.buff,
    //              v.buff_size * sizeof(float));

    int n_batch = q.size[0];
    int n_head = 8;
    int n_query = q.size[1];
    int n_key = k.size[1];

    Tensor<float> attn(1, 1, n_query, n_key);
    // attn.shape();
    dout->zeros();

    int i, j;
    for (i = 0; i < n_batch; i++) {
        for (j = 0; j < n_head; j++) {
            int q_offset = i * q.size[1] * q.size[2] * q.size[3] + j * 64;
            int k_offset = i * k.size[1] * k.size[2] * k.size[3] + j * 64;

            attn.zeros();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, q.size[1],
                        k.size[1], q.size[3], 1, q.buff + q_offset, 512,
                        k.buff + k_offset, 512, 1, attn.buff, n_key);
            int ii, jj;
            for (ii = 0; ii < attn.buff_size; ii++) {
                attn.buff[ii] = attn.buff[ii] / 8;
            }

            for (ii = 0; ii < n_query; ii++) {
                int offset = ii * n_key;
                softmax(attn.buff + offset, mask->buff[i * n_query + ii],
                        n_key);
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, attn.size[2],
                        v.size[3], attn.size[3], 1, attn.buff, attn.size[3],
                        v.buff + k_offset, 512, 1, dout->buff + q_offset, 512);
        }
    }

    Tensor<float> linear_in(dout);
    int mm = dout->buff_size / 512;
    for (i = 0; i < mm; i++) {
        int offset = i * 512;
        memcpy(dout->buff + offset, params->linear_out_bias,
               sizeof(float) * 512);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mm, 512, 512, 1,
                linear_in.buff, 512, params->linear_out_weight, 512, 1,
                dout->buff, 512);
}
