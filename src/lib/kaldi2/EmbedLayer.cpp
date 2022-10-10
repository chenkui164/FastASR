#include "EmbedLayer.h"
#include "../util.h"
#include <cblas.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
using namespace kaldi2;

EmbedLayer::EmbedLayer(EncEmbedParams *params) : params(params)
{
}

EmbedLayer::~EmbedLayer()
{
}

void EmbedLayer::forward(Tensor<float> *&din)
{
    conv0_forward(din);
    conv1_forward(din);
    conv2_forward(din);
    linear_out_forward(din);
    basic_norm(din, *params->out_norm);
}

void EmbedLayer::get_conv_ind(int in_row, int in_column, int kernel, int stride,
                              int padding, int &out_row, int &out_column,
                              int *&out_idxs)
{

    out_row = (in_row - kernel + stride + 2 * padding) / stride;
    out_column = (in_column - kernel + stride + 2 * padding) / stride;
    out_idxs = (int *)malloc(sizeof(int) * out_row * out_column * 9);
    // printf("out_row is %d,out_column is %d\n", out_row, out_column);

    int m = in_row;
    int n = in_column;

    int idx = 0;
    int init_i = 0 - padding;
    int init_j = 0 - padding;
    int i, j;

    for (j = init_j; j <= m - kernel + padding; j = j + stride) {
        for (i = init_i; i <= n - kernel + padding; i = i + stride) {
            int *sub_idxs = out_idxs + 9 * idx;
            int tmp_ii = 0;
            for (int kj = 0; kj < kernel; kj++) {
                for (int ki = 0; ki < kernel; ki++) {
                    int sub_i = i + ki;
                    int sub_j = j + kj;
                    if ((sub_i >= 0) && (sub_i <= n - 1) && (sub_j >= 0) &&
                        (sub_j <= m - 1))
                        sub_idxs[tmp_ii] = sub_j * n + sub_i;
                    else
                        sub_idxs[tmp_ii] = -1;

                    tmp_ii = tmp_ii + 1;
                }
            }
            idx = idx + 1;
        }
    }
}

void EmbedLayer::conv0_forward(Tensor<float> *&din)
{

    int row = din->size[2];
    int column = din->size[3];

    int conv0_row, conv0_column;
    int *conv0_idxs;

    get_conv_ind(row, column, 3, 1, 1, conv0_row, conv0_column, conv0_idxs);
    int len = conv0_row * conv0_column * 9;

    int conv0_size = conv0_row * conv0_column;

    Tensor<float> blas_in(conv0_size, 9);
    Tensor<float> blas_out(conv0_size, 8);

    int i;
    for (i = 0; i < blas_in.buff_size; i++) {
        int ii = conv0_idxs[i];
        blas_in.buff[i] = (ii == -1) ? 0 : din->buff[ii];
    }

    delete din;
    din = new Tensor<float>(8, conv0_row, conv0_column);

    for (i = 0; i < conv0_size; i++) {
        int offset = i * 8;
        memcpy(blas_out.buff + offset, params->conv0_bias, 8 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, conv0_size, 8, 9, 1,
                blas_in.buff, 9, params->conv0_weight, 8, 1, blas_out.buff, 8);

    for (i = 0; i < blas_out.buff_size; i++) {
        int ii = i % 8;
        int jj = i / 8;
        int kk = ii * conv0_size + jj;
        float val = blas_out.buff[i];
        din->buff[kk] = val / (1 + exp(-val + 1));
    }

    free(conv0_idxs);
}

void EmbedLayer::conv1_forward(Tensor<float> *&din)
{

    int row = din->size[2];
    int column = din->size[3];

    int conv_row, conv_column;
    int *conv_idxs;

    get_conv_ind(row, column, 3, 2, 0, conv_row, conv_column, conv_idxs);
    int conv_size = conv_row * conv_column;

    Tensor<float> blas_in(conv_size, 9);
    Tensor<float> blas_out(conv_size, 32);

    int i;
    for (i = 0; i < conv_size; i++) {
        int offset = i * 32;
        memcpy(blas_out.buff + offset, params->conv1_bias, 32 * sizeof(float));
    }

    for (i = 0; i < 8; i++) {
        int in_offset = i * row * column;
        int weight_offset = i * 32 * 9;
        float *sub_conv_in = din->buff + in_offset;
        float *sub_weight = params->conv1_weight + weight_offset;

        int mm;
        for (mm = 0; mm < blas_in.buff_size; mm++) {
            int ii = conv_idxs[mm];
            blas_in.buff[mm] = sub_conv_in[ii];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, conv_size, 32, 9,
                    1, blas_in.buff, 9, sub_weight, 32, 1, blas_out.buff, 32);
    }

    delete din;
    din = new Tensor<float>(32, conv_row, conv_column);

    for (i = 0; i < blas_out.buff_size; i++) {
        int ii = i % 32;
        int jj = i / 32;
        int kk = ii * conv_size + jj;
        float val = blas_out.buff[i];
        din->buff[kk] = val / (1 + exp(-val + 1));
    }
}

void EmbedLayer::conv2_forward(Tensor<float> *&din)
{

    int row = din->size[2];
    int column = din->size[3];

    int conv_row, conv_column;
    int *conv_idxs;

    get_conv_ind(row, column, 3, 2, 0, conv_row, conv_column, conv_idxs);
    int conv_size = conv_row * conv_column;

    Tensor<float> blas_in(conv_size, 9);
    Tensor<float> blas_out(conv_size, 128);

    int i;
    for (i = 0; i < conv_size; i++) {
        int offset = i * 128;
        memcpy(blas_out.buff + offset, params->conv2_bias, 128 * sizeof(float));
    }

    for (i = 0; i < 32; i++) {
        int in_offset = i * row * column;
        int weight_offset = i * 128 * 9;
        float *sub_conv_in = din->buff + in_offset;
        float *sub_weight = params->conv2_weight + weight_offset;

        int mm;
        for (mm = 0; mm < blas_in.buff_size; mm++) {
            int ii = conv_idxs[mm];
            blas_in.buff[mm] = sub_conv_in[ii];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, conv_size, 128,
                    9, 1, blas_in.buff, 9, sub_weight, 128, 1, blas_out.buff,
                    128);
    }

    delete din;
    din = new Tensor<float>(conv_row, 128, conv_column);

    for (i = 0; i < blas_out.buff_size; i++) {
        int ii = i / (128 * 19);
        int jj = (i >> 7) % 19;
        int kk = i & 0x7f;
        int hh = ii * 128 * 19 + kk * 19 + jj;
        float val = blas_out.buff[i];
        din->buff[hh] = val / (1 + exp(-val + 1));
    }
}

void EmbedLayer::linear_out_forward(Tensor<float> *&din)
{

    int Tmax = din->size[1];

    Tensor<float> *dout = new Tensor<float>(Tmax, 512);

    int i;
    for (i = 0; i < Tmax; i++) {
        int offset = i * 512;
        memcpy(dout->buff + offset, params->out_bias, 512 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Tmax, 512, 19 * 128,
                1, din->buff, 19 * 128, params->out_weight, 512, 1, dout->buff,
                512);
    delete din;
    din = dout;
}

void EmbedLayer::norm_forward(Tensor<float> *&din)
{

    int Tmax = din->size[2];

    int i, j;
    for (i = 0; i < Tmax; i++) {
        float sum = 0;
        for (j = 0; j < 512; j++) {
            int ii = i * 512 + j;
            sum += din->buff[ii] * din->buff[ii];
        }
        float mean = sqrt(sum / 512 + *params->out_norm);
        for (j = 0; j < 512; j++) {
            int ii = i * 512 + j;
            din->buff[ii] = din->buff[ii] / mean;
        }
    }
}
