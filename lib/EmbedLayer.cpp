#include "EmbedLayer.h"
#include "util.h"
#include <cblas.h>
#include <iostream>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

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
    linear_out_forward(din);
}

void EmbedLayer::get_conv_ind(int trans, int in_row, int in_column, int kernel,
                              int stride, int &out_row, int &out_column,
                              int *&out_idxs)
{

    out_row = (in_row - kernel + stride) / stride;
    out_column = (in_column - kernel + stride) / stride;
    if (trans) {
        int tmp = out_row;
        out_row = out_column;
        out_column = tmp;
    }
    out_idxs = (int *)malloc(sizeof(int) * out_row * out_column * 9);
    int i, j;
    int mm = 0;
    if (trans) {
        for (i = 0; i < in_column - kernel + 1; i = i + stride) {
            for (j = 0; j < in_row - kernel + 1; j = j + stride) {
                int nn1 = (j * in_column + i);
                int nn2 = nn1 + in_column;
                int nn3 = nn2 + in_column;

                out_idxs[mm++] = nn1++;
                out_idxs[mm++] = nn2++;
                out_idxs[mm++] = nn3++;

                out_idxs[mm++] = nn1++;
                out_idxs[mm++] = nn2++;
                out_idxs[mm++] = nn3++;

                out_idxs[mm++] = nn1++;
                out_idxs[mm++] = nn2++;
                out_idxs[mm++] = nn3++;
            }
        }
    } else {
        for (i = 0; i < in_row - kernel + 1; i = i + stride) {
            for (j = 0; j < in_column - kernel + 1; j = j + stride) {
                int nn = (i * in_column + j);
                out_idxs[mm++] = nn;
                out_idxs[mm++] = nn + 1;
                out_idxs[mm++] = nn + 2;

                nn = nn + in_column;
                out_idxs[mm++] = nn;
                out_idxs[mm++] = nn + 1;
                out_idxs[mm++] = nn + 2;

                nn = nn + in_column;
                out_idxs[mm++] = nn;
                out_idxs[mm++] = nn + 1;
                out_idxs[mm++] = nn + 2;
            }
        }
    }
}

void EmbedLayer::conv0_forward(Tensor<float> *&din)
{

    int row = din->size[2];
    int column = din->size[3];

    int conv0_row, conv0_column;
    int *conv0_idxs;

    get_conv_ind(1, row, column, 3, 2, conv0_row, conv0_column, conv0_idxs);

    int conv0_size = conv0_row * conv0_column;

    Tensor<float> blas_in(conv0_size, 9);
    Tensor<float> blas_out(conv0_size, 512);

    int i;
    for (i = 0; i < blas_in.buff_size; i++) {
        int ii = conv0_idxs[i];
        blas_in.buff[i] = din->buff[ii];
    }

    delete din;
    din = new Tensor<float>(512, conv0_row, conv0_column);

    for (i = 0; i < conv0_size; i++) {
        int offset = i * 512;
        memcpy(blas_out.buff + offset, params->conv0_bias, 512 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, conv0_size, 512, 9,
                1, blas_in.buff, 9, params->conv0_weight, 512, 1, blas_out.buff,
                512);

    for (i = 0; i < blas_out.buff_size; i++) {
        int ii = i % 512;
        int jj = i / 512;
        int kk = ii * conv0_size + jj;
        din->buff[kk] = blas_out.buff[i] > 0 ? blas_out.buff[i] : 0;
    }

    free(conv0_idxs);
}

void EmbedLayer::conv1_forward(Tensor<float> *&din)
{

    int row = din->size[2];
    int column = din->size[3];

    int conv_row, conv_column;
    int *conv_idxs;

    get_conv_ind(0, row, column, 3, 2, conv_row, conv_column, conv_idxs);
    int conv_size = conv_row * conv_column;

    Tensor<float> blas_in(conv_size, 9);
    Tensor<float> blas_out(conv_size, 512);

    int i;

    for (i = 0; i < conv_size; i++) {
        int offset = i * 512;
        memcpy(blas_out.buff + offset, params->conv1_bias, 512 * sizeof(float));
    }

    for (i = 0; i < 512; i++) {
        int in_offset = i * row * column;
        int weight_offset = i * 512 * 9;
        float *sub_conv_in = din->buff + in_offset;
        float *sub_weight = params->conv1_weight + weight_offset;

        int mm;
        for (mm = 0; mm < blas_in.buff_size; mm++) {
            int ii = conv_idxs[mm];
            blas_in.buff[mm] = sub_conv_in[ii];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, conv_size, 512,
                    9, 1, blas_in.buff, 9, sub_weight, 512, 1, blas_out.buff,
                    512);
    }

    delete din;
    din = new Tensor<float>(512, conv_row, conv_column);

    for (i = 0; i < blas_out.buff_size; i++) {
        int ii = i / (conv_column * 512);
        int jj = (i % (conv_column * 512)) / 512;
        int kk = i % 512;
        int mm = jj * conv_row * 512 + conv_row * kk + ii;
        din->buff[mm] = blas_out.buff[i] > 0 ? blas_out.buff[i] : 0;
    }
}

void EmbedLayer::linear_out_forward(Tensor<float> *&din)
{

    int Tmax = din->size[3];
    int Fmax = din->size[2];

    int tmp = 0x41b504f3;
    float scale = *((float *)&tmp);

    Tensor<float> *dout = new Tensor<float>(Tmax, 512);

    int i;
    for (i = 0; i < Tmax; i++) {
        int offset = i * 512;
        memcpy(dout->buff + offset, params->out0_bias, 512 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Tmax, 512,
                512 * Fmax, scale, din->buff, 512 * Fmax, params->out0_weight,
                512, scale, dout->buff, 512);
    delete din;
    din = dout;
}
