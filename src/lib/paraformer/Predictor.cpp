
#include <cblas.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../Tensor.h"
#include "../util.h"
#include "ModelParams.h"
#include "Predictor.h"

using namespace std;
using namespace paraformer;

Predictor::Predictor(PredictorParams *params) : params(params)
{
}

Predictor::~Predictor()
{
}

void Predictor::get_conv_im2col(int mm)
{

    int idxs_size = mm * 3;
    conv_im2col = (int *)malloc(sizeof(int) * idxs_size);
    int step = 512;
    int i, j;
    int ii = 0;
    for (i = 0; i < mm; i++) {
        int start_idx = -1 + i;
        for (j = 0; j < 3; j++) {
            int val = start_idx + j;
            if (val >= 0 && val < mm)
                conv_im2col[ii++] = val * step;
            else
                conv_im2col[ii++] = -1;
        }
    }
}

void Predictor::cif_conv1d(Tensor<float> *&din)
{
    int mm = din->size[2];
    int v_offset = 0;

    Tensor<float> blasin(mm, 3);
    Tensor<float> *blasout = new Tensor<float>(mm, 512);
    int i, j;

    for (i = 0; i < mm; i++) {
        int offset = i * 512;
        memcpy(blasout->buff + offset, params->cif_conv1d_bias,
               sizeof(float) * 512);
    }

    for (i = 0; i < 512; i++) {
        for (j = 0; j < mm * 3; j++) {
            int tmp_idx = conv_im2col[j];
            if (tmp_idx == -1)
                blasin.buff[j] = 0;
            else
                blasin.buff[j] = din->buff[tmp_idx + v_offset];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 512, 3, 1,
                    blasin.buff, 3, params->cif_conv1d_weight + i * 512 * 3, 3,
                    1, blasout->buff, 512);

        v_offset++;
    }

    delete din;
    din = blasout;
}

void Predictor::disp_conv_im2col(int mm)
{
    int i, j;
    for (i = 0; i < mm; i++) {
        for (j = 0; j < 3; j++) {
            int idx = i * 3 + j;
            printf("%d,\t", conv_im2col[idx]);
        }
        printf("\n");
    }
}

void Predictor::forward(Tensor<float> *&din)
{
    int mm = din->size[2];
    int nn = din->size[3];

    Tensor<float> alphas(mm, 1);
    Tensor<float> hidden(din);

    get_conv_im2col(mm);
    cif_conv1d(din);
    relu(din);

    int i, j;
    int offset = 0;
    for (i = 0; i < mm; i++) {
        alphas.buff[i] = params->cif_output_bias[0];
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 1, 512, 1,
                din->buff, 512, params->cif_output_weight, 512, 1, alphas.buff,
                1);

    sigmoid(&alphas);

    float fires[mm];
    float intergrate = 0;
    Tensor<float> frame(512);
    frame.zeros();

    Tensor<float> frames(mm, 512);

    float sum_alphas = 0;
    for (i = 0; i < mm; i++) {
        float alpha = alphas.buff[i];
        float distribution_completion = 1 - intergrate;
        sum_alphas += alpha;
        intergrate = intergrate + alpha;
        fires[i] = intergrate;
        float cur = alpha;

        int fire_place = (intergrate >= 1);
        if (fire_place) {
            intergrate = intergrate - 1;
            cur = distribution_completion;
        }

        for (j = 0; j < 512; j++) {
            int hidx = i * 512 + j;
            frame.buff[j] += cur * hidden.buff[hidx];
        }
        int foffset = i * 512;
        memcpy(frames.buff + foffset, frame.buff, 512 * sizeof(float));
        if (fire_place) {
            float remainds = alpha - cur;
            for (j = 0; j < 512; j++) {
                int hidx = i * 512 + j;
                frame.buff[j] = remainds * hidden.buff[hidx];
            }
        }
    }
    // printf("sum_alphas is %f\n", sum_alphas);

    int len_labels = round(sum_alphas + 0.45);
    // printf("len_labels is %d\n", len_labels);

    Tensor<float> *tout = new Tensor<float>(len_labels, 512);

    offset = 0;
    for (i = 0; i < mm; i++) {
        if (fires[i] >= 1) {
            memcpy(tout->buff + offset, frames.buff + i * 512,
                   512 * sizeof(float));
            offset += 512;
        }
    }
    delete din;
    din = tout;
}
