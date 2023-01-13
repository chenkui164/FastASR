#include <cblas.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../Tensor.h"
#include "../util.h"
#include "Encoder.h"

using namespace std;
using namespace paraformer;

Encoder::Encoder(EncoderParams *params) : params(params)
{
    conv_im2col = NULL;
    encoders0 = new SubEncoder(&params->sub_encoders0, 560);
    int i;
    for (i = 0; i < 49; i++) {
        encoders[i] = new SubEncoder(&params->sub_encoders[i], 512);
    }
    after_norm = new LayerNorm(&params->after_norm, 1e-12, 512);
}

Encoder::~Encoder()
{

    free(conv_im2col);
    int i;
    for (i = 0; i < 49; i++) {
        delete encoders[i];
    }
}

void Encoder::reset()
{
}

void Encoder::get_poscode(Tensor<float> *poscode)
{
    int mm = poscode->size[2];

    int i;
    float scale = -0.0330119726594128;
    for (i = 0; i < 280; i++) {
        float tmptime = exp(i * scale);
        int j;
        for (j = 0; j < mm; j++) {
            int sin_idx = j * 560 + i;
            int cos_idx = j * 560 + i + 280;
            float coe = tmptime * (j + 1);
            poscode->buff[sin_idx] = sin(coe);
            poscode->buff[cos_idx] = cos(coe);
        }
    }
}

void Encoder::get_conv_im2col(int mm)
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

void Encoder::disp_conv_im2col(int mm)
{
    int i, j;
    for (i = 0; i < mm; i++) {
        for (j = 0; j < 11; j++) {
            int idx = i * 11 + j;
            printf("%d,\t", conv_im2col[idx]);
        }
        printf("\n");
    }
}

void Encoder::forward(Tensor<float> *&din)
{
    int mm = din->size[2];
    Tensor<float> poscode(mm, 560);
    get_poscode(&poscode);

    din->add(&poscode);
    get_conv_im2col(mm);
    encoders0->forward(din, conv_im2col);
    int i;
    for (i = 0; i < 49; i++) {
        // printf("i is %d\n", i);
        encoders[i]->forward(din, conv_im2col);
        // din->disp();
    }
    after_norm->forward(din);

    // disp_conv_im2col(mm);

    // int Tmax = din->size[2];
    // Tensor<float> *pe_code;
    // pos_enc->fetch(Tmax, pe_code);
    // int i;
    // for (i = 0; i < 12; i++) {
    //     subencoder[i]->forward(din, pe_code);
    // }
    // delete pe_code;
}
