#include "EmbedLayer.h"
#include "../util.h"
#include <cblas.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
using namespace paraformer;

EmbedLayer::EmbedLayer()
{
}

EmbedLayer::~EmbedLayer()
{
}

void EmbedLayer::forward(Tensor<float> *din)
{
}

void EmbedLayer::get_poscode(int nn)
{
    Tensor<float> poscode(nn, 560);
    int i;
    float scale = -0.0330119726594128;
    for (i = 0; i < 280; i++) {
        float tmptime = exp(i * scale);
        int j;
        for (j = 0; j < nn; j++) {
            int sin_idx = j * 560 + i;
            int cos_idx = j * 560 + i + 280;
            poscode.buff[sin_idx] = sin(tmptime);
            poscode.buff[cos_idx] = cos(tmptime);
        }
    }
}
