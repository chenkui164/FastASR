#include <math.h>
#include <stdio.h>

#include "LayerNorm.h"

using namespace paraformer;

LayerNorm::LayerNorm(NormParams *params, float error, int layer_size)
    : params(params), error(error), layer_size(layer_size)
{
}

LayerNorm::~LayerNorm()
{
}

void LayerNorm::mean_var(float *din, float &mean, float &var)
{
    int i;
    mean = 0;
    for (i = 0; i < layer_size; i++) {
        mean += din[i];
    }
    mean = mean / layer_size;

    var = 0;
    for (i = 0; i < layer_size; i++) {
        float tmp = din[i] - mean;
        var += tmp * tmp;
    }
    var = var / layer_size;
}

void LayerNorm::norm(float *din, float mean, float var)
{
    int i;
    float dd = sqrt(var + error);
    for (i = 0; i < layer_size; i++) {
        din[i] = (din[i] - mean) / dd;
        din[i] = din[i] * params->weight[i] + params->bias[i];
    }
}

void LayerNorm::forward(Tensor<float> *din)
{
    // return;
    int mm = din->buff_size / layer_size;
    int i;
    for (i = 0; i < mm; i++) {
        int offset = i * layer_size;
        float mean, var;
        float *buff = din->buff + offset;
        mean_var(buff, mean, var);
        norm(buff, mean, var);
    }
}
