#include <deque>
#include <fftw3.h>
#include <iostream>
#include <list>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "../Tensor.h"
#include "../Vocab.h"
#include "../predefine_coe.h"
#include "../util.h"
#include "ModelImp.h"

using namespace std;
using namespace paraformer;

ModelImp::ModelImp(const char *path, int mode)
{
    string wenet_path = pathAppend(path, "wenet_params.bin");
    string vocab_path = pathAppend(path, "vocab.txt");

    fe = new FeatureExtract(mode);

    p_helper = new ModelParamsHelper(wenet_path.c_str(), 500);

    encoder = new Encoder(&p_helper->params.encoder);
    predictor = new Predictor(&p_helper->params.predictor);
    decoder = new Decoder(&p_helper->params.decoder);

    vocab = new Vocab(vocab_path.c_str());
}

ModelImp::~ModelImp()
{
}

void ModelImp::reset()
{
    fe->reset();
}

void ModelImp::apply_lfr(Tensor<float> *&din)
{
    int mm = din->size[2];
    int ll = ceil(mm / 6.0);
    Tensor<float> *tmp = new Tensor<float>(ll, 560);
    int i, j;
    int out_offset = 0;
    for (i = 0; i < ll; i++) {
        for (j = 0; j < 7; j++) {
            int idx = i * 6 + j - 3;
            if (idx < 0) {
                idx = 0;
            }
            if (idx >= mm) {
                idx = mm - 1;
            }
            memcpy(tmp->buff + out_offset, din->buff + idx * 80,
                   sizeof(float) * 80);
            out_offset += 80;
        }
    }
    delete din;
    din = tmp;
}

void ModelImp::apply_cmvn(Tensor<float> *din)
{
    const float *var;
    const float *mean;
    float scale = 22.6274169979695;
    int m = din->size[2];
    int n = din->size[3];

    var = (const float *)paraformer_cmvn_var_hex;
    mean = (const float *)paraformer_cmvn_mean_hex;
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            int idx = i * n + j;
            din->buff[idx] = (din->buff[idx] + mean[j]) * var[j];
        }
    }
}

string ModelImp::greedy_search(Tensor<float> *&encoder_out)
{
    vector<int> hyps;
    int Tmax = encoder_out->size[2];
    int i;
    for (i = 0; i < Tmax; i++) {
        int max_idx;
        float max_val;
        findmax(encoder_out->buff + i * 8404, 8404, max_val, max_idx);
        hyps.push_back(max_idx);
    }

    return vocab->vector2stringV2(hyps);
}

string ModelImp::forward(float *din, int len, int flag)
{

    Tensor<float> *in;
    fe->insert(din, len, flag);
    fe->fetch(in);
    apply_lfr(in);
    apply_cmvn(in);
    encoder->forward(in);
    Tensor<float> enc_out(in);
    predictor->forward(in);
    decoder->forward(in, &enc_out);
    string result = greedy_search(in);

    return result;
}

string ModelImp::forward_chunk(float *din, int len, int flag)
{

    printf("Not Imp!!!!!!\n");
    return "Hello";
}

string ModelImp::rescoring()
{
    printf("Not Imp!!!!!!\n");
    return "Hello";
}
