#include <deque>
#include <fftw3.h>
#include <iostream>
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
using namespace kaldi2;

ModelImp::ModelImp(const char *path, int mode)
{
    string wenet_path = pathAppend(path, "wenet_param.bin");
    string vocab_path = pathAppend(path, "vocab.txt");
    vocab = new Vocab(vocab_path.c_str());

    fe = new FeatureExtract(mode);

    cout << wenet_path << endl;
    p_helper = new ModelParamsHelper(wenet_path.c_str(), 500);
    disp_params(
        p_helper->params.encoder.sub_encoder[0].feedforward_macaron.w1_bias,
        10);

    pos_enc = new PositionEncoding(5000);
    encoder = new Encoder(&p_helper->params.encoder, pos_enc, mode);
    joiner = new Joiner(&p_helper->params.joiner);
    decoder = new Decoder(&p_helper->params.decoder, 5537);
}

ModelImp::~ModelImp()
{
}

void ModelImp::reset()
{
}

string ModelImp::greedy_search(Tensor<float> *&encoder_out)
{
    joiner->encoder_forward(encoder_out);
    vector<int> hyps = {0, 0};
    hyps.reserve(200);

    Tensor<float> *decoder_out;
    int *hyps_last = &hyps.back() - 1;

    decoder->forward(hyps_last, decoder_out);
    joiner->decoder_forward(decoder_out);

    int Tmax = encoder_out->size[2];
    int i;
    for (i = 0; i < Tmax; i++) {
        float *sub_encoder_out = encoder_out->buff + i * 512;
        float *sub_decoder_out = decoder_out->buff;
        Tensor<float> logit(1, 5537);
        joiner->linear_forward(sub_encoder_out, sub_decoder_out, &logit);
        float max_val;
        int max_idx;
        findmax(logit.buff, 5537, max_val, max_idx);
        if (max_idx != 0) {
            hyps.push_back(max_idx);
            hyps_last = &hyps.back() - 1;
            decoder->forward(hyps_last, decoder_out);
            joiner->decoder_forward(decoder_out);
        }
    }

    hyps.erase(hyps.begin());
    hyps.erase(hyps.begin());

    return vocab->vector2string(hyps);
}

string ModelImp::forward(float *din, int len, int flag)
{
    Tensor<float> *in;
    fe->insert(din, len, flag);
    fe->fetch(in);

    encoder->forward(in);
    string info = greedy_search(in);

    return info;
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
