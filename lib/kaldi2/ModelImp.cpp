#include <deque>
#include <fftw3.h>
#include <iostream>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

    fe = new FeatureExtract(mode);

    cout << wenet_path << endl;
    p_helper = new ModelParamsHelper(wenet_path.c_str(), 500);

    pos_enc = new PositionEncoding(5000);
    encoder = new Encoder(&p_helper->params.encoder, pos_enc, mode);

    printf("Not Imp!!!!!!\n");
}

ModelImp::~ModelImp()
{
    printf("Not Imp!!!!!!\n");
}

void ModelImp::reset()
{
    printf("Not Imp!!!!!!\n");
}

string ModelImp::forward(float *din, int len, int flag)
{
    Tensor<float> *in;
    fe->insert(din, len, flag);
    fe->fetch(in);
    // in->shape();
    // in->dump();

    encoder->forward(in);

    return "Hello";
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
