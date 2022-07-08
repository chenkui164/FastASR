#include <deque>
#include <fftw3.h>
#include <iostream>
#include <locale.h>
#include <malloc.h>
#include <math.h>
#include <string.h>

#include "CTCDecode.h"
#include "EmbedLayer.h"
#include "Encoder.h"
#include "Model.h"
#include "Tensor.h"
#include "Vocab.h"
#include "predefine_coe.h"
#include "util.h"

using namespace std;

void disp_params(float *din, int size)
{
    int i;
    for (i = 0; i < size; i++) {
        printf("%f ", din[i]);
    }
    printf("\n");
}
Model::Model()
{
    loadparams("wenet_params.bin");

    vocab = new Vocab("vocab.txt");
    pos_enc = new PositionEncoding(5000);
    encoder = new Encoder(&params->encoder, pos_enc);
    ctc = new CTCdecode(params->ctc_weight, params->ctc_bias);
    decoder = new Decoder(&params->decoder, pos_enc);
    // disp_params(params->decoder.sub_decoder[0].self_attn.linear_q_bias, 10);
}

Model::~Model()
{
    delete encoder;
    delete ctc;
}

void hyps_process(deque<PathProb> hyps, Tensor<float> *din,
                  Tensor<int> *&hyps_pad, Tensor<int> *&hyps_mask,
                  Tensor<float> *&encoder_out, Tensor<int> *&encoder_mask)
{
    int mm = hyps.size();

    int i = 0;
    int j = 0;
    int max = 0;
    for (auto hyps_it = hyps.begin(); hyps_it != hyps.end(); hyps_it++) {
        int ll = hyps_it->prefix.size() + 1;
        max = max > ll ? max : ll;
        i++;
    }

    hyps_pad = new Tensor<int>(mm, max);
    hyps_mask = new Tensor<int>(mm, max);

    encoder_out = new Tensor<float>(mm, din->size[2], din->size[3]);

    i = 0;
    for (auto hyps_it = hyps.begin(); hyps_it != hyps.end(); hyps_it++) {
        int ll = hyps_it->prefix.size() + 1;
        hyps_pad->buff[i * max] = 5536;
        hyps_mask->buff[i * max] = 1;
        for (j = 1; j < max; j++) {
            int ii = i * max + j;
            hyps_pad->buff[ii] = j < ll ? hyps_it->prefix[j - 1] : 5536;
            hyps_mask->buff[ii] = j < ll ? j + 1 : ll;
        }

        int offset = i * din->buff_size;
        memcpy(encoder_out->buff + offset, din->buff,
               sizeof(float) * din->buff_size);
        i++;
    }

    encoder_mask = new Tensor<int>(mm, max);
    for (i = 0; i < encoder_mask->buff_size; i++) {
        encoder_mask->buff[i] = din->size[2];
    }
}

void rescoring(deque<PathProb> hyps, Tensor<float> *decoder_out,
               Tensor<float> *scorce)
{

    int i = 0;
    scorce->zeros();
    for (auto hyps_it = hyps.begin(); hyps_it != hyps.end(); hyps_it++) {
        int j;
        int ll = hyps_it->prefix.size();
        float *prob;

        for (j = 0; j < ll; j++) {
            int char_idx = hyps_it->prefix[j];
            prob = decoder_out->buff +
                   (i * decoder_out->buff_size / 10 + j * 5537);
            scorce->buff[i] = scorce->buff[i] + prob[char_idx];
        }

        prob = decoder_out->buff + (i * decoder_out->buff_size / 10 + j * 5537);
        scorce->buff[i] = scorce->buff[i] + prob[5536];
        i++;
    }
}

string Model::forward(Tensor<float> *din, Tensor<float> *dout)
{
    encoder->forward(din, dout);
    deque<PathProb> hyps;
    ctc->forward(dout, hyps);
    // ctc->show_hyps(hyps);
    Tensor<int> *hyps_pad;
    Tensor<int> *hyps_mask;
    Tensor<float> *encoder_out;
    Tensor<int> *encoder_mask;
    hyps_process(hyps, dout, hyps_pad, hyps_mask, encoder_out, encoder_mask);
    // printf("hyps size is %ld\n", hyps.size());
    Tensor<float> *decoder_out;
    decoder->forward(hyps_pad, hyps_mask, encoder_out, encoder_mask,
                     decoder_out);

    Tensor<float> scorce(1, 10);
    rescoring(hyps, decoder_out, &scorce);

    int i = 0;
    float max = -INFINITY;
    vector<int> result;
    for (auto hyps_it = hyps.begin(); hyps_it != hyps.end(); hyps_it++) {
        float tmp_scorce = 0.5 * hyps_it->prob + scorce.buff[i];
        if (tmp_scorce > max) {
            max = tmp_scorce;
            result = hyps_it->prefix;
        }
        scorce.buff[i] = tmp_scorce;
        i++;
    }
    // cout << vocab->vector2string(result) << endl;
    return vocab->vector2string(result);
}

void Model::loadparams(const char *filename)
{

    FILE *fp;
    fp = fopen(filename, "rb");
    fseek(fp, 0, SEEK_END);
    uint32_t nFileLen = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    params = (WenetParams *)memalign(32, nFileLen);
    int n = fread(params, 1, nFileLen, fp);

    fclose(fp);
}
