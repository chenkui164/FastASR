
#ifndef AUDIO_H
#define AUDIO_H


#include <stdint.h>

#include "Tensor.h"

class Audio {
  private:
    int16_t *speech;
    int speech_len;
    int16_t sample_rate;

    void loadwav(const char *filename);
    void audio2feature();
    void melspect(float *din, float *dout);
    void global_cmvn(float *din);

  public:
    Tensor<float> *fbank_feature;
    Audio(const char *filename);
    ~Audio();
};

#endif
